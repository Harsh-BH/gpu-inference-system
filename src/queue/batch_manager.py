"""Dynamic batching: turning N independent requests into one GPU call (Phase 11).

THE TRADE, STATED PRECISELY

    Phase 4 measured it. At batch 1 this engine does 331 img/s at 2.50 ms p50;
    at batch 16, 703 img/s at 23.01 ms. Batching buys throughput and charges
    latency, and the exchange rate is not linear.

    A batch is dispatched when EITHER:
      - MAX_BATCH_SIZE requests are available, or
      - the OLDEST waiting request has waited MAX_BATCH_WAIT_MS.

    The second condition is what makes it safe. Without a deadline, a batch
    that never fills waits forever, and under light traffic every request pays
    unbounded latency to serve a batch that will never arrive. The timer is
    started from the first request in the batch, not reset by later arrivals --
    otherwise a steady trickle could postpone dispatch indefinitely.

WHY THE WAIT IS CHARGED TO EVERY REQUEST

    MAX_BATCH_WAIT_MS is added to the p99 of an idle system for nothing: with
    one request in flight there is no one to batch with, and it waits anyway.
    That is why the default is 5 ms rather than 50. Under load the queue is
    never empty, batches fill before the deadline, and the wait costs nothing.

    Tuning this is tuning the latency SLO. Longer wait, fuller batches, better
    throughput, worse tail.

OUTPUT MAPPING IS THE CORRECTNESS-CRITICAL PART

    The engine returns an (N, num_classes) tensor. Row i must go to the request
    that supplied row i. Get this wrong and every client receives a plausible,
    confident prediction belonging to somebody else -- no exception, no log
    line, no test failure unless someone wrote one. tests/test_batching.py
    exists mostly for this.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.inference.base import EngineError, InferenceEngine
from src.queue.request_queue import InferenceRequest, RequestQueue

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """What one dispatched batch cost. Consumed by the metrics layer."""

    batch_size: int
    formation_ms: float  # time spent assembling, i.e. what the wait bought
    inference_ms: float
    h2d_ms: float
    compute_ms: float
    d2h_ms: float


class BatchManager:
    """Owns the single inference loop.

    One loop, one engine, one GPU. Deliberately not a pool: a second worker
    sharing one engine would serialise on the same CUDA context anyway, and
    would make the memory ceiling harder to reason about. Scaling past one GPU
    is model replication across processes, not threads in this one.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        queue: RequestQueue,
        *,
        max_batch_size: int,
        max_batch_wait_ms: float,
        on_batch: callable | None = None,
    ) -> None:
        self._engine = engine
        self._queue = queue
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_batch_wait_ms / 1000.0
        self._on_batch = on_batch
        self._task: asyncio.Task | None = None
        self._running = False
        self.batches_dispatched = 0
        self.images_processed = 0

    # --- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="batch-manager")

    async def stop(self, drain_timeout: float = 5.0) -> None:
        """Stop accepting, finish what is in flight, fail the rest cleanly.

        A shutdown that just cancels leaves every waiting client hanging until
        their own timeout. Draining is the difference between a graceful stop
        and a rude one.
        """
        self._running = False
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=drain_timeout)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        finally:
            self._task = None

        for request in self._queue.drain():
            request.fail(EngineError("server is shutting down"))

    # --- the loop -------------------------------------------------------

    async def _run(self) -> None:
        while self._running:
            try:
                batch = await self._collect()
            except TimeoutError:
                continue
            if not batch:
                continue
            await self._dispatch(batch)

    async def _collect(self) -> list[InferenceRequest]:
        """Assemble a batch under the size-or-deadline rule."""
        try:
            # Poll rather than block forever, so stop() is observed promptly.
            first = await asyncio.wait_for(self._queue.get(), timeout=0.05)
        except TimeoutError:
            return []

        batch = [first]
        deadline = perf_counter() + self._max_wait_s

        while len(batch) < self._max_batch_size:
            # Anything already waiting joins for free -- no reason to sleep
            # when the queue is non-empty.
            queued = self._queue.get_nowait()
            if queued is not None:
                batch.append(queued)
                continue

            remaining = deadline - perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        return batch

    async def _dispatch(self, batch: list[InferenceRequest]) -> None:
        formation_ms = max(r.queue_wait_ms for r in batch)
        stacked = np.stack([r.tensor for r in batch], axis=0)

        started = perf_counter()
        try:
            # predict() blocks: it synchronises on CUDA. Running it directly
            # here would stall the event loop for the whole forward pass, so
            # nothing else -- health checks, new arrivals, metrics -- could be
            # served meanwhile. to_thread keeps the loop responsive.
            result = await asyncio.to_thread(self._engine.predict, stacked)
        except EngineError as exc:
            # One bad batch must not kill the loop or the process. Every
            # request in it fails; the next batch is unaffected.
            logger.warning("batch of %d failed: %s", len(batch), exc)
            for request in batch:
                request.fail(exc)
            return
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            logger.exception("unexpected error in batch of %d", len(batch))
            for request in batch:
                request.fail(EngineError(f"inference failed: {exc}"))
            return
        inference_ms = (perf_counter() - started) * 1000.0

        logits = result.logits
        if logits.shape[0] != len(batch):
            # Cannot map rows to requests, so nobody gets a wrong answer.
            error = EngineError(f"engine returned {logits.shape[0]} rows for {len(batch)} requests")
            for request in batch:
                request.fail(error)
            return

        # Row i belongs to request i. This is the line the whole design
        # protects; np.stack preserved the order and nothing has reordered.
        for i, request in enumerate(batch):
            request.resolve(logits[i])

        self.batches_dispatched += 1
        self.images_processed += len(batch)
        if self._on_batch is not None:
            self._on_batch(
                BatchOutcome(
                    batch_size=len(batch),
                    formation_ms=formation_ms,
                    inference_ms=inference_ms,
                    h2d_ms=result.timings.h2d_ms,
                    compute_ms=result.timings.compute_ms,
                    d2h_ms=result.timings.d2h_ms,
                )
            )
