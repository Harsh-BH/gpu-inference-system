"""The runner: turns a list of stages into a working, observable pipeline.

WHAT THIS OWNS, SO NO STAGE HAS TO

    queues, workers, thread pools, batch formation, deadlines, backpressure,
    row-to-job mapping, per-stage timing, failure isolation, and an ordered
    startup and drain. A stage author writes `process()`.

    Every one of those was previously written by hand in `src/queue/` for
    exactly one stage. Written once here, a new stage gets all of it for free
    and cannot implement any of it wrongly.

THE SHAPE

    submit() -> [queue 0] -> stage 0 workers -> [queue 1] -> stage 1 workers
                                                          -> ... -> Completion

    One bounded queue in front of every stage. Between stages, a full queue
    makes the upstream worker *wait* -- that is backpressure, and it propagates
    all the way to queue 0, where it becomes a 503 instead of unbounded memory
    growth. Only queue 0 rejects; the rest block. That asymmetry is deliberate:
    dropping a request that has already been decoded wastes the most expensive
    work the system does.

WHY ASYNCIO TASKS DRIVING THREAD POOLS

    The transformations are synchronous and mostly release the GIL (PIL during
    decode and resize, CUDA during a forward pass). asyncio alone would run
    them on the event loop and stall every other request for the duration;
    threads alone would need their own queueing and would not compose with
    FastAPI. So: one asyncio task per worker, each awaiting its stage's thread
    pool. The loop stays free to accept requests while N decodes and one
    forward pass are in flight.

    `workers=0` opts out of the thread hop for stages too cheap to justify it.

BATCH FORMATION, AND THE DEADLINE THAT MAKES IT SAFE

    A batch dispatches when EITHER `max_batch` items are in hand OR the OLDEST
    item has waited `max_batch_wait_ms`. Without the deadline a batch that
    never fills waits forever. The timer starts with the first item and is
    never reset by later arrivals -- otherwise a steady trickle postpones
    dispatch indefinitely.

SHUTDOWN IS A DRAIN, NOT AN EXIT

    stop() refuses new work, lets every worker finish what its queue already
    holds, then fails whatever remains with a clean error. A shutdown that
    simply cancels leaves every waiting client hanging until their own timeout
    expires. Stages are torn down in reverse order, after the last worker has
    stopped touching them.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

from src.pipeline.stage import (
    Completion,
    Job,
    PipelineError,
    PipelineFull,
    PipelineNotRunning,
    StageContractError,
    StageReport,
    StageSpec,
)

logger = logging.getLogger(__name__)

#: How often an idle worker re-checks whether the pipeline is stopping. Short
#: enough that shutdown feels immediate, long enough to cost nothing.
_POLL_S = 0.05


class Pipeline:
    """An ordered sequence of stages, run concurrently.

    Assembly is declarative and total -- everything about how this pipeline
    behaves is in the list you pass:

        pipeline = Pipeline(
            [
                StageSpec(DecodeStage(...), workers=8),
                StageSpec(InferStage(...), max_batch=16, max_batch_wait_ms=5.0),
                StageSpec(RankStage(...), workers=0, max_batch=64),
            ],
            on_stage=metrics.record,
        )
    """

    def __init__(
        self,
        specs: Sequence[StageSpec],
        *,
        on_stage: Callable[[StageReport], None] | None = None,
    ) -> None:
        if not specs:
            raise ValueError("a pipeline needs at least one stage")
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            # Names key the metrics labels and the latency breakdown, so
            # duplicates would silently merge two stages into one number.
            raise ValueError(f"stage names must be unique, got {names}")

        self._specs = list(specs)
        self._on_stage = on_stage
        self._queues: list[asyncio.Queue[Job]] = []
        self._workers: list[asyncio.Task] = []
        self._pools: list[Any] = []  # ThreadPoolExecutor | None, per stage
        self._running = False
        self._started: list[StageSpec] = []

    # --- introspection ---------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self._specs]

    @property
    def depths(self) -> dict[str, int]:
        """Items waiting in front of each stage. The shape of a backlog says
        which stage is behind: depth piles up in front of the slow one."""
        return {s.name: q.qsize() for s, q in zip(self._specs, self._queues, strict=True)}

    @property
    def depth(self) -> int:
        """Total items in flight in queues. What /ready reports."""
        return sum(q.qsize() for q in self._queues)

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Set every stage up, then start the workers.

        Setup runs first and in order, so a stage that cannot acquire its
        resources fails startup before a single worker exists. If one fails,
        the stages already set up are torn down in reverse -- a half-initialised
        pipeline must not be left holding a GPU.
        """
        if self._running:
            return
        from concurrent.futures import ThreadPoolExecutor

        for spec in self._specs:
            try:
                spec.stage.setup()
            except Exception:
                self._teardown_started()
                raise
            self._started.append(spec)

        self._queues = [asyncio.Queue(maxsize=s.capacity) for s in self._specs]
        self._pools = [
            ThreadPoolExecutor(max_workers=s.workers, thread_name_prefix=s.name)
            if s.workers > 0
            else None
            for s in self._specs
        ]
        self._running = True
        self._workers = [
            asyncio.create_task(self._run(i, w), name=f"{spec.name}-{w}")
            for i, spec in enumerate(self._specs)
            for w in range(spec.concurrency)
        ]
        logger.info(
            "pipeline started: %s",
            " -> ".join(
                f"{s.name}(workers={s.workers},batch={s.max_batch})" for s in self._specs
            ),
        )

    async def stop(self, drain_timeout: float = 5.0) -> None:
        """Refuse new work, drain what is in flight, then release resources."""
        if not self._running:
            self._teardown_started()
            return
        self._running = False

        if self._workers:
            done, pending = await asyncio.wait(self._workers, timeout=drain_timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                logger.warning("%d pipeline worker(s) did not drain in time", len(pending))
        self._workers = []

        # Anything still queued never ran. Tell those clients so, rather than
        # leaving them to discover it via their own timeout.
        shutting_down = PipelineError("server is shutting down")
        stranded = 0
        for queue in self._queues:
            while not queue.empty():
                queue.get_nowait().fail(shutting_down)
                stranded += 1
        if stranded:
            logger.info("failed %d request(s) stranded by shutdown", stranded)

        for pool in self._pools:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
        self._pools = []
        self._teardown_started()
        logger.info("pipeline stopped")

    def _teardown_started(self) -> None:
        """Reverse-order teardown. Never raises: this runs on the shutdown
        path, where one stage's failure must not skip the rest."""
        while self._started:
            spec = self._started.pop()
            try:
                spec.stage.teardown()
            except Exception:  # noqa: BLE001 - cleanup must complete
                logger.exception("teardown failed for stage %s", spec.name)

    # --- ingress ---------------------------------------------------------

    async def submit(self, payload: Any, *, job_id: str | None = None) -> Completion:
        """Push one item through every stage and await its result.

        Raises `PipelineFull` immediately when the ingress queue is at capacity
        -- never waits for space, because waiting for space *is* the unbounded
        behaviour, just with the backlog held in suspended coroutines instead
        of a list.

        Apply a deadline at the call site (`asyncio.wait_for`); the pipeline
        deliberately has no opinion about how long a caller is willing to wait.
        """
        if not self._running:
            raise PipelineNotRunning("pipeline is not running")

        job = Job(
            id=job_id or uuid.uuid4().hex,
            payload=payload,
            future=asyncio.get_running_loop().create_future(),
        )
        try:
            self._queues[0].put_nowait(job)
        except asyncio.QueueFull:
            raise PipelineFull(
                f"pipeline is full ({self._specs[0].capacity} items); the server is at capacity"
            ) from None
        return await job.future

    # --- the worker loop -------------------------------------------------

    async def _run(self, index: int, worker: int) -> None:
        """One worker for one stage. Must survive anything a stage does."""
        spec = self._specs[index]
        queue = self._queues[index]
        # Keep draining after stop() so in-flight work finishes; see the
        # module docstring on shutdown.
        while self._running or not queue.empty():
            batch = await self._collect(queue, spec)
            if not batch:
                continue
            await self._process(index, spec, batch)

    async def _collect(self, queue: asyncio.Queue[Job], spec: StageSpec) -> list[Job]:
        """Assemble a batch under the size-or-deadline rule."""
        try:
            first = await asyncio.wait_for(queue.get(), timeout=_POLL_S)
        except TimeoutError:
            return []

        batch = [first]
        if spec.max_batch == 1:
            return batch

        # The deadline belongs to the first item and is never extended.
        deadline = perf_counter() + spec.max_batch_wait_ms / 1000.0
        while len(batch) < spec.max_batch:
            try:
                batch.append(queue.get_nowait())  # already waiting: free
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
            except TimeoutError:
                break
        return batch

    async def _process(self, index: int, spec: StageSpec, batch: list[Job]) -> None:
        """Run one batch through one stage and hand every job onwards.

        `settled` tracks how many jobs of this batch have been disposed of --
        completed, failed or forwarded. It exists for the cancellation path: a
        worker cancelled by a hard shutdown holds the only reference to these
        jobs, so anything it has not settled must be failed here or its client
        waits for its own timeout to expire. That is precisely the hanging
        shutdown the drain exists to prevent, and nothing else would catch it.
        """
        now = perf_counter()
        wait_ms = max((now - job.entered_queue_at) * 1000.0 for job in batch)
        for job in batch:
            job.wait_ms[spec.name] = (now - job.entered_queue_at) * 1000.0

        payloads = [job.payload for job in batch]
        pool = self._pools[index]
        started = perf_counter()
        settled = 0
        try:
            try:
                if pool is None:
                    results = spec.stage.process(payloads)
                else:
                    results = await asyncio.get_running_loop().run_in_executor(
                        pool, spec.stage.process, payloads
                    )
            except Exception as exc:  # noqa: BLE001 - one bad batch must not kill the loop
                work_ms = (perf_counter() - started) * 1000.0
                logger.warning("stage %s failed a batch of %d: %s", spec.name, len(batch), exc)
                for job in batch:
                    job.fail(exc)
                self._report(spec.name, len(batch), wait_ms, work_ms, 0, len(batch))
                return
            work_ms = (perf_counter() - started) * 1000.0

            if len(results) != len(batch):
                # Cannot map results to jobs, so nobody gets a wrong answer.
                error = StageContractError(
                    f"stage {spec.name!r} returned {len(results)} results for {len(batch)} items"
                )
                logger.error("%s", error)
                for job in batch:
                    job.fail(error)
                self._report(spec.name, len(batch), wait_ms, work_ms, 0, len(batch))
                return

            # Result i belongs to job i. The single line this whole design
            # exists to make unmissable; `strict=True` makes a mismatch loud.
            per_item = work_ms / len(batch)
            ok = 0
            for job, result in zip(batch, results, strict=True):
                job.stage_ms[spec.name] = per_item
                if isinstance(result, Exception):
                    job.fail(result)
                    settled += 1
                    continue
                job.payload = result
                ok += 1
                await self._forward(index, job)
                settled += 1
            self._report(spec.name, len(batch), wait_ms, work_ms, ok, len(batch) - ok)
        except asyncio.CancelledError:
            shutting_down = PipelineError("server is shutting down")
            for job in batch[settled:]:
                job.fail(shutting_down)
            raise

    async def _forward(self, index: int, job: Job) -> None:
        """Hand a job to the next stage, or complete it."""
        nxt = index + 1
        if nxt >= len(self._specs):
            job.complete()
            return
        job.entered_queue_at = perf_counter()
        # Awaits when the downstream queue is full. That wait IS the
        # backpressure; it stalls this worker, which stalls its queue, which
        # eventually reaches the ingress and becomes a 503. A cancellation here
        # is handled by the caller, which knows what else is still unsettled.
        await self._queues[nxt].put(job)

    def _report(
        self, name: str, size: int, wait_ms: float, work_ms: float, ok: int, failed: int
    ) -> None:
        if self._on_stage is None:
            return
        try:
            self._on_stage(StageReport(name, size, wait_ms, work_ms, ok, failed))
        except Exception:  # noqa: BLE001 - observability must never break serving
            logger.exception("on_stage observer raised for stage %s", name)
