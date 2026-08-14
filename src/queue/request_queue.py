"""The bounded queue between HTTP and the GPU (Phase 10).

WHY A QUEUE EXISTS AT ALL

    One GPU, one engine, one inference at a time. Requests arrive concurrently
    and independently. Something has to hold them in between, and that
    something is where three production properties are decided: how much load
    the process will absorb, what happens when it cannot, and how requests get
    grouped into batches.

WHY IT IS BOUNDED

    An unbounded queue does not absorb overload, it defers it. Under a spike
    the queue grows, each request's wait grows with it, clients time out and
    retry, and the retries enqueue too. Memory climbs until the kernel OOM-kills
    the process and every in-flight request dies at once.

    A bounded queue converts that into a decision: when it is full, reject the
    newest arrival immediately with 503. One client gets a fast, honest failure
    instead of everyone getting a slow one. Backpressure is a feature.

WHY EACH ENTRY CARRIES A FUTURE

    HTTP is request/response, but the GPU works in batches. A request that
    enters the queue must be reunited with its own row of the output tensor
    later, on a different task. An asyncio.Future is the join point: the
    handler awaits it, the batch worker resolves it, and the request_id keeps
    the mapping auditable.

WHAT IS QUEUED IS A TENSOR, NOT AN IMAGE

    Preprocessing happens before enqueueing, in a thread pool. Phase 4 measured
    one preprocessing thread sustaining ~46 img/s against an engine that can
    absorb ~700-3200, so preprocessing inside the batch worker would serialise
    the exact stage that most needs parallelism. By the time a request reaches
    this queue it is already a (3, H, W) float32 array, and the batch worker
    only stacks and runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np


class QueueFull(Exception):
    """The queue is at capacity. Maps to HTTP 503 with a Retry-After."""


class RequestTimeout(Exception):
    """The request's deadline passed before it was served. Maps to HTTP 504."""


@dataclass(slots=True)
class InferenceRequest:
    """One in-flight request, from arrival to resolution."""

    request_id: str
    tensor: np.ndarray  # (3, H, W) float32, already preprocessed
    future: asyncio.Future
    enqueued_at: float = field(default_factory=perf_counter)
    preprocess_ms: float = 0.0

    @property
    def queue_wait_ms(self) -> float:
        """How long this request sat waiting to be picked up.

        The most diagnostic number in the whole system. Rising queue wait with
        flat inference time means the GPU is saturated and the fix is capacity
        or batching, not a faster kernel.
        """
        return (perf_counter() - self.enqueued_at) * 1000.0

    def resolve(self, value: Any) -> None:
        """Deliver a result. Safe if the client already went away."""
        if not self.future.done():
            self.future.set_result(value)

    def fail(self, exc: BaseException) -> None:
        if not self.future.done():
            self.future.set_exception(exc)


class RequestQueue:
    """Bounded FIFO of preprocessed requests.

    Thin on purpose: asyncio.Queue already does the hard part correctly. What
    this adds is the rejection policy, the counters the metrics layer needs,
    and a name for the concept.
    """

    def __init__(self, max_size: int) -> None:
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=max_size)
        self.max_size = max_size
        self.total_enqueued = 0
        self.total_rejected = 0

    def submit(self, request: InferenceRequest) -> None:
        """Enqueue without waiting, or reject.

        put_nowait rather than await put(): awaiting a full queue is exactly
        the unbounded behaviour we are avoiding, just with the backlog held in
        suspended coroutines instead of a list. Rejection has to be immediate
        to be useful.
        """
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            self.total_rejected += 1
            raise QueueFull(
                f"queue is full ({self.max_size} requests); the server is at capacity"
            ) from None
        self.total_enqueued += 1

    async def get(self) -> InferenceRequest:
        return await self._queue.get()

    def get_nowait(self) -> InferenceRequest | None:
        """Drain an already-waiting request, or None. Used to fill a batch."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def drain(self) -> list[InferenceRequest]:
        """Remove everything still queued. For shutdown, so waiting clients get
        a clean error instead of a hung connection."""
        remaining = []
        while (item := self.get_nowait()) is not None:
            remaining.append(item)
        return remaining

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def __len__(self) -> int:
        return self._queue.qsize()
