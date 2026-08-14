"""Dynamic batching.

The load-bearing test is `test_each_request_gets_its_own_row`. If output rows
are mapped to the wrong requests, every client receives a confident prediction
belonging to someone else -- no exception, no log line, nothing else catches it.
"""

import asyncio

import numpy as np
import pytest

from src.inference.base import (
    EngineError,
    EngineMetadata,
    InferenceEngine,
    InferenceResult,
    StageTimings,
)
from src.queue import BatchManager, InferenceRequest, RequestQueue

SHAPE = (3, 224, 224)


class RecordingEngine(InferenceEngine):
    """Returns logits derived from the input so rows are traceable.

    Row i's logits are all equal to the input's first pixel, which the test
    sets to a per-request marker. Any mis-mapping shows up as a wrong marker
    rather than as plausible-looking noise.
    """

    def __init__(self, delay: float = 0.0, fail: bool = False):
        self._loaded = True
        self.delay = delay
        self.fail = fail
        self.batch_sizes: list[int] = []

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            backend="fake",
            model_name="fake",
            model_version="v1",
            precision="fp32",
            math_mode="fp32",
            device="cpu",
            input_shape=SHAPE,
            max_batch_size=64,
            num_classes=10,
        )

    def _predict(self, batch: np.ndarray) -> InferenceResult:
        import time

        self.batch_sizes.append(batch.shape[0])
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise EngineError("engine exploded")
        markers = batch[:, 0, 0, 0]
        logits = np.repeat(markers[:, None], 10, axis=1).astype(np.float32)
        return InferenceResult(
            logits=logits,
            timings=StageTimings(h2d_ms=0.1, compute_ms=1.0, d2h_ms=0.1),
        )


def marked_request(loop, marker: float) -> InferenceRequest:
    tensor = np.zeros(SHAPE, dtype=np.float32)
    tensor[0, 0, 0] = marker
    return InferenceRequest(request_id=f"r{marker:.0f}", tensor=tensor, future=loop.create_future())


async def run_manager(engine, requests, *, max_batch_size=8, max_wait_ms=10.0):
    queue = RequestQueue(max_size=100)
    outcomes = []
    manager = BatchManager(
        engine,
        queue,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_wait_ms,
        on_batch=outcomes.append,
    )
    manager.start()
    for r in requests:
        queue.submit(r)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(r.future for r in requests), return_exceptions=True),
            timeout=5.0,
        )
    finally:
        await manager.stop()
    return results, outcomes


async def test_each_request_gets_its_own_row():
    """Output row i must reach request i. The correctness-critical invariant."""
    loop = asyncio.get_running_loop()
    requests = [marked_request(loop, float(i)) for i in range(8)]
    results, _ = await run_manager(RecordingEngine(), requests)
    for i, logits in enumerate(results):
        assert logits[0] == pytest.approx(float(i)), f"request {i} got another request's result"


async def test_requests_are_actually_batched_together():
    loop = asyncio.get_running_loop()
    engine = RecordingEngine()
    requests = [marked_request(loop, float(i)) for i in range(8)]
    await run_manager(engine, requests, max_batch_size=8, max_wait_ms=50.0)
    # All eight were queued before the loop ran, so they belong in one call.
    assert max(engine.batch_sizes) > 1
    assert sum(engine.batch_sizes) == 8


async def test_batch_never_exceeds_max_batch_size():
    loop = asyncio.get_running_loop()
    engine = RecordingEngine()
    requests = [marked_request(loop, float(i)) for i in range(20)]
    await run_manager(engine, requests, max_batch_size=4, max_wait_ms=50.0)
    assert max(engine.batch_sizes) <= 4
    assert sum(engine.batch_sizes) == 20


async def test_a_lone_request_does_not_wait_forever():
    """Without the deadline a partial batch would never dispatch."""
    loop = asyncio.get_running_loop()
    request = marked_request(loop, 7.0)
    started = loop.time()
    results, _ = await run_manager(
        RecordingEngine(), [request], max_batch_size=32, max_wait_ms=20.0
    )
    elapsed = loop.time() - started
    assert results[0][0] == pytest.approx(7.0)
    assert elapsed < 1.0  # dispatched on the timer, not on a full batch


async def test_engine_failure_fails_only_that_batch():
    loop = asyncio.get_running_loop()
    requests = [marked_request(loop, float(i)) for i in range(4)]
    results, _ = await run_manager(RecordingEngine(fail=True), requests)
    assert all(isinstance(r, EngineError) for r in results)


async def test_loop_survives_a_failed_batch_and_serves_the_next():
    loop = asyncio.get_running_loop()
    engine = RecordingEngine(fail=True)
    queue = RequestQueue(max_size=100)
    manager = BatchManager(engine, queue, max_batch_size=2, max_batch_wait_ms=5.0)
    manager.start()
    try:
        bad = marked_request(loop, 1.0)
        queue.submit(bad)
        with pytest.raises(EngineError):
            await asyncio.wait_for(bad.future, timeout=2.0)

        engine.fail = False  # recover
        good = marked_request(loop, 2.0)
        queue.submit(good)
        assert (await asyncio.wait_for(good.future, timeout=2.0))[0] == pytest.approx(2.0)
    finally:
        await manager.stop()


async def test_outcomes_are_reported_for_metrics():
    loop = asyncio.get_running_loop()
    requests = [marked_request(loop, float(i)) for i in range(4)]
    _, outcomes = await run_manager(RecordingEngine(), requests, max_wait_ms=50.0)
    assert outcomes
    assert sum(o.batch_size for o in outcomes) == 4
    assert all(o.inference_ms > 0 for o in outcomes)


async def test_shutdown_fails_waiting_requests_instead_of_hanging():
    loop = asyncio.get_running_loop()
    queue = RequestQueue(max_size=100)
    manager = BatchManager(
        RecordingEngine(delay=0.05), queue, max_batch_size=1, max_batch_wait_ms=1.0
    )
    manager.start()
    pending = [marked_request(loop, float(i)) for i in range(20)]
    for r in pending:
        queue.submit(r)
    await manager.stop(drain_timeout=0.1)

    # Everything must be settled: served or cleanly failed, never left hanging.
    await asyncio.sleep(0.05)
    assert all(r.future.done() for r in pending)
