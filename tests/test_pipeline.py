"""The pipeline runtime: mapping, batching, backpressure, failure, lifecycle.

The load-bearing test is `test_result_i_goes_to_job_i`. If results are mapped
to the wrong jobs, every client receives a confident answer belonging to
somebody else -- no exception, no log line, nothing else catches it. It used to
be hand-written per batching implementation; now there is one runner and this
is the test that guards it.

Every guarantee the old `src/queue/` package was tested for is re-asserted here
against the generic runner, plus the ones the generic runner newly makes
possible (per-item failure, contract violations, multi-stage timing).
"""

from __future__ import annotations

import asyncio

import pytest

from src.pipeline import (
    Pipeline,
    PipelineFull,
    PipelineNotRunning,
    Stage,
    StageContractError,
    StageSpec,
)


class Echo(Stage[int, int]):
    """Records what it was handed, so batching can be asserted on."""

    def __init__(self, name: str = "echo", *, delay: float = 0.0, offset: int = 0) -> None:
        self.name = name
        self.delay = delay
        self.offset = offset
        self.batch_sizes: list[int] = []
        self.seen: list[int] = []

    def process(self, items: list[int]) -> list[int]:
        import time

        self.batch_sizes.append(len(items))
        self.seen.extend(items)
        if self.delay:
            time.sleep(self.delay)
        return [i + self.offset for i in items]


class Exploding(Stage[int, int]):
    """Raises for the whole batch -- the CUDA-OOM shape of failure."""

    name = "boom"

    def __init__(self, fail: bool = True) -> None:
        self.fail = fail

    def process(self, items: list[int]) -> list[int]:
        if self.fail:
            raise RuntimeError("stage exploded")
        return list(items)


class OddItemsFail(Stage[int, int]):
    """Returns an Exception in some slots -- the corrupt-JPEG shape of failure."""

    name = "picky"

    def process(self, items):
        return [ValueError(f"{i} is odd") if i % 2 else i for i in items]


class WrongLength(Stage[int, int]):
    name = "liar"

    def process(self, items):
        return list(items)[:-1] if len(items) > 1 else [items[0], items[0]]


async def run(specs, payloads, *, timeout=5.0):
    """Start, submit everything concurrently, gather, stop."""
    pipeline = Pipeline(specs)
    pipeline.start()
    try:
        return await asyncio.wait_for(
            asyncio.gather(
                *(pipeline.submit(p) for p in payloads), return_exceptions=True
            ),
            timeout=timeout,
        )
    finally:
        await pipeline.stop()


# --- mapping: the correctness-critical invariant -------------------------


async def test_result_i_goes_to_job_i():
    """Every job must receive its OWN result, not a plausible neighbour's."""
    results = await run(
        [StageSpec(Echo(offset=1000), max_batch=8, max_batch_wait_ms=20.0)],
        list(range(8)),
    )
    assert [r.result for r in results] == [i + 1000 for i in range(8)]


async def test_mapping_holds_across_several_stages():
    results = await run(
        [
            StageSpec(Echo("a", offset=1), max_batch=4, max_batch_wait_ms=20.0),
            StageSpec(Echo("b", offset=10), max_batch=3, max_batch_wait_ms=20.0),
            StageSpec(Echo("c", offset=100), workers=0, max_batch=8),
        ],
        list(range(12)),
    )
    assert sorted(r.result for r in results) == sorted(i + 111 for i in range(12))
    # And each individual job got its own value, not merely the right multiset.
    assert [r.result for r in results] == [i + 111 for i in range(12)]


# --- batching ------------------------------------------------------------


async def test_items_are_actually_batched_together():
    echo = Echo()
    await run([StageSpec(echo, max_batch=8, max_batch_wait_ms=50.0)], list(range(8)))
    assert max(echo.batch_sizes) > 1
    assert sum(echo.batch_sizes) == 8


async def test_batch_never_exceeds_max_batch():
    echo = Echo()
    await run([StageSpec(echo, max_batch=4, max_batch_wait_ms=50.0)], list(range(20)))
    assert max(echo.batch_sizes) <= 4
    assert sum(echo.batch_sizes) == 20


async def test_a_lone_item_does_not_wait_forever():
    """Without the deadline a partial batch would never dispatch."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    results = await run([StageSpec(Echo(), max_batch=32, max_batch_wait_ms=20.0)], [7])
    assert results[0].result == 7
    assert loop.time() - started < 1.0  # dispatched on the timer, not a full batch


async def test_max_batch_one_never_groups():
    echo = Echo()
    await run([StageSpec(echo, max_batch=1)], list(range(6)))
    assert echo.batch_sizes == [1] * 6


# --- backpressure and capacity -------------------------------------------


class Blocked(Stage[int, int]):
    """Holds its worker until released, so capacity can be tested without
    racing a sleep. `gate.wait()` blocks the pool thread, never the loop."""

    name = "blocked"

    def __init__(self, gate) -> None:
        self.gate = gate

    def process(self, items: list[int]) -> list[int]:
        self.gate.wait(timeout=5.0)
        return list(items)


async def test_pipeline_rejects_when_full_instead_of_growing():
    """The whole reason the ingress is bounded.

    An unbounded queue turns a load spike into an OOM kill that drops every
    in-flight request; a bounded one drops the newest arrival and keeps serving.
    """
    import threading

    gate = threading.Event()
    pipeline = Pipeline([StageSpec(Blocked(gate), max_batch=1, capacity=2)])
    pipeline.start()
    try:
        # One worker is stuck, so at most 2 can queue and 1 can be in flight.
        tasks = [asyncio.create_task(pipeline.submit(i)) for i in range(10)]
        await asyncio.sleep(0.05)
        assert pipeline.depth <= 2, "the queue grew past its bound"

        gate.set()
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=5.0
        )
        rejected = [r for r in results if isinstance(r, PipelineFull)]
        served = [r for r in results if not isinstance(r, BaseException)]
        assert rejected, "a full pipeline must reject rather than grow"
        assert len(served) <= 3  # 2 queued + 1 held by the worker
        assert len(rejected) + len(served) == 10  # nothing vanished
    finally:
        gate.set()
        await pipeline.stop()


async def test_rejection_is_immediate_not_a_wait():
    """submit() must not block waiting for space. Waiting for space IS the
    unbounded behaviour, just with the backlog in suspended coroutines."""
    import threading

    loop = asyncio.get_running_loop()
    gate = threading.Event()
    pipeline = Pipeline([StageSpec(Blocked(gate), max_batch=1, capacity=1)])
    pipeline.start()
    try:
        # Sequenced, not raced: the first submit must reach the worker before
        # the second fills the single queue slot.
        held = asyncio.create_task(pipeline.submit(0))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(pipeline.submit(1))
        await asyncio.sleep(0.05)
        busy = [held, queued]
        assert pipeline.depth == 1  # the one slot is now occupied

        started = loop.time()
        with pytest.raises(PipelineFull):
            await pipeline.submit(99)
        assert loop.time() - started < 0.05

        gate.set()
        await asyncio.wait_for(asyncio.gather(*busy, return_exceptions=True), timeout=5.0)
    finally:
        gate.set()
        await pipeline.stop()


async def test_capacity_frees_as_items_are_consumed():
    pipeline = Pipeline([StageSpec(Echo(), max_batch=1, capacity=2)])
    pipeline.start()
    try:
        for i in range(10):  # far more than capacity, drained as it goes
            assert (await asyncio.wait_for(pipeline.submit(i), 2.0)).result == i
    finally:
        await pipeline.stop()


async def test_a_slow_downstream_stage_does_not_lose_items():
    """Backpressure, not loss. A full downstream queue stalls the upstream
    worker; every item still arrives exactly once."""
    slow = Echo("slow", delay=0.01)
    results = await run(
        [
            StageSpec(Echo("fast"), workers=4, max_batch=1, capacity=64),
            StageSpec(slow, workers=1, max_batch=1, capacity=1),
        ],
        list(range(20)),
        timeout=15.0,
    )
    assert len(results) == 20
    assert sorted(slow.seen) == list(range(20))


# --- failure isolation ---------------------------------------------------


async def test_a_raising_stage_fails_only_its_own_batch():
    results = await run([StageSpec(Exploding(), max_batch=4, max_batch_wait_ms=20.0)], [1, 2, 3, 4])
    assert all(isinstance(r, RuntimeError) for r in results)


async def test_pipeline_survives_a_failed_batch_and_serves_the_next():
    stage = Exploding(fail=True)
    pipeline = Pipeline([StageSpec(stage, max_batch=2, max_batch_wait_ms=5.0)])
    pipeline.start()
    try:
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(pipeline.submit(1), 2.0)
        stage.fail = False  # recover
        assert (await asyncio.wait_for(pipeline.submit(2), 2.0)).result == 2
    finally:
        await pipeline.stop()


async def test_a_returned_exception_fails_only_that_item():
    """One corrupt input must not fail the fifteen good ones beside it."""
    results = await run(
        [StageSpec(OddItemsFail(), max_batch=8, max_batch_wait_ms=20.0)], list(range(8))
    )
    for i, r in enumerate(results):
        if i % 2:
            assert isinstance(r, ValueError)
        else:
            assert r.result == i


async def test_a_failed_item_does_not_reach_the_next_stage():
    downstream = Echo("downstream")
    await run(
        [
            StageSpec(OddItemsFail(), max_batch=8, max_batch_wait_ms=20.0),
            StageSpec(downstream, max_batch=8, max_batch_wait_ms=20.0),
        ],
        list(range(8)),
    )
    assert sorted(downstream.seen) == [0, 2, 4, 6]


async def test_wrong_result_count_fails_everyone_rather_than_guessing():
    """If lengths disagree the runner cannot know whose result is whose.
    Guessing here is exactly how everyone gets somebody else's answer."""
    results = await run(
        [StageSpec(WrongLength(), max_batch=4, max_batch_wait_ms=20.0)], [1, 2, 3, 4]
    )
    assert all(isinstance(r, StageContractError) for r in results)


# --- lifecycle -----------------------------------------------------------


class Resourceful(Stage[int, int]):
    def __init__(self, name: str, log: list[str], *, fail_setup: bool = False) -> None:
        self.name = name
        self.log = log
        self.fail_setup = fail_setup

    def setup(self):
        if self.fail_setup:
            raise RuntimeError(f"{self.name} cannot start")
        self.log.append(f"setup:{self.name}")

    def teardown(self):
        self.log.append(f"teardown:{self.name}")

    def process(self, items):
        return list(items)


async def test_setup_runs_in_order_and_teardown_in_reverse():
    log: list[str] = []
    pipeline = Pipeline(
        [StageSpec(Resourceful("a", log)), StageSpec(Resourceful("b", log))]
    )
    pipeline.start()
    await pipeline.stop()
    assert log == ["setup:a", "setup:b", "teardown:b", "teardown:a"]


async def test_a_failed_setup_tears_down_what_already_started():
    """A half-initialised pipeline must not be left holding a GPU."""
    log: list[str] = []
    pipeline = Pipeline(
        [
            StageSpec(Resourceful("a", log)),
            StageSpec(Resourceful("b", log, fail_setup=True)),
        ]
    )
    with pytest.raises(RuntimeError, match="b cannot start"):
        pipeline.start()
    assert log == ["setup:a", "teardown:a"]
    assert not pipeline.is_running


async def test_submit_before_start_and_after_stop_is_refused():
    pipeline = Pipeline([StageSpec(Echo())])
    with pytest.raises(PipelineNotRunning):
        await pipeline.submit(1)
    pipeline.start()
    await pipeline.stop()
    with pytest.raises(PipelineNotRunning):
        await pipeline.submit(1)


async def test_shutdown_settles_every_waiting_request_instead_of_hanging():
    """Served or cleanly failed, never left for the client's own timeout."""
    pipeline = Pipeline([StageSpec(Echo(delay=0.05), max_batch=1, capacity=64)])
    pipeline.start()
    pending = [asyncio.create_task(pipeline.submit(i)) for i in range(20)]
    await asyncio.sleep(0.02)
    await pipeline.stop(drain_timeout=0.1)

    settled = await asyncio.gather(*pending, return_exceptions=True)
    assert len(settled) == 20
    assert all(s is not None for s in settled)


async def test_stop_is_idempotent():
    pipeline = Pipeline([StageSpec(Echo())])
    pipeline.start()
    await pipeline.stop()
    await pipeline.stop()  # must not raise


# --- observability -------------------------------------------------------


async def test_each_stage_reports_itself_for_metrics():
    reports = []
    pipeline = Pipeline(
        [
            StageSpec(Echo("a"), max_batch=4, max_batch_wait_ms=20.0),
            StageSpec(Echo("b"), workers=0, max_batch=4),
        ],
        on_stage=reports.append,
    )
    pipeline.start()
    try:
        await asyncio.gather(*(pipeline.submit(i) for i in range(4)))
    finally:
        await pipeline.stop()

    by_stage = {r.stage for r in reports}
    assert by_stage == {"a", "b"}
    assert sum(r.succeeded for r in reports if r.stage == "a") == 4
    assert all(r.work_ms >= 0 and r.wait_ms >= 0 for r in reports)


async def test_an_observer_that_raises_does_not_break_serving():
    def hostile(_report):
        raise RuntimeError("monitoring is down")

    pipeline = Pipeline([StageSpec(Echo())], on_stage=hostile)
    pipeline.start()
    try:
        assert (await asyncio.wait_for(pipeline.submit(5), 2.0)).result == 5
    finally:
        await pipeline.stop()


async def test_completion_carries_per_stage_timings():
    pipeline = Pipeline(
        [StageSpec(Echo("a", delay=0.01)), StageSpec(Echo("b"), workers=0)]
    )
    pipeline.start()
    try:
        completion = await asyncio.wait_for(pipeline.submit(1), 2.0)
    finally:
        await pipeline.stop()

    assert set(completion.stage_ms) == {"a", "b"}
    assert set(completion.wait_ms) == {"a", "b"}
    assert completion.stage_ms["a"] >= 5.0  # the 10 ms sleep, less timer slop
    assert completion.total_ms >= completion.worked_ms
    assert completion.queued_ms >= 0


async def test_depths_are_reported_per_stage():
    import threading

    gate = threading.Event()
    pipeline = Pipeline([StageSpec(Blocked(gate), max_batch=1, capacity=8)])
    pipeline.start()
    try:
        tasks = [asyncio.create_task(pipeline.submit(i)) for i in range(4)]
        await asyncio.sleep(0.05)
        assert pipeline.depths == {"blocked": 3}  # one is held by the worker
        assert pipeline.depth == 3
        gate.set()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
    finally:
        gate.set()
        await pipeline.stop()


# --- construction --------------------------------------------------------


async def test_duplicate_stage_names_are_refused():
    """Names key metric labels and the latency breakdown; duplicates would
    silently merge two stages into one number."""
    with pytest.raises(ValueError, match="unique"):
        Pipeline([StageSpec(Echo("same")), StageSpec(Echo("same"))])


async def test_an_empty_pipeline_is_refused():
    with pytest.raises(ValueError, match="at least one stage"):
        Pipeline([])


@pytest.mark.parametrize(
    "kwargs",
    [{"workers": -1}, {"max_batch": 0}, {"max_batch_wait_ms": -1.0}, {"capacity": 0}],
)
async def test_nonsense_stage_specs_are_refused_at_construction(kwargs):
    with pytest.raises(ValueError):
        StageSpec(Echo(), **kwargs)


async def test_inline_stage_runs_without_a_thread():
    """workers=0 opts out of the thread hop for stages too cheap to justify it."""
    import threading

    seen: list[str] = []

    class WhereAmI(Stage[int, int]):
        name = "inline"

        def process(self, items):
            seen.append(threading.current_thread().name)
            return list(items)

    pipeline = Pipeline([StageSpec(WhereAmI(), workers=0)])
    pipeline.start()
    try:
        await asyncio.wait_for(pipeline.submit(1), 2.0)
    finally:
        await pipeline.stop()
    assert seen == [threading.main_thread().name]
