"""Request queue: capacity, rejection and shutdown."""

import asyncio

import numpy as np
import pytest

from src.queue import InferenceRequest, QueueFull, RequestQueue


def make_request(loop, rid="r") -> InferenceRequest:
    return InferenceRequest(
        request_id=rid,
        tensor=np.zeros((3, 224, 224), dtype=np.float32),
        future=loop.create_future(),
    )


async def test_fifo_order_is_preserved():
    loop = asyncio.get_running_loop()
    q = RequestQueue(max_size=10)
    for i in range(5):
        q.submit(make_request(loop, f"r{i}"))
    assert [(await q.get()).request_id for _ in range(5)] == ["r0", "r1", "r2", "r3", "r4"]


async def test_queue_rejects_when_full_instead_of_growing():
    """The whole reason the queue is bounded.

    An unbounded queue turns a load spike into an OOM kill that drops every
    in-flight request; a bounded one drops the newest arrival and keeps serving.
    """
    loop = asyncio.get_running_loop()
    q = RequestQueue(max_size=3)
    for _ in range(3):
        q.submit(make_request(loop))

    with pytest.raises(QueueFull, match="full"):
        q.submit(make_request(loop))

    assert q.depth == 3  # did not grow past its bound
    assert q.total_rejected == 1
    assert q.total_enqueued == 3


async def test_rejection_is_immediate_not_a_wait():
    # submit() must not block. Awaiting a full queue is the unbounded
    # behaviour again, just with the backlog in suspended coroutines.
    loop = asyncio.get_running_loop()
    q = RequestQueue(max_size=1)
    q.submit(make_request(loop))
    started = loop.time()
    with pytest.raises(QueueFull):
        q.submit(make_request(loop))
    assert loop.time() - started < 0.05


async def test_capacity_frees_as_requests_are_taken():
    loop = asyncio.get_running_loop()
    q = RequestQueue(max_size=2)
    q.submit(make_request(loop))
    q.submit(make_request(loop))
    await q.get()
    q.submit(make_request(loop))  # room again


async def test_get_nowait_returns_none_when_empty():
    assert RequestQueue(max_size=4).get_nowait() is None


async def test_drain_empties_and_returns_everything():
    loop = asyncio.get_running_loop()
    q = RequestQueue(max_size=10)
    for i in range(4):
        q.submit(make_request(loop, f"r{i}"))
    drained = q.drain()
    assert [r.request_id for r in drained] == ["r0", "r1", "r2", "r3"]
    assert q.depth == 0


async def test_resolve_and_fail_are_safe_after_the_client_left():
    loop = asyncio.get_running_loop()
    r = make_request(loop)
    r.future.cancel()  # client disconnected
    r.resolve(np.zeros(1000))  # must not raise
    r.fail(RuntimeError("boom"))


async def test_queue_wait_is_measured():
    loop = asyncio.get_running_loop()
    r = make_request(loop)
    await asyncio.sleep(0.02)
    assert r.queue_wait_ms >= 15
