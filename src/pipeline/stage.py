"""What a pipeline stage is, and what flows between stages.

THE ONE IDEA

    A serving system is a sequence of transformations with different costs and
    different parallelism. Decoding a JPEG is 9-22 ms of CPU that parallelises
    perfectly across threads. A GPU forward pass is 7 ms that must be
    serialised through one CUDA context but wants many images at once. Softmax
    is 0.1 ms that is not worth a thread hop.

    Those are the same shape of thing -- `items in, items out` -- differing
    only in how they should be *run*. So this module defines the shape once
    (`Stage`) and describes the running separately (`StageSpec`). A stage
    author writes a pure transformation and never writes a queue, a thread
    pool, a deadline or a metric.

WHY BATCH IN, BATCH OUT

    `process()` takes a list and returns a list, even for stages that work one
    item at a time. A per-item stage simply runs with `max_batch=1` and never
    notices. The alternative -- separate `Stage` and `BatchStage` types --
    doubles the interface so that the *runner* can be simpler, which is the
    wrong trade: there is one runner and there will be many stages.

    Two hard rules, both enforced by the runner rather than trusted:

      1. `len(output) == len(input)`.
      2. `output[i]` belongs to `input[i]`.

    Rule 2 is the correctness-critical one in this whole codebase. The GPU
    returns an (N, num_classes) tensor and row i must reach the request that
    supplied row i. Get it wrong and every client receives a confident,
    plausible prediction belonging to somebody else -- no exception, no log
    line, nothing to alert on. It used to be hand-written in the batch manager.
    Now it is one `zip()` in one runner, checked by one test, and no stage can
    get it wrong because no stage performs it.

HOW A STAGE REPORTS FAILURE

    Two ways, and the difference is the blast radius:

      raise            -> every item in this call fails. Correct for a GPU OOM
                          or a dead engine: the batch genuinely did not happen.
      return Exception -> only that slot fails, the rest proceed. Correct for
                          one corrupt JPEG in a batch of sixteen.

    Both are supported because both occur. A stage that could only raise would
    force a batched decoder to fail fifteen good images because of one bad one.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


class PipelineError(RuntimeError):
    """Base for every failure the pipeline itself raises."""


class PipelineFull(PipelineError):
    """The ingress queue is at capacity. Maps to HTTP 503 + Retry-After.

    Bounded, always. An unbounded queue does not absorb overload, it defers it:
    the backlog grows, waits grow with it, clients time out and retry, the
    retries enqueue too, and memory climbs until the kernel OOM-kills the
    process and every in-flight request dies at once. Rejecting the newest
    arrival converts that into one fast honest failure instead of everyone
    getting a slow one. Backpressure is a feature.
    """


class PipelineNotRunning(PipelineError):
    """submit() before start(), or after stop(). A programming error."""


class StageContractError(PipelineError):
    """A stage returned the wrong number of results.

    Not recoverable and not ignorable: if the output length does not match the
    input length, the runner cannot know which result belongs to which item, so
    it fails all of them rather than guess. Guessing here is how everyone gets
    somebody else's answer.
    """


# A stage returns one entry per input item. An `Exception` in slot i fails
# item i and nothing else; see the module docstring.
type StageOutput[Out] = list[Out | Exception]


class Stage[In, Out](ABC):
    """One transformation in a pipeline.

    Implement `process`. Optionally implement `setup`/`teardown` if the stage
    owns a resource -- an engine, a session, a device buffer.

    Subclasses must be safe to call from several threads at once whenever they
    are configured with `workers > 1`. In practice that means "hold no mutable
    per-call state on self", which is the natural way to write them anyway.
    """

    #: Identifies this stage in metrics, logs and the latency breakdown a
    #: client receives. Lowercase, short, stable -- it is a metric label and a
    #: JSON key, so changing it breaks dashboards.
    name: str = "stage"

    def setup(self) -> None:  # noqa: B027 - optional hook, not every stage owns a resource
        """Acquire whatever serving requires. Called once, before any worker.

        Separate from `__init__` for the same reason `InferenceEngine.load()`
        is: constructing a stage must be cheap and infallible, so that a
        pipeline can be assembled and inspected before anything touches a GPU.
        Raising here fails startup with a readable reason instead of producing
        an object that exists but cannot serve.
        """

    def teardown(self) -> None:  # noqa: B027 - optional hook, pairs with setup()
        """Release what `setup` acquired. Called once, after every worker stops.

        Must be idempotent and must not raise: it runs on the shutdown path,
        where a second call during a signal-handler race must not abort the
        rest of the cleanup.
        """

    @abstractmethod
    def process(self, items: list[In]) -> StageOutput[Out]:
        """Transform a batch. MUST return exactly `len(items)` results, in order.

        `items` is never empty. See the module docstring for the two failure
        modes and which to use.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


@dataclass(frozen=True, slots=True)
class StageSpec:
    """How to *run* a stage -- concurrency, batching, backpressure.

    Separate from the stage itself so the same transformation can be run three
    different ways without touching its code. That separation is not decoration:
    it is what let this project move image decoding from `workers=8, max_batch=1`
    to a batched device stage and back by editing one line, and measure both.

    Defaults describe the common case: one worker, one item at a time, no wait.
    """

    stage: Stage[Any, Any]

    #: Concurrent workers for this stage.
    #:
    #:   0  -> run inline on the event loop. No thread hop (~50 us saved), but
    #:         the loop is blocked for the duration. Only for stages measured in
    #:         microseconds, like softmax.
    #:   1  -> one worker in one thread. Correct for anything holding a single
    #:         CUDA context: a second worker would serialise on it anyway while
    #:         making the VRAM ceiling harder to reason about.
    #:   N  -> N workers in an N-thread pool. Correct for CPU work that releases
    #:         the GIL, which is what PIL does during decode and resize.
    workers: int = 1

    #: Upper bound on items handed to `process()` in one call.
    max_batch: int = 1

    #: How long to wait for a batch to fill, measured from the arrival of its
    #: FIRST item and never reset by later arrivals -- otherwise a steady
    #: trickle postpones dispatch indefinitely. Charged to every request on an
    #: idle system, which is why the serving default is small.
    max_batch_wait_ms: float = 0.0

    #: Bound on this stage's input queue. Reaching it blocks the *upstream*
    #: stage, which propagates backpressure to the ingress queue, which is
    #: where it becomes a 503. Only the ingress bound is visible to clients.
    capacity: int = 128

    def __post_init__(self) -> None:
        if self.workers < 0:
            raise ValueError(f"{self.stage.name}: workers must be >= 0, got {self.workers}")
        if self.max_batch < 1:
            raise ValueError(f"{self.stage.name}: max_batch must be >= 1, got {self.max_batch}")
        if self.max_batch_wait_ms < 0:
            raise ValueError(f"{self.stage.name}: max_batch_wait_ms must be >= 0")
        if self.capacity < 1:
            raise ValueError(f"{self.stage.name}: capacity must be >= 1, got {self.capacity}")

    @property
    def name(self) -> str:
        return self.stage.name

    @property
    def concurrency(self) -> int:
        """Worker tasks to run. `workers=0` still needs one task to pump."""
        return max(1, self.workers)


@dataclass(frozen=True, slots=True)
class StageReport:
    """What one `process()` call cost. Handed to the pipeline's observer.

    Exists so the pipeline never imports Prometheus. The pipeline measures and
    reports; the application decides what that means and where it goes.
    Dependency inversion, and the payoff is that `src/pipeline/` is reusable in
    a project that exports metrics some other way, or none.
    """

    stage: str
    batch_size: int
    #: Longest time any item in this batch spent queued before the stage.
    #: The most diagnostic number a pipeline produces: rising wait with flat
    #: work means this stage is the bottleneck and nothing downstream is.
    wait_ms: float
    #: Wall time inside `process()`, for the whole batch.
    work_ms: float
    succeeded: int
    failed: int


@dataclass(frozen=True, slots=True)
class Completion[Out]:
    """A finished job: the result, and where its time went.

    Per-stage timings rather than the fixed `preprocess/inference/postprocess`
    triple the API used to return. A template cannot know its own stage names,
    and a pipeline that grows a stage should report it without a schema change.
    """

    result: Out
    #: stage name -> milliseconds inside that stage's `process()`.
    stage_ms: dict[str, float]
    #: stage name -> milliseconds queued *before* that stage ran.
    wait_ms: dict[str, float]
    #: Submit to completion, as measured inside the pipeline. Excludes whatever
    #: the caller did before submitting (reading an upload) and after.
    total_ms: float

    @property
    def queued_ms(self) -> float:
        """Total time spent waiting rather than working."""
        return sum(self.wait_ms.values())

    @property
    def worked_ms(self) -> float:
        return sum(self.stage_ms.values())


@dataclass(slots=True)
class Job:
    """One item in flight, from submit to completion.

    Mutable and deliberately so: `payload` is reassigned as the job moves from
    stage to stage (bytes -> tensor -> logits -> predictions). Allocating a new
    job per stage would be cleaner on paper and would throw away the identity
    and the timings, which are the two things that must survive the whole trip.
    """

    id: str
    payload: Any
    future: asyncio.Future
    submitted_at: float = field(default_factory=perf_counter)
    #: Set when the job enters a stage's queue; read when it leaves it.
    entered_queue_at: float = field(default_factory=perf_counter)
    stage_ms: dict[str, float] = field(default_factory=dict)
    wait_ms: dict[str, float] = field(default_factory=dict)

    def complete(self) -> None:
        """Resolve with the final payload. Safe if the client already left."""
        if not self.future.done():
            self.future.set_result(
                Completion(
                    result=self.payload,
                    stage_ms=self.stage_ms,
                    wait_ms=self.wait_ms,
                    total_ms=(perf_counter() - self.submitted_at) * 1000.0,
                )
            )

    def fail(self, exc: BaseException) -> None:
        """Deliver a failure. Safe if the client already went away -- a client
        that timed out and disconnected must not turn into an unretrieved
        exception warning on the event loop."""
        if not self.future.done():
            self.future.set_exception(exc)
        elif self.future.cancelled():  # pragma: no cover - defensive
            pass
