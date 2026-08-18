# The pipeline

`src/pipeline/` is a small staged-execution runtime. It knows nothing about
images, models, GPUs or HTTP — that is what makes it the part of this project
you copy into the next one.

This document is the guide to using it as a template.

---

## The idea in one paragraph

A serving system is a sequence of transformations with wildly different costs
and wildly different parallelism. Decoding a JPEG is ~20 ms of CPU that
parallelises perfectly across threads. A GPU forward pass is ~2 ms that must be
serialised through one CUDA context but wants many images at once. Softmax is
0.1 ms that is not worth a thread hop. Those are all the same *shape* of thing —
items in, items out — differing only in how they should be **run**. So the shape
is defined once (`Stage`) and the running is described separately (`StageSpec`).

A stage author writes a pure transformation. Queues, thread pools, batch
deadlines, backpressure, result mapping, per-stage timing, failure isolation and
an ordered startup and drain all come from the runner, for free, and cannot be
implemented wrongly because they are not implemented per stage at all.

---

## The three types

```python
from src.pipeline import Pipeline, Stage, StageSpec

class Double(Stage[int, int]):
    name = "double"
    def process(self, items: list[int]) -> list[int]:
        return [i * 2 for i in items]

pipeline = Pipeline([StageSpec(Double())])
pipeline.start()
completion = await pipeline.submit(21)     # Completion(result=42, ...)
await pipeline.stop()
```

| type | what it is |
|---|---|
| `Stage[In, Out]` | **what** the transformation does. You implement `process()`, and optionally `setup()`/`teardown()` if the stage owns a resource. |
| `StageSpec` | **how** to run it: `workers`, `max_batch`, `max_batch_wait_ms`, `capacity`. |
| `Pipeline` | the ordered list, plus every runtime concern. |

The split between the first two is the load-bearing one. It is what let this
project move image decoding between three different execution strategies by
editing one line, and measure each.

---

## Writing a stage

```python
class ImageDecodeStage(Stage[bytes, np.ndarray]):
    name = "decode"

    def __init__(self, preprocessor): self._preprocessor = preprocessor

    def process(self, items: list[bytes]) -> list[np.ndarray | Exception]:
        out = []
        for data in items:
            try:
                out.append(self._preprocessor.from_bytes(data))
            except PreprocessingError as exc:
                out.append(exc)          # fails one item, not the batch
        return out
```

### The contract, which the runner enforces rather than trusts

1. **`len(output) == len(input)`.** A mismatch fails every item in the call with
   a `StageContractError`. The runner refuses to guess which result belongs to
   whom, because guessing is precisely how everyone ends up with somebody else's
   answer.
2. **`output[i]` belongs to `input[i]`.** This is *the* correctness-critical
   invariant in a batching server. Get it wrong and every client receives a
   confident, plausible prediction belonging to another request — no exception,
   no log line, nothing to alert on. It is now one `zip(..., strict=True)` in one
   runner, guarded by one test, and no stage performs it.
3. **`items` is never empty.**
4. A stage with `workers > 1` must be safe to call from several threads at once.
   In practice: hold no mutable per-call state on `self`.

### The two ways to fail, and how to choose

| you do | blast radius | use when |
|---|---|---|
| `raise` | every item in the call | the batch genuinely did not happen — CUDA OOM, dead engine, lost connection |
| `return Exception` in a slot | that item only | one bad input among good ones — a corrupt JPEG, a validation failure |

Both exist because both occur. A stage that could only raise would force a
batched decoder to fail fifteen good images because of one bad one.

### Owning a resource

```python
def setup(self):     self._engine.load(); self._engine.warmup(10)
def teardown(self):  self._engine.unload()
```

`setup()` runs once, in pipeline order, **before any worker exists**, so a stage
that cannot acquire what it needs fails startup instead of failing requests. If
a later stage's setup fails, the earlier ones are torn down in reverse — a
half-initialised pipeline never keeps a GPU.

`teardown()` runs after the last worker has stopped touching the stage. It must
be idempotent and must not raise; it runs on the shutdown path, where one
stage's failure must not skip the rest.

---

## Configuring how a stage runs

```python
StageSpec(stage, workers=1, max_batch=1, max_batch_wait_ms=0.0, capacity=128)
```

### `workers`

| value | behaviour | correct for |
|---|---|---|
| `0` | runs inline on the event loop, no thread hop (~50 µs saved) | stages measured in **microseconds**. `classify` is 0.09 ms. |
| `1` | one worker in one thread | anything holding a single CUDA context — a second worker would serialise on it anyway while making the VRAM ceiling harder to reason about |
| `N` | N workers over an N-thread pool | CPU work that **releases the GIL**. PIL does during decode and resize, which is the only reason threads help at all. |

`workers=0` is an opt-out for cheap stages, and it should be a *measured*
exemption, not a guess.

### `max_batch` and `max_batch_wait_ms`

A batch dispatches when **either** `max_batch` items are in hand **or** the
oldest has waited `max_batch_wait_ms`.

The deadline is what makes it safe: without it, a batch that never fills waits
forever. The timer starts with the **first** item and is never reset by later
arrivals — otherwise a steady trickle postpones dispatch indefinitely.

Batch only where there is a **per-call** fixed cost to amortise. Inference has
one (kernel launches, the H2D round trip, the Python/CUDA boundary), which is
why batch 16 is 2× the throughput of batch 1. Decoding has none — the cost is
purely per-image — so batching it would buy nothing and would couple sixteen
requests' fates together. That is the whole reason `decode` runs `max_batch=1`
and `infer` runs `max_batch=16`.

`max_batch_wait_ms` is added to the p99 of an idle system for nothing: with one
request in flight there is nobody to batch with and it waits anyway. Under load
the queue is never empty and batches fill before the deadline, so it costs
nothing. Tuning it is tuning the latency SLO.

### `capacity` and backpressure

Every stage has a bounded input queue.

- **Queue 0 rejects.** Full → `PipelineFull` → HTTP 503 with `Retry-After`,
  immediately, never a wait. Waiting for space *is* the unbounded behaviour,
  just with the backlog held in suspended coroutines instead of a list.
- **Every other queue blocks.** A full downstream queue stalls the upstream
  worker, which stalls its own queue, and the stall propagates back to queue 0
  where it becomes a 503.

The asymmetry is deliberate: dropping a request that has already been decoded
throws away the most expensive work the system does.

An unbounded queue does not absorb overload, it defers it. The backlog grows,
waits grow with it, clients time out and retry, the retries enqueue too, and
memory climbs until the kernel OOM-kills the process and every in-flight request
dies at once. Backpressure is a feature.

---

## What you get back

```python
completion = await pipeline.submit(payload, job_id="req-123")
```

```python
Completion(
    result   = <the last stage's output>,
    stage_ms = {"decode": 19.9, "infer": 1.9, "classify": 0.1},   # working
    wait_ms  = {"decode":  0.1, "infer": 5.3, "classify": 0.3},   # queued
    total_ms = 27.6,
)
```

`stage_ms` and `wait_ms` diagnose opposite problems and are never summed into
one number:

- rising **work** → the stage itself got slower (a bigger image, a throttled GPU)
- rising **wait** with flat work → the stage is saturated; the fix is capacity or
  concurrency, not a faster implementation

The pipeline applies no deadline of its own. Impose one at the call site:

```python
await asyncio.wait_for(pipeline.submit(data), timeout=10.0)
```

---

## Observability

Pass `on_stage=` and every stage reports itself after every batch:

```python
Pipeline(specs, on_stage=metrics.record_stage)
```

```python
StageReport(stage="decode", batch_size=1, wait_ms=0.1, work_ms=19.9,
            succeeded=1, failed=0)
```

The pipeline never imports Prometheus — it measures and reports, and the
application decides what that means. `src/monitoring/metrics.py` maps a report
onto one instrument per *kind* of measurement, labelled by stage:

```
pipeline_stage_work_seconds{stage="decode"}
pipeline_stage_wait_seconds{stage="decode"}
pipeline_stage_batch_size{stage="infer"}
pipeline_stage_items_total{stage="decode", outcome="ok"|"error"}
pipeline_stage_queue_depth{stage="decode"}
```

So `sum by (stage) (...)` answers "which stage is the bottleneck" for a pipeline
the metrics module has never heard of. Adding a stage requires no code there at
all. From a real 900-request run on this box:

| stage | total work | share |
|---|---|---|
| decode | 17.40 s | **90.5%** |
| infer | 1.74 s | 9.0% |
| classify | 0.08 s | 0.4% |

That table is the entire performance story of this system, and it came out of
`/metrics` with no special instrumentation.

An observer that raises is logged and swallowed. Monitoring must never break
serving.

---

## Lifecycle

```
start()   setup() every stage in order  ->  start workers  ->  serving
stop()    refuse new work  ->  drain in-flight  ->  fail the remainder
          ->  shut down pools  ->  teardown() in reverse
```

`stop()` is a **drain, not an exit**. Workers keep processing until their queues
are empty, bounded by `drain_timeout`. Anything still queued after that is failed
with a clean error, and anything a cancelled worker was holding is failed too —
otherwise those clients wait for their own timeouts to expire, which is exactly
the hanging shutdown the drain exists to prevent.

---

## This project's pipeline

From `src/main.py`:

```python
Pipeline([
    StageSpec(decode,   workers=16, max_batch=1),                        # CPU, parallel
    StageSpec(infer,    workers=1,  max_batch=16, max_batch_wait_ms=5),  # GPU, serial, batched
    StageSpec(classify, workers=0,  max_batch=64),                       # inline, trivial
], on_stage=metrics.record_stage)
```

```
POST /predict
  -> [queue] -> decode   x16 workers   bytes -> (3,224,224) float32
  -> [queue] -> infer    x1  batch 16  -> (1000,) logits
  -> [queue] -> classify inline         -> top-5 predictions
  -> 200 JSON
```

Every transformation is in that list. The HTTP handler reads bytes, calls
`submit()`, and serialises the answer — it does not know that decoding, queues,
batching or softmax exist.

---

## Using this as a template

The boundary to cut along:

```
src/pipeline/     KEEP     domain-free runtime, no changes needed
src/stages/       REPLACE  your transformations
src/main.py       EDIT     the list in build_pipeline()
```

1. **Write your stages.** One class per transformation, `process()` only.
   Reuse existing logic rather than reimplementing it inside a stage — every
   stage in `src/stages/` is a thin adapter over a module that already existed
   and is already tested.
2. **Decide how each runs.** Ask two questions per stage: *does it release the
   GIL?* (→ `workers > 1`) and *does it have a per-call fixed cost?* (→
   `max_batch > 1`). If neither, `workers=1, max_batch=1`.
3. **Compose them** in `build_pipeline()`.
4. **Measure before tuning.** `pipeline_stage_work_seconds` will tell you which
   stage is 90% of the work. It is rarely the one you expected — on this project
   the GPU turned out to be 9%.

### Things worth not re-deriving

- Name stages in lowercase and keep the names stable. They are metric labels and
  JSON keys; changing one breaks dashboards. Duplicates are refused at
  construction because they would silently merge two stages into one number.
- A stage that grows past ~100 lines is usually two stages.
- Put the deadline at the call site, not in the pipeline.
- `capacity` on stage 0 is your admission control. It is the only queue a client
  ever sees.

---

## What is deliberately not here

- **Branching or fan-out.** Stages are a line, not a DAG. Every workload this
  project has is a line, and a DAG needs a routing policy, join semantics and a
  cycle check — none of which anything here would use. Add it when a real branch
  exists.
- **Retries.** A retry policy needs to know which failures are transient, which
  is domain knowledge the runner does not have. Retry in the stage, or at the
  client.
- **Cross-process stages.** Threads reach the GIL ceiling eventually, and the
  answer is process replication behind a load balancer (see
  `docs/architecture.md`), not a shared-memory transport in this runtime.
- **Priorities.** One FIFO per stage. A priority queue needs a starvation policy
  and nothing here has two classes of traffic.

Each of those is a real feature and each is absent because nothing measured here
needs it.
