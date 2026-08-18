# One request, from HTTP to CUDA kernel and back

The final report the project set out to produce. This traces a single
`POST /predict` through every layer, with the measured cost of each. Numbers are
from an RTX 3050 Laptop (Ampere SM 8.6, 20 SMs, 6 GB, 60 W cap) serving
TensorRT FP16 at `MAX_BATCH_SIZE=16` with 16 decode workers.

The path is a three-stage pipeline — `decode → infer → classify` — described in
[pipeline.md](pipeline.md). Each stage has its own bounded queue, so "where did
the time go" has a per-stage answer rather than one number.

---

## 1. HTTP arrives

A multipart POST reaches uvicorn, which parses headers and hands a
`starlette.UploadFile` to the handler. Nothing is buffered yet.

The handler reads the body in 64 KiB chunks, **counting as it goes**, and aborts
with 413 the moment the count passes `MAX_UPLOAD_BYTES`. Content-Length is not
trusted: it can be absent under chunked encoding and it can lie. A hostile
client cannot make the server allocate what it refuses to accept.

## 2. Submission — the pipeline takes the bytes

The raw bytes go straight to `pipeline.submit()`, which creates a `Job` (an id,
the payload, a Future) and `put_nowait`s it onto queue 0. Full → immediate 503
with `Retry-After`, never a wait. The handler then awaits and yields the loop.

Queueing the **bytes** rather than a decoded tensor is a change from the previous
architecture and it is the cheaper end to reject from: a rejection now costs
nothing, where before it discarded a completed 20 ms decode. It also holds less
memory — a ~100 KB JPEG against a 588 KiB tensor.

This is where HTTP's request/response shape and the GPU's batch shape are
reconciled: the request will be reunited with its own row of an output tensor
later, on a different task.

## 3. The decode stage — 19.9 ms, still the largest single cost

Sixteen workers over a thread pool, one image each, off the event loop.

```
JPEG bytes
  -> Image.open           header only, no decode yet
  -> pixel-count guard    a 10 KB PNG can declare 60000x60000 and ask for ~10 GB
  -> draft("RGB", 256)    libjpeg emits 1/2, 1/4 or 1/8 size straight from the
                          DCT coefficients — 2.47x on this photo, and exactly
                          nothing when no reduction stays above the crop
  -> exif_transpose       phone photos carry rotation in metadata, not pixels
  -> convert("RGB")       forces the decode; normalises grayscale/RGBA/CMYK
  -> resize shorter->256  aspect-preserving; squashing costs ~0.5% top-1
  -> center crop 224      matching how the weights were evaluated
  -> HWC uint8 -> CHW     one copy; conv kernels want channel-planar
  -> /255, -mean, /std    in place, no further allocations
```

Out comes `(3, 224, 224)` float32 — 150,528 elements, **588 KiB**.

This stage is still the system's bottleneck, and by a wide margin: across a
900-request run it accounted for **90.5% of all work done anywhere in the
pipeline**. `draft()` cut it from 21.9 ms to 8.6 ms on this image (Experiment
21), which is why it is 19.9 ms under load rather than 49 ms.

A corrupt image fails here, and fails **only its own request** — the stage
returns the exception in that item's slot rather than raising, so the fifteen
good images beside it are unaffected.

## 4. Batch formation — the infer stage collects

The single infer worker pulls from its queue until **either** 16 items are in
hand **or** the oldest has waited `MAX_BATCH_WAIT_MS`. The timer starts with the
first item and is never reset, so a steady trickle cannot postpone dispatch
forever.

In practice the deadline almost always fires first: a 900-request run averaged
**2.3 images per GPU call**. Decode cannot produce 16 tensors within 5 ms, so
the batch size is set by the upstream stage, not by the configured maximum.

`np.stack` copies the samples into one contiguous `(N, 3, 224, 224)` block.
The copy is required: those samples arrived at different times and are scattered
across the heap, while the DMA engine needs one contiguous region.

## 5. Host to device — 2.2 ms at batch 32

`engine.predict` runs via `asyncio.to_thread`; calling it inline would stall the
event loop for the entire forward pass.

`torch.from_numpy` wraps the numpy buffer with no copy. `.to(device, dtype)`
crosses PCIe into VRAM, casting to FP16 on the way. This is the only moment
"CPU memory" becomes "GPU memory".

The memory is pageable, so the driver stages it through an internal pinned
buffer. Phase 19 measured pinned memory as 1.16–1.54× faster here and not worth
the complexity: 0.22 ms against a 21.9 ms preprocessing stage.

## 6. CUDA — the kernels run

`context.set_input_shape` fixes the batch dimension, selecting kernels
specialised for it inside the engine's optimisation profile.
`execute_async_v3` enqueues the whole plan onto a stream.

What happens on the device:

- Each fused layer is a **kernel launch**. TensorRT has already fused conv +
  bias + ReLU into single kernels and planned the layout once for the whole
  graph, which is why it avoids the `nchwToNhwc`/`nhwcToNchw` transposes that
  cost PyTorch FP16 **22.4% of its GPU time**.
- A launch creates a **grid** of **blocks**. Each block is assigned to one of
  the 20 **SMs** and never migrates.
- Within a block, threads are grouped into **warps** of 32 that execute in
  lockstep. The warp is what the SM actually schedules; having many resident is
  how memory latency gets hidden.
- Convolutions are lowered to matrix multiplies and issued to **Tensor Cores** —
  `cutlass_tensorop_f16_s16816`, where `s16816` is the 16×8×16 MMA fragment
  shape. Each instruction multiplies small tiles and accumulates in FP32, which
  is why FP16 storage does not mean FP16 accumulation.
- Data moves registers → shared memory/L1 → L2 → VRAM, spanning roughly 1 to
  400 cycles.

`stream.synchronize()` blocks the host until the GPU is genuinely finished.
Without it every timing in this project would measure enqueue cost — Phase 0
shows that gap as 0.065 ms versus 5.723 ms on the same matmul.

**6.98 ms of compute at batch 32.** The equivalent PyTorch FP16 number is
23.14 ms.

## 7. Device to host — 0.13 ms

`(N, 1000)` logits come back: 4 KB per sample. Three orders of magnitude
smaller than the input, which is why the return trip never matters for
classification and would matter enormously for segmentation.

## 8. Demultiplexing — the correctness-critical step

The infer stage splits the output tensor into one row per item, and the pipeline
runner pairs those results back to their jobs:

```python
# src/stages/infer.py — row i belongs to item i
return [logits[i] for i in range(len(items))]

# src/pipeline/pipeline.py — the one place results meet jobs
for job, result in zip(batch, results, strict=True):
    ...
```

Get this wrong and every client receives a confident, plausible prediction
belonging to somebody else. No exception, no log line, no failing test unless
someone wrote one.

This is the single strongest argument for the pipeline being generic: the
mapping used to be hand-written inside each batching implementation, and now it
exists once, in a runner no stage can bypass. `strict=True` turns a length
mismatch into an immediate error, and a stage that returns the wrong number of
results fails every item in the batch rather than letting the runner guess.
`tests/test_pipeline.py::test_result_i_goes_to_job_i` writes per-request markers
into the input and checks they come back in the right place.

## 9. The classify stage — 0.09 ms

Runs **inline on the event loop** (`workers=0`): a thread hop costs ~50 µs
against 90 µs of work, so the pool would be most of the cost. That is a measured
exemption from the "never block the loop" rule, not a guess.

Softmax subtracts the row max before exponentiating — not an optimisation, a
correctness requirement: `exp(89)` overflows float32 to `inf`, and `inf/inf` is
`nan`, so a confident prediction would return `nan` confidence for every class.

`argpartition` finds the top 5 in O(n) rather than sorting 1000.

Note that `argmax(logits) == argmax(softmax(logits))` — softmax is monotonic, so
the *predicted class* never depends on it. Only the confidence number does,
which is why the cross-runtime comparisons report class agreement and confidence
drift separately.

## 10. Response

```json
{
  "request_id": "…", "model_name": "resnet18", "model_version": "v1",
  "backend": "tensorrt", "precision": "fp16",
  "prediction": "Samoyed", "confidence": 0.8852,
  "predictions": [ … top 5 … ],
  "latency": {
    "stages":  { "decode": 29.35, "infer": 14.30, "classify": 0.19 },
    "waits":   { "decode":  0.04, "infer":  5.32, "classify": 0.30 },
    "queued_ms": 5.66, "pipeline_ms": 49.57, "total_ms": 50.11
  }
}
```

(An idle-server sample, so `infer` still carries a cold-ish 14 ms and nothing is
queued. Under load the figures are the ones in the table below.)

The breakdown is keyed by stage name rather than by fixed fields, so a pipeline
that grows a stage reports it without a schema change. `stages` and `waits` are
kept apart because they diagnose opposite problems: rising **work** means the
stage itself got slower, rising **wait** with flat work means it is saturated.

The engine identity travels with every prediction. After a deploy changes
accuracy, "which engine produced this" must be answerable from the response
rather than inferred from what happened to be configured.

---

## Where the time actually goes

At 25 concurrent clients, **326 req/s, p50 72.8 ms, p99 122.2 ms**. Stage costs
are from `pipeline_stage_*_seconds` over 900 requests:

| stage | work | wait | share of all work | who |
|---|---|---|---|---|
| read + validate | <1 ms | — | — | CPU |
| **decode** | **19.3 ms** | 11.4 ms | **90.5%** | **CPU — the bottleneck** |
| infer | 1.9 ms | 3.4 ms | 9.0% | GPU (+ H2D/D2H) |
| classify | 0.09 ms | 0.15 ms | 0.4% | CPU |

Inside the infer stage, at batch 32: H2D 2.2 ms, compute 6.98 ms, D2H 0.13 ms.

The GPU sustains 3300 img/s and the system delivers 326 req/s — roughly **10% of
the accelerator**, up from 7%. Decoding is *still* 90% of the work after being
made 2.5× faster, which is the most honest summary of this system there is.

Everything follows from that. What has been tried, with numbers in
[experiments.md](experiments.md):

- **Scaled DCT decode** (`draft()`) — 1.62× end-to-end, p99 *fell*. Shipped.
- **GPU decode via nvJPEG** — 0.14× against 16 CPU threads. Rejected.
- **A faster kernel** — TensorRT made the model 5.2× faster and moved end-to-end
  throughput far less, exactly as Amdahl's law says it must.

What is left is not a faster decode but fewer of them per core: process
replication behind a load balancer, so decoding escapes one GIL entirely.
