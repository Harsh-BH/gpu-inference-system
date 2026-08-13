# Experiments

Every performance claim in this repo has to come from a row in this file.
Each experiment records **hypothesis, setup, measurement, result, explanation,
trade-off** — including the ones where the hypothesis was wrong, which are the
useful ones.

Hardware for all results below: RTX 3050 Laptop (Ampere SM 8.6, 6 GB, 20 SMs),
driver 610.43.03, PyTorch 2.13.0+cu130, ResNet-18, 224x224.

> Note on laptop GPUs: this chip has a 60 W power cap and clocks down under
> sustained load. Absolute numbers drift a few percent between runs; the
> comparisons are taken back-to-back so the *ratios* stay meaningful.

---

## Experiment 1 — CPU vs GPU inference

**Hypothesis.** GPU inference will be several times faster than CPU for
ResNet-18, and that speedup will carry through to end-to-end request latency.

**Setup.** `scripts/infer.py data/dog.jpg --runs 50`, batch 1, FP32, TF32 off.
Median of 50 runs, engine warmed up first. Same preprocessing code on both
paths; only `DEVICE` changes.

**Measurement.**

| stage | GPU (`cuda:0`) | CPU |
|---|---|---|
| preprocess | 17.770 ms | 23.109 ms |
| host → device | 0.227 ms | — |
| **inference** | **2.982 ms** | **16.090 ms** |
| device → host | 0.062 ms | — |
| postprocess | 0.109 ms | 0.139 ms |
| **end to end** | **21.248 ms** | **39.367 ms** |

**Result.** The GPU is **5.4x faster at the model** but only **1.85x faster at
the request**.

**Explanation.** Amdahl's law, in one table. Preprocessing is a serial CPU
stage that the GPU cannot help with, and at batch 1 it costs ~6x the GPU
inference it feeds. Accelerating a stage that is 14% of the request caps the
achievable speedup no matter how fast the accelerator is.

The secondary observation is that preprocessing itself got *slower* in the CPU
run (23.1 vs 17.8 ms). Same code, same image. PyTorch's CPU inference saturates
the cores that PIL's decode and resize also need, so the two stages contend.
On the GPU path they do not.

**Trade-off.** The honest conclusion is not "use a GPU", it is "at batch 1 this
workload is CPU-bound and a GPU is poor value". The GPU wins decisively only
once batching amortises the fixed per-call overhead across many images
(Phase 4) — or once preprocessing is moved off the critical path. That is why
the queue and batch manager exist at all, and why they come before TensorRT in
the build order: optimising the 3 ms before fixing the 18 ms would be
optimising the wrong thing.

---

## Experiment 2 — TF32 is on by default, and it is not FP32

**Hypothesis.** `PRECISION=fp32` on PyTorch gives IEEE-754 single-precision
inference, suitable as the numerical baseline every other runtime is compared
against.

**Setup.** Same input through the CPU engine and the CUDA engine, comparing
raw logits. `torch.backends.cudnn.allow_tf32` toggled between runs; everything
else held constant.

**Measurement.**

| `cudnn.allow_tf32` | max &#124;gpu − cpu&#124; on logits | top-1 agreement |
|---|---|---|
| `True` (**PyTorch's default**) | 3.56e-03 | yes |
| `False` | 5.25e-06 | yes |

**Result.** Hypothesis was wrong. PyTorch defaults to TF32 for convolutions on
Ampere and later, and the default baseline disagrees with CPU FP32 by **680x
more** than true FP32 does.

**Explanation.** TF32 is a Tensor Core format: FP32's 8 exponent bits, but only
10 mantissa bits instead of 23. Tensors remain `float32` in memory and every
shape, dtype and device print looks identical — only the *accumulation* inside
the convolution changes. `torch.backends.cuda.matmul.allow_tf32` defaults to
`False` while `torch.backends.cudnn.allow_tf32` defaults to `True`, so the
behaviour also differs between the matmul and convolution paths of the same
model.

Top-1 predictions were unaffected here, which is exactly what makes it
dangerous: nothing looks broken.

**Trade-off.** TF32 is a good production default — real speedup on Tensor
Cores, negligible accuracy impact for classification. It is a bad *baseline*.
Phases 5, 7 and 9 all measure runtimes against "PyTorch FP32"; if that is
secretly TF32, then part of every accuracy delta later attributed to ONNX
Runtime or TensorRT is really this flag.

Resolution: promoted to an explicit `ALLOW_TF32` setting, defaulting to `false`.
The project's rule about never silently converting a model applies to the
framework's silent conversions too. `tests/test_pytorch_engine.py` pins both
branches so a future torch release flipping the default produces a failing test
rather than a quietly wrong benchmark.

---

## Experiment 3 — CUDA timing without synchronisation

**Hypothesis.** Wrapping a GPU op in `time.perf_counter()` measures how long
the GPU took.

**Setup.** `scripts/check_gpu.py`, 2048x2048 FP32 matmul, warmed up, timed with
and without `torch.cuda.synchronize()`.

**Measurement.**

| timing method | reported |
|---|---|
| `perf_counter` only | 0.065 ms |
| `perf_counter` + `synchronize()` | 5.723 ms |

**Result.** The unsynchronised measurement is **89x too fast**.

**Explanation.** CUDA kernel launches are asynchronous. `torch.matmul` returns
to Python once the work is *enqueued* on the stream, not once it has run. The
0.065 ms is the cost of the launch. `synchronize()` blocks the host until the
stream drains, which is the only way a host-side clock can observe GPU
completion.

**Trade-off.** Synchronising also prevents overlap between stages, so a
per-stage-synchronised measurement slightly perturbs what it measures. For a
single synchronous `predict()` there is nothing to overlap — one stream,
executed in issue order — so the cost here is a few microseconds on an
already-idle stream. Phases 18 and 19 are where overlap becomes real, and they
are measured separately for that reason.

---

## Experiment 4 — Where the VRAM actually goes

**Hypothesis.** VRAM usage is roughly the model weights (~45 MB for ResNet-18
FP32) plus the input batch, and it scales linearly with batch size.

**Setup.** `scripts/gpu_memory_report.py`, FP32, allocator capped at 75% of
5.67 GiB. `reset_peak_memory_stats()` before each measurement so peaks are
isolated rather than cumulative.

**Measurement.**

| stage | allocated | reserved | transient peak |
|---|---|---|---|
| CUDA context (before any tensor) | 0 | 0 | — (104.81 MiB of *device* memory) |
| model load | +44.69 MiB | +62.00 MiB | 44.69 MiB |
| warmup, 10 × batch 1 | +8.12 MiB | +32.00 MiB | 35.18 MiB |

| batch | input | transient peak | peak / image |
|---|---|---|---|
| 1 | 588.00 KiB | 27.05 MiB | 27.05 MiB |
| 2 | 1.15 MiB | 28.11 MiB | 14.05 MiB |
| 4 | 2.30 MiB | 30.79 MiB | 7.70 MiB |
| 8 | 4.59 MiB | 53.59 MiB | 6.70 MiB |
| 16 | 9.19 MiB | 109.19 MiB | 6.82 MiB |
| 32 | 18.38 MiB | 214.38 MiB | 6.70 MiB |

**Result.** Hypothesis was directionally right and quantitatively wrong in
three ways.

1. **The CUDA context costs 104.81 MiB before a single tensor exists** and is
   invisible to `memory_allocated()`.
2. **Memory is affine, not proportional.** 1 → 32 images is 32× the work but
   only 7.9× the peak. Beyond batch 4 the marginal cost settles at ~6.7 MiB per
   image; below that a fixed component dominates.
3. **The input batch is negligible.** At batch 32 the input is 18 MiB of a
   214 MiB peak. Activations, not inputs, are what fill a GPU.

**Explanation.** Three separate pools are in play. The weights are paid once
and never grow. The activations are transient — under `inference_mode` each
intermediate is freed once the next layer consumes it, so the peak is the
largest few tensors alive at once, not their sum. And `reserved` exceeds
`allocated` throughout because PyTorch's caching allocator does not `cudaFree`
on every tensor death; `cudaMalloc`/`cudaFree` synchronise the device, so
per-tensor freeing would serialise the pipeline.

The affine shape is why per-image memory falls with batch size: the fixed
component (cuDNN workspace, weights already resident) is amortised over more
images, exactly as the fixed *time* cost is.

**Trade-off.** Larger batches are more memory-efficient per image right up to
the point where they are not available at all. The peak is what OOMs, the peak
is transient, and it is invisible to any monitoring that samples current
allocation. `MAX_BATCH_SIZE` is therefore a memory ceiling first and a
throughput knob second.

---

## Experiment 5 — Deliberate OOM, and surviving it

**Hypothesis.** Doubling batch size until CUDA OOM will raise a catchable
error, and the process can continue serving afterwards.

**Setup.** Same script, allocator capped at 4.25 GiB via
`set_per_process_memory_fraction()` — deliberately, because this GPU also
drives the display and exhausting physical VRAM can hang the compositor. That
cap is itself the production technique for stopping one model from starving
everything else on a shared card.

**Measurement.**

| batch | outcome | transient peak |
|---|---|---|
| 64 | ok | 428.75 MiB |
| 128 | ok | 857.50 MiB |
| 256 | ok | 1.67 GiB |
| 512 | ok | 3.35 GiB |
| 1024 | **OOM** | — |

After `empty_cache()`, batch 1 served normally: `(1, 1000)` logits in 4.44 ms.

**Result.** Confirmed. Batch 1024 needed roughly 6.7 GiB against a 4.25 GiB
ceiling and failed; the process recovered fully.

**Explanation.** Activations scale linearly while weights stay fixed, so a
large enough batch needs a contiguous block the allocator cannot supply. The
recovery matters more than the failure: the engine catches
`torch.cuda.OutOfMemoryError`, calls `empty_cache()` so the failed request's
reserved blocks do not poison the next one, and re-raises as `EngineError` —
which the API layer maps to 503 rather than a stack trace.

**Trade-off.** `empty_cache()` on the OOM path is correct but not free: it
synchronises the device and makes subsequent allocations slower until the pool
regrows. That is the right price on a failure path and the wrong one on the
request path, which is why it appears in exactly two places — OOM recovery and
model unload.

---

## Experiment 6 — The 8 MiB that survives `unload()`

**Hypothesis.** After unloading the model and calling `empty_cache()`,
`memory_allocated()` returns to zero.

**Setup.** Load, run one inference, unload, `gc.collect()`, `empty_cache()`,
then walk `gc.get_objects()` for live CUDA tensors. Then vary
`CUBLAS_WORKSPACE_CONFIG`.

**Measurement.**

```
after load          44.69 MiB
after 1 predict     52.82 MiB      (+8.12 MiB)
after unload + gc    8.12 MiB
live CUDA tensors reachable from Python:  none
```

| `CUBLAS_WORKSPACE_CONFIG` | residual |
|---|---|
| `:4096:2:16:8` (torch default) | 8.12 MiB — 4096 KiB × 2 + 16 KiB × 8 |
| `:1024:4` | 4.00 MiB |
| `:16:8` | 128.00 KiB |

**Result.** Hypothesis was wrong. The residual is the **cuBLAS workspace**,
allocated on the first GEMM (ResNet-18's final `fc` layer) and cached per
handle+stream for the lifetime of the process. cuDNN was ruled out first —
disabling it left the residual unchanged.

**Explanation.** cuBLAS keeps a scratch buffer per handle so that GEMM kernels
have somewhere to stage partial results without allocating per call. It belongs
to the library, not to any Python object, so no amount of `del` or
`gc.collect()` reclaims it.

**Trade-off.** Shrinking it via `CUBLAS_WORKSPACE_CONFIG` is possible and
almost always wrong — a smaller workspace forces cuBLAS toward slower kernels.
8 MiB on a 6 GB card is not worth optimising. It is worth *knowing*, because
"`memory_allocated()` is non-zero after I freed everything" otherwise reads as
a leak, and hunting a leak that does not exist costs a day.

---

## Experiment 7 — Batch size vs latency, throughput and VRAM

**Hypothesis.** Larger batches raise throughput and raise latency. Throughput
should scale close to linearly until the GPU saturates.

**Setup.** `scripts/benchmark.py --iterations 500`, ResNet-18 FP32, TF32 off,
20 warmup iterations **per batch size** (cuDNN caches a convolution algorithm
per input shape, so a new batch size is a new autotune), `empty_cache()`
between sizes so each peak is its own.

**Measurement.**

| batch | p50 ms | p95 ms | img/s | ms/img | peak VRAM | SM clock |
|---|---|---|---|---|---|---|
| 1 | 2.50 | 4.50 | 331.7 | 2.505 | 79.87 MiB | 1800 MHz |
| 2 | 3.77 | 5.71 | 450.8 | 1.886 | 80.92 MiB | 1747 MHz |
| 4 | 8.23 | 8.47 | 525.7 | 2.056 | 83.60 MiB | 1740 MHz |
| 8 | 13.35 | 15.22 | 591.7 | 1.668 | 106.41 MiB | 1642 MHz |
| **16** | **23.01** | **24.65** | **702.5** | **1.438** | 162.00 MiB | 1680 MHz |
| 32 | 46.83 | 48.94 | 681.8 | 1.464 | 267.19 MiB | 1657 MHz |

| step | throughput | p50 latency | peak VRAM |
|---|---|---|---|
| 1 → 2 | 1.36× | 1.51× | 1.01× |
| 2 → 4 | 1.17× | 2.18× | 1.03× |
| 4 → 8 | 1.13× | 1.62× | 1.27× |
| 8 → 16 | 1.19× | 1.72× | 1.52× |
| 16 → 32 | 0.97× | 2.04× | 1.65× (within noise) |

**Result.** Throughput plateaus at batch 16. **16× the batch buys 2.1× the
throughput, not 16×.** Past the plateau, batch 32 costs 2.0× the latency and
1.6× the VRAM for 0.97× the throughput.

**Explanation.** Batching amortises fixed per-call cost — kernel launch
overhead, Python dispatch, the weight reads that happen once regardless of
batch size — across more images. That is why `ms/img` falls from 2.505 to
1.438. But once the SMs are saturated the work becomes compute-bound, and
adding images just adds proportional work. On this card, 20 SMs at a 60 W cap
reach that point around batch 16.

Latency, meanwhile, grows roughly linearly with batch size throughout, because
every image in a batch waits for the whole batch to finish.

**Trade-off.** This is the entire latency/throughput dial. Batch 1 gives
2.50 ms p50 and 332 img/s; batch 16 gives 703 img/s at 23.01 ms p50 — 2.1× the
throughput for 9.2× the latency. Neither is "better". The right batch size is
whatever the latency SLO permits, and the answer to "is bigger batch better" is:
better until the accelerator is full, pure cost afterwards.

For dynamic batching (Phase 11) this sets `MAX_BATCH_SIZE`: above 16 there is
nothing left to win here, and every millisecond spent waiting to fill a larger
batch is charged to a client.

---

## Experiment 8 — The measurement that was wrong

**Hypothesis.** (Stated after the first run of Experiment 7.) Throughput peaks
at batch 16 and *declines* at batch 32 — 681.4 vs 666.0 img/s — so oversized
batches actively hurt.

**Setup.** Re-run batch 16 and batch 32 alone, in separate processes, rather
than as the 5th and 6th rows of a sequential sweep. Then sample
`nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu` during sustained
load at each size.

**Measurement.**

| | in sweep (5th/6th) | run in isolation |
|---|---|---|
| batch 16 | 681.4 img/s | 662.9 img/s |
| batch 32 | 666.0 img/s | 665.2 img/s |

Under sustained load at *both* batch sizes:

```
batch 16   1665-1725 MHz   59.75-60.06 W   81-83 C   99% util
batch 32   1627-1717 MHz   59.33-59.74 W   80-82 C   99-100% util
idle       1912 MHz        44.42 W
```

**Result.** Hypothesis was wrong. In isolation the two batch sizes are
indistinguishable (662.9 vs 665.2), and batch 16 itself moved 2.8% between its
own two measurements. The "decline" was an artefact of running last.

**Explanation.** This is a 60 W laptop GPU and it is power-limited, not
thermally headroomed: it sits within 1% of its cap under load with SM clocks
8–15% below boost. A sequential sweep therefore measures each successive batch
size on a progressively hotter, slower device. In the final run the effect is
plainly visible — 1800 MHz at batch 1 declining to 1657 MHz at batch 32 as
temperature rises 77 → 85 °C.

**Trade-off / what changed.** Three things, all in the harness rather than in
the write-up:

1. `nvidia-ml-py` added, and `src/gpu/telemetry.py` records SM clock, power and
   temperature on **every** benchmark row. A benchmark that cannot see
   throttling will attribute it to whatever it happened to be varying.
2. `NOISE_FLOOR = 0.05`, derived from the measured run-to-run variation above.
   Steps inside it are printed as `(within noise)` instead of being read as
   results.
3. The plateau is reported as the *first batch size statistically tied with the
   best*, not the argmax — the argmax is what produced the false finding.

The direction of the bias is worth keeping in mind: it runs against large
batches, so the scaling in Experiment 7 is if anything understated.

---

## Experiment 9 — Engine throughput is not system throughput

**Hypothesis.** The batch sweep's peak, 703 img/s, is what this system can
serve.

**Setup.** Measure preprocessing on the same sample image (`data/dog.jpg`,
1546×1213 JPEG), median of 30 runs, single thread, and compare against the
engine's peak throughput.

**Measurement.**

| | |
|---|---|
| preprocessing | 21.88 ms/image |
| one CPU thread sustains | 45.7 img/s |
| engine peak | 702.5 img/s (batch 16) |
| **ratio** | **15.3×** |

**Result.** Hypothesis was wrong by more than an order of magnitude. A single
preprocessing thread can feed ~6.5% of what the GPU can consume.

**Explanation.** JPEG decode, resize and normalise are serial CPU work that
scales with *image* count, not batch count — batching does not help it at all.
Saturating this GPU needs roughly 15 concurrent preprocessing workers.

**Trade-off.** This is the number that orders the remaining phases. With one
preprocessing thread the batch manager will never assemble a full batch under
real load; it will hit `MAX_BATCH_WAIT_MS` waiting for images the CPU has not
decoded yet, and the batch-16 row above describes hardware the system cannot
reach. Making a 2.5 ms inference into 1 ms changes nothing while a 21 ms
serial stage stands in front of it — which is why the queue, dynamic batching
and concurrency phases come before TensorRT.

---

## Experiment 10 — FP32 vs TF32 vs FP16

**Hypothesis.** FP16 halves weight memory and roughly doubles throughput via
Tensor Cores, at some cost in numerical accuracy. TF32 sits between the two.

**Setup.** `scripts/benchmark_precision.py`, 300 iterations per (batch, mode).

Two methodology choices carried over from Experiment 8:

- **Interleaved and rotated.** All three precisions are measured back-to-back
  within each batch size, and the order rotates per batch size, so every mode
  runs first, middle and last across the sweep. Comparing precisions is where
  thermal drift would do the most damage — a "10% FP16 win" is worthless if
  FP16 simply ran first.
- **Agreement on real image content.** 32 random crops at varying scale from
  the sample photo, not gaussian noise. FP16's error depends on activation
  magnitude, and unstructured input does not produce activations that transfer.

**Measurement.**

Weights in VRAM (11,689,512 parameters):

| mode | weights | vs fp32 |
|---|---|---|
| fp32 | 44.69 MiB | 1.00× |
| tf32 | 44.69 MiB | 1.00× — identical, TF32 changes the maths not the bytes |
| fp16 | 22.96 MiB | 0.51× |

Agreement vs FP32:

| mode | top-1 | top-5 | max &#124;Δlogit&#124; | mean &#124;Δlogit&#124; | max Δconf |
|---|---|---|---|---|---|
| tf32 | 100.0% | 100.0% | 1.03e-02 | 1.28e-03 | 8.25e-04 |
| fp16 | 100.0% | 100.0% | 3.44e-02 | 4.59e-03 | 6.51e-03 |

Throughput and peak VRAM:

| batch | mode | p50 ms | img/s | peak VRAM | vs fp32 |
|---|---|---|---|---|---|
| 8 | fp32 | 15.34 | 537.6 | 106.41 MiB | 1.00× |
| 8 | tf32 | 11.67 | 650.2 | 106.41 MiB | 1.21× |
| 8 | **fp16** | **8.04** | **996.4** | **60.95 MiB** | **1.85×** |
| 16 | fp32 | 24.95 | 649.3 | 162.00 MiB | 1.00× |
| 16 | tf32 | 22.16 | 707.3 | 162.00 MiB | 1.09× |
| 16 | **fp16** | **14.97** | **1091.6** | **91.09 MiB** | **1.68×** |
| 32 | fp32 | 50.58 | 630.0 | 267.19 MiB | 1.00× |
| 32 | tf32 | 43.84 | 729.1 | 267.19 MiB | 1.16× |
| 32 | **fp16** | **27.65** | **1148.0** | **152.27 MiB** | **1.82×** |

**Result.** Hypothesis mostly confirmed, with one important exception.

- FP16: **~1.8× throughput, 0.51× weights, 0.57× peak VRAM, 100% top-1
  agreement.** A clear win at batch ≥ 8.
- TF32: a consistent **1.1–1.2×** for free — no storage change, no top-1 change.
- **Batch 1 is not a gain.** Across five independent runs FP16/FP32 at batch 1
  measured 1.52×, 1.03×, 1.31×, 1.19×, 1.12×, while FP32 at batch 1 stayed
  within 4% of itself. That range is jitter, not a speedup, and it is reported
  as such rather than cherry-picked.

**Explanation.** `--profile-kernels` replaces inference with evidence. At
batch 32 the hottest CUDA kernels are:

```
fp32   scudnn_128x64_relu_xregs      36.4%   CUDA cores, NCHW
       scudnn_winograd_128x128        26.4%   CUDA cores, Winograd
tf32   sm86_xmma_..._tf32f32_...      13.0%   Tensor Core
       nchwToNhwcKernel<float>        12.9%   layout conversion, no maths
fp16   batch_norm_transform<Half>     14.1%
       nchwToNhwcKernel<__half>       12.0%   layout conversion
       nhwcToNchwKernel<__half>       10.4%   layout conversion
       cutlass_tensorop_f16_s16816     9.3%   Tensor Core, 16x8x16 MMA
```

Three things follow. FP32 never touches a Tensor Core — it runs Winograd on
CUDA cores, which is why it is respectable rather than terrible. TF32 and FP16
do reach Tensor Cores (`xmma`, `cutlass_tensorop`, `s16816`). And **FP16 spends
22.4% of its GPU time purely converting NCHW to the NHWC layout Tensor Cores
want and back again**, per operator.

Why batch 1 gains nothing: at ~2.6 ms of GPU work the pipeline is bound by
kernel launches and layout conversions rather than arithmetic, and halving the
arithmetic does not help a workload that is not arithmetic-bound. Batching is
what makes the GPU compute-bound, and only then does precision pay.

**Trade-off.** FP16 is the right default for this workload *at batch ≥ 8*, and
the accuracy cost is real but small — top-1 identical on every sample tested,
with logit drift 3.4e-02 versus TF32's 1.03e-02. Where FP16 does bite is
dynamic range: float16 overflows above 65504 and underflows below ~6e-5, so
models with large activations, accumulations over long sequences, or
loss-scaling-sensitive layers can produce inf/nan where FP32 would not.
ResNet-18 classification is nowhere near those limits. A model that is would
need BF16 instead — same 8 exponent bits as FP32, 7 mantissa bits, so it
trades precision for range.

The 22.4% layout tax is the actionable finding: FP16 wins 1.8× *while* wasting
nearly a quarter of its time shuffling memory. Eliminating that is exactly what
TensorRT's whole-graph layout planning does, so Phase 8 has a concrete target
rather than a hope.

---

## Pending

| # | Experiment | Phase |
|---|---|---|
| 11 | PyTorch vs ONNX Runtime | 7 |
| 12 | ONNX Runtime vs TensorRT | 9 |
| 13 | Static vs dynamic batching | 11 |
| 14 | Cold start vs warm inference | 13 |
| 15 | Concurrent users | 12 |
| 16 | CUDA streams | 18 |
| 17 | Pinned memory and non-blocking transfer | 19 |
| 18 | CUDA Graphs | 20 |
| 19 | INT8 quantisation | 21 |
