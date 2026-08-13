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

## Pending

| # | Experiment | Phase |
|---|---|---|
| 7 | Batch size vs latency and throughput | 4 |
| 8 | FP32 vs FP16 vs TF32 | 5 |
| 9 | PyTorch vs ONNX Runtime | 7 |
| 10 | ONNX Runtime vs TensorRT | 9 |
| 11 | Static vs dynamic batching | 11 |
| 12 | Cold start vs warm inference | 13 |
| 13 | Concurrent users | 12 |
| 14 | CUDA streams | 18 |
| 15 | Pinned memory and non-blocking transfer | 19 |
| 16 | CUDA Graphs | 20 |
| 17 | INT8 quantisation | 21 |
