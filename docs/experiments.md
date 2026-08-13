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

## Pending

| # | Experiment | Phase |
|---|---|---|
| 4 | Batch size sweep (VRAM, latency, throughput) | 4 |
| 5 | FP32 vs FP16 vs TF32 | 5 |
| 6 | PyTorch vs ONNX Runtime | 7 |
| 7 | ONNX Runtime vs TensorRT | 9 |
| 8 | Static vs dynamic batching | 11 |
| 9 | Cold start vs warm inference | 13 |
| 10 | GPU memory usage and OOM | 3 |
| 11 | Concurrent users | 12 |
| 12 | CUDA streams | 18 |
| 13 | Pinned memory and non-blocking transfer | 19 |
| 14 | CUDA Graphs | 20 |
| 15 | INT8 quantisation | 21 |
