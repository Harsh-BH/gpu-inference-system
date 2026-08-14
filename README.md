# gpu-inference-system

A production-shaped image inference server, built from scratch to understand every layer
between an HTTP request and a CUDA kernel.

The goal is **not** "call `model(x)` and return a label". The goal is to be able to trace
and defend this path end to end:

```
HTTP request -> queue -> batch -> preprocess -> tensor -> H2D copy
             -> runtime (PyTorch | ONNX Runtime | TensorRT) -> CUDA -> SMs/Tensor Cores
             -> output tensor -> D2H copy -> postprocess -> HTTP response
```

## Why GPU inference at all

A ResNet-18 forward pass at 224x224 is ~1.8 GFLOPs. A modern CPU core does a few GFLOP/s
on this kind of work; an RTX 3050 does ~9 TFLOP/s FP32 and ~18 TFLOP/s FP16 via Tensor
Cores. The arithmetic is embarrassingly parallel — every output pixel of a convolution is
independent — which is exactly the shape of problem a GPU's thousands of ALUs exist for.

The catch, and the actual subject of this project: **the GPU is rarely the bottleneck.**
Preprocessing on the CPU, PCIe transfer, kernel launch overhead, Python's GIL, and requests
sitting in a queue routinely dominate. Making a model fast is easy. Making a *system* fast
requires measuring where the time actually goes — which is why nearly every phase here ends
in a benchmark rather than a claim.

## Hardware this was developed against

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop (Ampere, SM 8.6) |
| VRAM | 6 GB |
| Tensor Cores | 3rd gen — FP16, BF16, TF32, INT8 |
| Driver | 610.43.03 (CUDA 13.3) |

6 GB is a real constraint and that is deliberate: it forces the memory phases to be honest
rather than theoretical.


## Quickstart

```bash
uv sync                                     # core + PyTorch backend
uv run python scripts/check_gpu.py          # verify the GPU stack (exits non-zero if not)
uv run python scripts/fetch_model.py        # provision models/resnet18/v1/

mkdir -p data && curl -sSL -o data/dog.jpg \
  https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg

uv run python scripts/infer.py data/dog.jpg --trace --runs 50
```

Later phases add the optional backends:

```bash
uv sync --extra onnx       # ONNX Runtime
uv sync --extra trt        # TensorRT
```

Configuration is entirely environment-driven — see `.env.example`. No model path, device,
batch size, or precision is hardcoded anywhere in `src/`. CLI flags override `.env`, which
overrides the defaults in `src/config.py`.

## First results

Batch 1, FP32, median of 50 warm runs. Full write-ups in [docs/experiments.md](docs/experiments.md).

```
preprocess       17.770 ms   <- CPU: JPEG decode, resize, crop, normalise
host -> device    0.227 ms
inference         2.982 ms   <- GPU
device -> host    0.062 ms
postprocess       0.109 ms
```

Three findings already worth the build:

**The GPU is not the bottleneck.** Preprocessing costs ~6x the inference it feeds. Against
a CPU-only run the GPU is 5.4x faster *at the model* but only 1.85x faster *at the request*
— Amdahl's law, measured. This is why the queue and batch manager come before TensorRT in
the build order: optimising the 3 ms while ignoring the 18 ms would be optimising the wrong
thing.

**`PRECISION=fp32` was not FP32.** PyTorch enables TF32 convolutions by default on Ampere,
cutting mantissa bits from 23 to 10 while tensors still report `float32`. Measured against
the CPU reference, the default disagreed by 3.56e-03 versus 5.25e-06 with TF32 off — 680x
worse, silently. It is now an explicit `ALLOW_TF32` flag, defaulting to off, so the FP32
baseline the other runtimes get compared against is genuinely FP32.

**Naive GPU timing lies by ~89x.** The same matmul reports 0.065 ms without
`torch.cuda.synchronize()` and 5.723 ms with it, because kernel launches are asynchronous.
Every timing in this repo synchronises explicitly.

### VRAM, fully accounted

`uv run python scripts/gpu_memory_report.py`

```
CUDA context                104.81 MiB   before a single tensor exists
ResNet-18 FP32 weights       44.69 MiB   (44.59 MiB predicted)
peak, batch 1                27.05 MiB
peak, batch 32              214.38 MiB   ~6.7 MiB marginal per image
cuBLAS workspace              8.12 MiB   survives unload(), owned by no Python object
```

Memory is **affine, not proportional**: 1 → 32 images is 32× the work but only 7.9× the
peak, because weights are paid once and only activations scale. The script also drives a
deliberate OOM (batch 1024 against a 4.25 GiB cap) and shows the process serving normally
afterwards — the allocator is capped with `set_per_process_memory_fraction` rather than
exhausting physical VRAM, since this GPU also drives the display.

### Batching: better until the accelerator is full

`uv run python scripts/benchmark.py --iterations 500`

| batch | p50 ms | img/s | ms/img | peak VRAM | SM clock |
|---|---|---|---|---|---|
| 1 | 2.50 | 331.7 | 2.505 | 79.87 MiB | 1800 MHz |
| 8 | 13.35 | 591.7 | 1.668 | 106.41 MiB | 1642 MHz |
| **16** | **23.01** | **702.5** | **1.438** | 162.00 MiB | 1680 MHz |
| 32 | 46.83 | 681.8 | 1.464 | 267.19 MiB | 1657 MHz |

**16× the batch buys 2.1× the throughput, not 16×.** Throughput plateaus at batch 16;
past it, batch 32 costs 2.0× the latency and 1.6× the VRAM for 0.97× the throughput.

The first reading of this sweep said throughput *dropped* at batch 32. That was wrong —
re-running the two sizes in isolation gave 662.9 and 665.2 img/s, indistinguishable. This
is a 60 W laptop GPU and the sweep measures each successive batch size on a hotter, slower
device (1800 → 1657 MHz as temperature climbs 77 → 85 °C). Every benchmark row now records
SM clock, power and temperature, and differences under a measured 5% noise floor are
reported as `(within noise)` instead of as findings.

**And the engine is not the system.** One preprocessing thread sustains ~46 img/s against
an engine peak of 703 — a **15×** gap. Saturating this GPU needs ~15 concurrent
preprocessing workers, which is why the queue and batching phases come before TensorRT.

### Precision: FP16 wins, but only once batching makes the GPU compute-bound

`uv run python scripts/benchmark_precision.py --profile-kernels`

| batch | mode | p50 ms | img/s | peak VRAM | vs fp32 |
|---|---|---|---|---|---|
| 8 | fp32 | 15.34 | 537.6 | 106.41 MiB | 1.00× |
| 8 | tf32 | 11.67 | 650.2 | 106.41 MiB | 1.21× |
| 8 | **fp16** | **8.04** | **996.4** | **60.95 MiB** | **1.85×** |
| 32 | **fp16** | **27.65** | **1148.0** | **152.27 MiB** | **1.82×** |

FP16 halves the weights (44.69 → 22.96 MiB) and gives ~1.8× throughput with **100% top-1
agreement** against FP32 on 32 real image crops (max logit drift 3.4e-02). TF32 gives a
free 1.1–1.2× with no storage change at all.

**Batch 1 is not reported as a gain.** Across five independent runs FP16/FP32 at batch 1
measured 1.52×, 1.03×, 1.31×, 1.19×, 1.12× while FP32 stayed within 4% of itself — that
range is jitter. At ~2.6 ms of GPU work the pipeline is launch-bound, not arithmetic-bound,
and halving the arithmetic cannot help.

`--profile-kernels` turns "presumably Tensor Cores" into evidence. FP32 runs
`scudnn_winograd` on CUDA cores and never touches a Tensor Core; FP16 runs
`cutlass_tensorop_f16_s16816`. But FP16 also spends **22.4% of GPU time** on
`nchwToNhwc` + `nhwcToNchw` transposes — converting to the layout Tensor Cores want, per
operator, and back. It wins 1.8× *while* wasting a quarter of its time shuffling memory.
Eliminating that is what TensorRT's whole-graph layout planning does, which gives Phase 8
a concrete target rather than a hope.

### ONNX Runtime is not faster than PyTorch here

`uv run python scripts/benchmark_backends.py`

| batch | pytorch | onnxruntime | ratio |
|---|---|---|---|
| 1 | 275.5 img/s | 303.0 img/s | 1.10× |
| 8 | 572.2 img/s | 568.0 img/s | 0.99× |
| 16 | 657.8 img/s | 674.6 img/s | 1.03× |
| 32 | 644.0 img/s | 649.5 img/s | 1.01× |

Indistinguishable at batch ≥ 8, and the batch-1 edge collapses to 1.06×/1.04×/1.02× when
re-run in isolation. ORT also uses *more* device VRAM (595 vs 523 MiB at batch 32).
Agreement with PyTorch is exact where it counts: top-1 100%, max |Δlogit| 1.43e-05.

That's explainable, not surprising. ORT's advantage is whole-graph optimisation — but the
export had already folded all 20 BatchNorms, so its biggest win was pre-applied. What's
left is 51 convolution-dominated nodes that both runtimes hand to the same cuDNN kernels.
Graph fusion pays on many small ops (transformer blocks), not on a plain convnet.

**The bug this phase nearly shipped with.** `get_available_providers()` listed
`CUDAExecutionProvider`, but the session silently ran on `CPUExecutionProvider` —
`libonnxruntime_providers_cuda.so` couldn't `dlopen` `libcublasLt.so.13`. No exception.
Correct answers, ~10× slower. Every "ONNX Runtime CUDA" number here would have been a CPU
number. `load()` now verifies the provider it got against the one it asked for, and
preloads torch's bundled CUDA libraries so ORT can find them.

## Layout

```
src/
  config.py         typed settings, single source of truth
  inference/        backend implementations behind one interface
  preprocessing/    image -> NCHW float32 tensor (no torch dependency)
  queue/            request queue + dynamic batch manager
  gpu/              memory, streams, profiling
  monitoring/       metrics
  api/              FastAPI routes + schemas
scripts/            operational entry points (check, export, build, benchmark)
benchmarks/results/ committed measurements
docs/               architecture, CUDA, tensors, ONNX, TensorRT, interview prep
```
