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
