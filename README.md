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

## Status

Built incrementally. Each phase leaves the system working.

| Phase | Component | Status |
|---|---|---|
| 0 | Environment + GPU verification | in progress |
| 1 | Baseline PyTorch inference path | pending |
| 2 | Tensor introspection | pending |
| 3 | GPU memory profiling + OOM experiment | pending |
| 4 | Batch size benchmark | pending |
| 5 | FP32 vs FP16 | pending |
| 6 | ONNX export | pending |
| 7 | ONNX Runtime backend | pending |
| 8 | TensorRT engine build | pending |
| 9 | Cross-backend benchmark | pending |
| 10 | Request queue | pending |
| 11 | Dynamic batching | pending |
| 12 | Concurrency testing | pending |
| 13 | Model warmup | pending |
| 14 | Model versioning | pending |
| 15 | HTTP API | pending |
| 16 | Observability | pending |
| 17 | Failure handling | pending |
| 18 | CUDA streams experiment | pending |
| 19 | Transfer optimization | pending |
| 20 | CUDA Graphs experiment | pending |
| 21 | INT8 quantization | pending |
| 22 | Final architecture + report | pending |

## Quickstart

```bash
uv sync                    # core + PyTorch backend
uv sync --extra onnx       # + ONNX Runtime backend
uv sync --extra trt        # + TensorRT backend

uv run python scripts/check_gpu.py
```

Configuration is entirely environment-driven — see `.env.example`. No model path, device,
batch size, or precision is hardcoded anywhere in `src/`.

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
