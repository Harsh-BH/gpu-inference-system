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
uv sync --extra onnx --extra trt            # + ONNX Runtime and TensorRT

uv run python scripts/check_gpu.py          # verify the stack (non-zero exit if not)
uv run python scripts/fetch_model.py        # provision models/resnet18/v1/

mkdir -p data && curl -sSL -o data/dog.jpg \
  https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg

uv run python scripts/infer.py data/dog.jpg --trace --runs 50
```

Optimised backends (build once, serve many times):

```bash
uv run python scripts/export_onnx.py                      # model.onnx
uv run python scripts/export_onnx.py --precision fp16     # model.fp16.onnx
uv run python scripts/build_engine.py --precision fp16    # TensorRT engine
uv run python scripts/quantize_int8.py                    # calibrated INT8 QDQ
uv run python scripts/build_engine.py --precision int8
```

Serve:

```bash
BACKEND=tensorrt PRECISION=fp16 MAX_BATCH_SIZE=16 PREPROCESS_WORKERS=16 \
  uv run uvicorn src.main:app --port 8000

curl -F file=@data/dog.jpg http://localhost:8000/predict
curl http://localhost:8000/ready
curl http://localhost:8000/metrics
```

Configuration is entirely environment-driven — see `.env.example`. No model
path, device, batch size or precision is hardcoded anywhere in `src/`.

## Results

Full write-ups, including the ones where the hypothesis was wrong, are in
[docs/experiments.md](docs/experiments.md).

### Backends — same weights, six ways (batch 8)

| configuration | img/s | vs baseline | top-1 agreement |
|---|---|---|---|
| pytorch/fp32 | 560.1 | 1.00× | — |
| pytorch/fp16 | 1011.3 | 1.81× | 100% |
| onnxruntime/fp32 | 552.2 | 0.99× | 100% |
| onnxruntime/fp16 | 817.0 | 1.46× | 100% |
| tensorrt/fp32 | 857.2 | 1.53× | 100% |
| **tensorrt/fp16** | **2911.2** | **5.20×** | 100% |
| tensorrt/int8 | 3153.1 | 5.57× | 90.6% |

TensorRT FP16 computes **3.3× faster than PyTorch FP16 on identical arithmetic**
(6.98 vs 23.14 ms at batch 32). The gap is layout: PyTorch converts NCHW→NHWC
per operator and back, burning **22.4% of GPU time** on transposes that compute
nothing. TensorRT plans layout once for the whole graph.

INT8 works and is not worth it here — 18% over FP16 while top-1 agreement drops
to 90.6%.

### And none of it made the system much faster

| clients | req/s | p50 | p99 | server queue wait |
|---|---|---|---|---|
| 1 | 37.3 | 26.2 ms | 34.3 ms | 7.2 ms |
| **25** | **223.6** | 107.4 ms | 158.0 ms | 21.0 ms |
| 100 | 189.6 | 404.4 ms | 682.8 ms | 14.8 ms |

Peak throughput is **223 req/s against a GPU that sustains 3300 img/s** — about
7% of the accelerator. Queue wait stays low even at 100 clients, so the backlog
is not in inference. Raising the preprocessing pool 4→16 workers gave +33%
throughput with no GPU change at all.

**The GPU accounts for ~7 ms of a 107 ms request.** That is the project's actual
finding, and it is why the queue and batching phases were built before TensorRT.

### Three traps this project walked into

**`PRECISION=fp32` was not FP32.** PyTorch enables TF32 convolutions by default
on Ampere — 10 mantissa bits instead of 23 — while tensors still report
`float32`. Divergence from a CPU reference: **3.56e-03 vs 5.25e-06**, 680×
worse, silently, with top-1 unaffected. Now an explicit `ALLOW_TF32` flag,
default off, pinned by a test.

**ONNX Runtime silently ran on the CPU.** `get_available_providers()` listed
CUDA; the session used `CPUExecutionProvider` because `libcublasLt.so.13` could
not load. No exception. Correct answers, ~10× slower. Every "ORT CUDA" number
would have been a CPU number. `load()` now verifies the provider it *got*.

**A benchmark result that was thermal drift.** I reported throughput dropping at
batch 32. Re-run in isolation the two sizes were indistinguishable (662.9 vs
665.2 img/s) — this is a 60 W power-capped card and whatever runs last in a
sweep is measured on a slower GPU. Fixed structurally: clock/power/temperature
on every benchmark row, rotated sweep order, and a 5% noise floor derived from
measured variance.

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Eight Mermaid diagrams: system, request and tensor lifecycle, conversion pipeline, GPU execution and memory hierarchy, batching, multi-GPU |
| [docs/request-lifecycle.md](docs/request-lifecycle.md) | One request traced from HTTP through CUDA kernels and back, with measured costs |
| [docs/experiments.md](docs/experiments.md) | All 17 experiments: hypothesis, setup, measurement, result, explanation, trade-off |
| [docs/interview.md](docs/interview.md) | 40 questions answered from measurements |
| [docker/README.md](docker/README.md) | Running with the NVIDIA Container Toolkit |

## Layout

```
src/
  config.py         typed settings, validated at import
  inference/        base.py contract + pytorch / onnx / tensorrt engines
  model/            architecture table and artifact loading
  preprocessing/    image -> NCHW float32 (no torch dependency)
  postprocessing/   softmax + top-k, shared by all backends
  queue/            bounded request queue + dynamic batch manager
  gpu/              memory introspection, NVML telemetry
  monitoring/       Prometheus metrics
  api/              FastAPI routes + schemas
  benchmark.py      one measurement harness for every engine
  comparison.py     interleaved, order-rotated configuration sweeps
  main.py           app assembly and lifecycle
scripts/            check_gpu, fetch_model, infer, export_onnx, build_engine,
                    quantize_int8, benchmark*, gpu_memory_report, cuda_experiments,
                    stress_test
tests/              186 tests
benchmarks/results/ committed measurements (JSON + CSV)
```

## What is deliberately not here

- **Multi-GPU.** One GPU on this machine. The shape is documented, not faked.
- **Accuracy.** Every comparison measures *agreement between runtimes*, never
  accuracy — that needs a labelled eval set this project does not have.
- **CUDA Graphs, streams, pinned memory in the serving path.** All three were
  implemented, measured, and rejected: 0.99×, 1.01×, and irrelevant next to a
  21.9 ms preprocessing stage.
- **`benchmark()` on the engine interface.** Benchmarking is done *to* an
  engine, not by one; three implementations would drift and could not fairly
  compare backends.
