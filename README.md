# gpu-inference-system

A production-shaped image inference server, built from scratch to understand every layer
between an HTTP request and a CUDA kernel.

The goal is **not** "call `model(x)` and return a label". The goal is to be able to trace
and defend this path end to end:

```
HTTP request -> queue -> decode -> tensor -> queue -> batch -> H2D copy
             -> runtime (PyTorch | ONNX Runtime | TensorRT) -> CUDA -> SMs/Tensor Cores
             -> output tensor -> D2H copy -> queue -> classify -> HTTP response
```

Serving is a three-stage pipeline built on a small, domain-free staged runtime
(`src/pipeline/`) that is meant to be lifted into other projects — see
[docs/pipeline.md](docs/pipeline.md).

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
| 1 | 52.2 | 18.8 ms | 21.1 ms | 5.3 ms |
| **25** | **326.2** | 72.8 ms | 122.2 ms | 11.7 ms |
| 100 | 305.6 | 264.5 ms | 497.7 ms | 25.6 ms |

Peak throughput is **326 req/s against a GPU that sustains 3300 img/s** — about
10% of the accelerator.

The per-stage metrics say exactly why, over a 900-request run:

| stage | total work | share of all work | avg batch |
|---|---|---|---|
| **decode** | **17.40 s** | **90.5%** | 1.0 |
| infer | 1.74 s | 9.0% | 2.3 |
| classify | 0.08 s | 0.4% | 2.3 |

**Decoding JPEGs is 90% of this system.** The GPU is 9%, and it never even fills
a batch — 2.3 images per call against a maximum of 16, because decode cannot
feed it faster. That is the project's actual finding, and it is why the queue and
batching phases were built before TensorRT.

### The one change that did make it faster

`Image.draft()` — a PIL feature already installed, one line — asks libjpeg for a
scaled inverse DCT instead of a full decode. **1.62× end-to-end throughput, and
p99 *fell* from 158.6 to 122.2 ms.** Top-1 agreement 8/8, mean confidence drift
0.0003, and bit-identical output on images it declines to scale.

GPU decode via nvJPEG, which the previous version of this README named as the
obvious next step, was implemented and **measured at 0.14×** — one serial device
against sixteen CPU threads, stealing SMs from the forward pass it was meant to
feed. Rejected with numbers.

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

**GPU decode was the obvious answer and it was wrong.** nvJPEG decodes one
1546×1213 JPEG in 8.08 ms against libjpeg's 13.75 ms — genuinely faster per
image. But there is one GPU and sixteen decode threads, so in aggregate it lost
7×, and it competes for the same SMs as inference. The free option (a scaled DCT
decode, already in PIL) won by 1.73×. Per-image speed is not throughput.

**A benchmark result that was thermal drift.** I reported throughput dropping at
batch 32. Re-run in isolation the two sizes were indistinguishable (662.9 vs
665.2 img/s) — this is a 60 W power-capped card and whatever runs last in a
sweep is measured on a slower GPU. Fixed structurally: clock/power/temperature
on every benchmark row, rotated sweep order, and a 5% noise floor derived from
measured variance.

## Documentation

| | |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | The staged runtime: writing a stage, choosing workers/batch/capacity, backpressure, and how to reuse this as a template |
| [docs/architecture.md](docs/architecture.md) | Eight Mermaid diagrams: system, request and tensor lifecycle, conversion pipeline, GPU execution and memory hierarchy, batching, multi-GPU |
| [docs/request-lifecycle.md](docs/request-lifecycle.md) | One request traced from HTTP through CUDA kernels and back, with measured costs |
| [docs/experiments.md](docs/experiments.md) | All 17 experiments: hypothesis, setup, measurement, result, explanation, trade-off |
| [docs/interview.md](docs/interview.md) | 40 questions answered from measurements |
| [docker/README.md](docker/README.md) | Running with the NVIDIA Container Toolkit |

## Layout

```
src/
  config.py         typed settings, validated at import
  pipeline/         REUSABLE: Stage, StageSpec, Pipeline. Knows nothing about
                    images, models, GPUs or HTTP. Copy this into other projects.
  stages/           THIS APP: decode -> infer -> classify, thin adapters over
                    the modules below. Replace these to repurpose the template.
  inference/        base.py contract + pytorch / onnx / tensorrt engines
  model/            architecture table and artifact loading
  preprocessing/    image -> NCHW float32 (no torch dependency)
  postprocessing/   softmax + top-k, shared by all backends
  gpu/              memory introspection, NVML telemetry
  monitoring/       Prometheus metrics, one instrument per kind, labelled by stage
  api/              FastAPI routes + schemas
  benchmark.py      one measurement harness for every engine
  comparison.py     interleaved, order-rotated configuration sweeps
  main.py           app assembly: build_pipeline() IS the architecture
scripts/            check_gpu, fetch_model, infer, export_onnx, build_engine,
                    quantize_int8, benchmark*, gpu_memory_report, cuda_experiments,
                    stress_test
tests/              224 tests
benchmarks/results/ committed measurements (JSON + CSV)
```

The `src/pipeline/` vs `src/stages/` split is the line to cut along when reusing
this: keep the runtime, replace the stages, edit one list in `main.py`.

## What is deliberately not here

- **Multi-GPU.** One GPU on this machine. The shape is documented, not faked.
- **Accuracy.** Every comparison measures *agreement between runtimes*, never
  accuracy — that needs a labelled eval set this project does not have.
- **CUDA Graphs, streams, pinned memory in the serving path.** All three were
  implemented, measured, and rejected: 0.99×, 1.01×, and irrelevant next to a
  decode stage that is 90% of the work.
- **GPU JPEG decode.** Implemented and measured at 0.14× against sixteen CPU
  threads. Deleted rather than kept behind a flag — a rejected optimisation left
  in the serving path is a maintenance cost with a measured negative return.
- **A DAG.** Stages are a line. Branching needs routing, join semantics and a
  cycle check, and nothing here has a branch.
- **`benchmark()` on the engine interface.** Benchmarking is done *to* an
  engine, not by one; three implementations would drift and could not fairly
  compare backends.
