# Running in Docker

```bash
# build
docker build -f docker/Dockerfile -t gpu-inference .

# provision artifacts on the host first (they are mounted, not baked in)
uv run python scripts/fetch_model.py
uv run python scripts/export_onnx.py

# run
docker run --gpus all -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  -e BACKEND=pytorch -e PRECISION=fp32 \
  gpu-inference

curl http://localhost:8000/ready
curl -F file=@data/dog.jpg http://localhost:8000/predict
```

## Requirements

`--gpus all` needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
on the host. Verify with:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

## Why models are mounted rather than copied in

A TensorRT engine is compiled for a specific GPU architecture, TensorRT version
and driver. Baking one into an image produces something that runs on the build
machine and fails on every other. Weights and ONNX graphs are portable but
large, and rebuilding a 3 GB image to change a model version is the wrong
workflow.

Build engines on the target host:

```bash
docker run --gpus all -v "$(pwd)/models:/app/models" gpu-inference \
  python scripts/build_engine.py --precision fp16
```

## Configuration

Every setting in `.env.example` is an environment variable:

```bash
docker run --gpus all -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  -e BACKEND=tensorrt -e PRECISION=fp16 \
  -e MAX_BATCH_SIZE=16 -e MAX_BATCH_WAIT_MS=5 \
  -e PREPROCESS_WORKERS=16 -e QUEUE_MAX_SIZE=100 \
  gpu-inference
```

## Scaling

One process per GPU. A single CUDA context serialises work anyway, so extra
uvicorn workers in one container buy nothing and complicate the memory ceiling.
Run one container per GPU with `--gpus '"device=N"'` and load-balance across
them.
