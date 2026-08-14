# GPU inference server.
#
#   docker build -f docker/Dockerfile -t gpu-inference .
#   docker run --gpus all -p 8000:8000 -v $(pwd)/models:/app/models gpu-inference
#
# --gpus all requires the NVIDIA Container Toolkit on the host. Without it the
# container starts, fails to find a GPU, and /ready returns 503 with the reason
# rather than crash-looping -- which is the behaviour you want while debugging
# exactly this.
#
# Model artifacts are MOUNTED, not baked in. Weights are 45 MB, TensorRT
# engines are another 50 MB each, and an engine is tied to the GPU
# architecture, TensorRT version and driver it was built with. Baking one in
# produces an image that silently fails on any other machine.

# Two stages so build tooling does not ship. The runtime image gets the venv
# and nothing that built it.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
# Dependency files first: this layer is cached until they change, so editing
# source does not re-download three gigabytes of CUDA wheels.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra onnx --extra trt

COPY src/ ./src/


FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 serve

WORKDIR /app
COPY --from=builder /opt/python /opt/python
COPY --from=builder --chown=serve:serve /app/.venv /app/.venv
COPY --from=builder --chown=serve:serve /app/src /app/src
COPY --chown=serve:serve scripts/ /app/scripts/

# Non-root. The process needs the GPU device nodes, which the container
# toolkit exposes to any user; it does not need to be able to write to /app.
USER serve

ENV MODEL_REPOSITORY=/app/models \
    DEVICE=cuda:0 \
    BACKEND=pytorch \
    PRECISION=fp32

EXPOSE 8000

# Liveness only. /ready depends on the model and would kill the container
# during a slow load, producing a restart loop that looks like a crash.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

# One worker. The batch manager and the engine are per-process state, and a
# second worker in the same container would mean two CUDA contexts competing
# for one GPU. Scale with more containers, one per GPU.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
