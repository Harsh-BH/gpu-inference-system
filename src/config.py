"""Typed configuration — the single source of truth for every tunable knob.

WHY this exists as its own module:

A serving system has two kinds of "settings". The first kind is a value you
change to run an experiment (batch size, precision, backend). The second kind
is a value that, if wrong, should stop the process at boot rather than surface
as a confusing failure ten minutes into a load test (a device string of
"cuda:9", a batch size of 0).

Scattering `os.environ.get(...)` through the codebase gets you neither. You get
untyped strings validated nowhere, defaults duplicated in three files, and no
single place to answer "what is this server actually configured to do?".

So: one model, validated once, at import. Invalid configuration is a startup
crash with a readable message. That is a feature.
"""

from __future__ import annotations

import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEVICE_RE = re.compile(r"^(cpu|cuda(:\d+)?)$")


class Backend(StrEnum):
    """Which runtime executes the model.

    This is an enum rather than a string so an unknown backend fails at config
    parse time, next to the typo, instead of as a KeyError inside the engine
    registry after the server has already reported itself healthy.
    """

    PYTORCH = "pytorch"
    ONNXRUNTIME = "onnxruntime"
    TENSORRT = "tensorrt"


class Precision(StrEnum):
    """Numeric precision of the weights and activations at inference time.

    Deliberately explicit. A model is never silently downcast: FP16 changes
    numerical results, and a system that quietly halves your precision to look
    good on a benchmark is lying to you. You ask for it or you don't get it.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    # INT8 is TensorRT-only here. It is not a runtime flag but a separate
    # calibrated QDQ graph (scripts/quantize_int8.py); PyTorch and ONNX Runtime
    # reject it explicitly rather than quietly serving fp32.
    INT8 = "int8"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # pydantic reserves the `model_` prefix for its own methods; our domain
        # genuinely is models, so we reclaim the namespace.
        protected_namespaces=(),
    )

    # --- Model identity -------------------------------------------------
    # Nothing in src/ may build a model path by hand. Everything resolves
    # through `model_dir` so that swapping versions is a config change, not
    # a code change (Phase 14).
    model_repository: Path = Path("models")
    model_name: str = "resnet18"
    model_version: str = "v1"

    # --- Execution ------------------------------------------------------
    backend: Backend = Backend.PYTORCH
    precision: Precision = Precision.FP32
    device: str = "cuda:0"

    # TF32: the silent precision change nobody asks for.
    #
    # On Ampere and later, PyTorch defaults torch.backends.cudnn.allow_tf32 to
    # True. Convolutions then run on Tensor Cores in TF32 -- 8 exponent bits
    # like FP32, but only 10 mantissa bits instead of 23. Tensors are still
    # float32 in memory; the *accumulation* is not. Nothing announces this.
    #
    # Measured on this box (RTX 3050, ResNet-18, max |gpu - cpu| on logits):
    #     TF32 on  ->  3.56e-03      (PyTorch's default)
    #     TF32 off ->  5.25e-06      (true FP32)
    # A 680x difference in numerical agreement, from a flag we never set.
    #
    # That is unusable as a baseline. Phases 5, 7 and 9 all measure runtimes
    # against "PyTorch FP32", and if that baseline is quietly TF32 then every
    # accuracy delta attributed to ONNX Runtime or TensorRT is partly just this.
    #
    # So it defaults to False -- honest FP32 -- and is exposed as a knob rather
    # than a default, because "do not silently convert the model" has to apply
    # to the framework's silent conversions too, not only to ours. Turning it
    # on is a legitimate production choice and Phase 5 benchmarks it as one.
    allow_tf32: bool = False

    # --- Preprocessing --------------------------------------------------
    image_size: int = Field(default=224, ge=32, le=1024)

    # Scaled JPEG decode. libjpeg can emit 1/2, 1/4 or 1/8 size straight from
    # the DCT coefficients, skipping most of the inverse transform, and PIL
    # exposes it as Image.draft(). It applies only when the result would still
    # be at least the resize target, so it is a no-op on images that are
    # already small -- and JPEG-only.
    #
    # Measured here (Experiment 18): 2.47x on a 1546x1213 photo, 1.73x across a
    # mixed set at 16 workers, top-1 agreement 8/8, mean confidence drift
    # 0.0003. Bit-identical output on the samples it declines to scale.
    #
    # Defaults ON, unlike ALLOW_TF32, and the difference is worth stating. TF32
    # silently degrades the FP32 *baseline* every cross-runtime comparison in
    # this repo is measured against, so it must be opted into. This changes the
    # input identically for every backend, so it cannot bias a comparison
    # between them -- and it is the single largest throughput win the serving
    # path has. Turn it off for a bit-exact reference decode.
    fast_decode: bool = True

    # --- Batching -------------------------------------------------------
    # These two numbers are the entire latency/throughput dial of the system.
    # Bigger batch: better GPU utilisation, more VRAM, higher tail latency.
    # Longer wait: fuller batches under light load, but every request pays it.
    # Upper bound is a sanity check, not a policy: it exists to reject a typo'd
    # 100000, while still leaving room for scripts/gpu_memory_report.py to probe
    # past any sane serving value on purpose. The serving default stays 8.
    max_batch_size: int = Field(default=8, ge=1, le=1024)
    max_batch_wait_ms: float = Field(default=5.0, ge=0.0, le=1000.0)

    # --- Queue / backpressure -------------------------------------------
    # Bounded, always. An unbounded queue does not absorb overload, it defers
    # it — turning a recoverable 503 into an OOM kill and dropping every
    # in-flight request instead of the few that arrived last.
    queue_max_size: int = Field(default=100, ge=1)
    request_timeout_ms: float = Field(default=10_000.0, gt=0)

    # Preprocessing thread pool. Phase 4 measured one thread sustaining
    # ~46 img/s against an engine absorbing 700-3200, so a single worker would
    # starve the GPU no matter how fast the engine is. PIL releases the GIL
    # during decode and resize, which is why threads help here at all.
    preprocess_workers: int = Field(default=8, ge=1, le=64)

    # --- Warmup ---------------------------------------------------------
    # The first inference on a fresh CUDA context pays context creation,
    # kernel module loading, cuDNN algorithm selection and allocator growth.
    # That can be hundreds of milliseconds. We pay it at boot; a user must
    # never be the one who pays it.
    warmup_requests: int = Field(default=10, ge=0)

    # --- Limits / safety ------------------------------------------------
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    # Guards against decompression bombs: a 10 KB PNG can declare a
    # 60000x60000 canvas and allocate ~10 GB when decoded.
    max_image_pixels: int = Field(default=50_000_000, gt=0)

    # --- Observability --------------------------------------------------
    log_level: str = "INFO"

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        v = v.strip().lower()
        if not _DEVICE_RE.match(v):
            raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {v!r}")
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = v.strip().upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}, got {v!r}")
        return v

    @computed_field
    @property
    def model_dir(self) -> Path:
        """Where this model version's artifacts live.

        models/<name>/<version>/ — holds weights, model.onnx, and per-precision
        TensorRT engines. Resolving it here is what keeps every other module
        free of hardcoded paths.
        """
        return self.model_repository / self.model_name / self.model_version

    @computed_field
    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    @computed_field
    @property
    def device_index(self) -> int | None:
        """GPU ordinal, or None on CPU. `cuda` with no index means device 0."""
        if not self.is_cuda:
            return None
        _, _, idx = self.device.partition(":")
        return int(idx) if idx else 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, parsed once.

    Cached rather than a module-level global so tests can swap the environment
    and call `get_settings.cache_clear()`, and so importing this module has no
    side effect on a machine with a broken .env.
    """
    return Settings()
