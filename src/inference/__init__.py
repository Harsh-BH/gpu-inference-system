"""Inference backends and the config-driven selector.

`create_engine` lives here rather than in base.py so the abstraction never
imports its own implementations, and so backend imports stay *lazy*. That
laziness is load-bearing: `import tensorrt` must not run — let alone fail — on
a machine that is serving PyTorch and never installed the extra.

A match statement rather than a decorator-based registry, deliberately. A
registry only populates itself once every backend module has been imported,
which is precisely the eager import we are avoiding. Three branches beat a
plugin system that defeats its own purpose.
"""

from src.config import Backend, Settings
from src.inference.base import (
    EngineError,
    EngineMetadata,
    EngineNotAvailableError,
    EngineNotLoadedError,
    InferenceEngine,
    InferenceResult,
    StageTimings,
)

__all__ = [
    "EngineError",
    "EngineMetadata",
    "EngineNotAvailableError",
    "EngineNotLoadedError",
    "InferenceEngine",
    "InferenceResult",
    "StageTimings",
    "create_engine",
]


def create_engine(settings: Settings) -> InferenceEngine:
    """Build the backend named by settings.backend. Does not load() it.

    Construction and loading stay separate so a caller can inspect or register
    an engine before paying for weights and GPU context.
    """
    match settings.backend:
        case Backend.PYTORCH:
            from src.inference.pytorch_engine import PyTorchEngine

            return PyTorchEngine(settings)

        case Backend.ONNXRUNTIME:
            from src.inference.onnx_engine import ONNXRuntimeEngine

            return ONNXRuntimeEngine(settings)

        case Backend.TENSORRT:
            raise EngineNotAvailableError(
                "the tensorrt backend is not implemented yet (Phase 8). Use BACKEND=pytorch."
            )

        case _:  # unreachable: Settings validates the enum at parse time
            raise EngineNotAvailableError(f"unknown backend {settings.backend!r}")
