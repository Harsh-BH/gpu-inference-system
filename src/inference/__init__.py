"""Inference backends.

`create_engine` is added here (rather than in base.py) so that the abstraction
never imports its own implementations, and so backend imports stay lazy —
`import tensorrt` must not run, let alone fail, on a machine serving PyTorch.
"""

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
]
