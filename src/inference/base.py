"""The contract every inference backend implements.

WHAT
    One abstract class, three future implementations (PyTorch, ONNX Runtime,
    TensorRT), and the value types they exchange with the rest of the system.

WHY it is an abstraction and not just "the PyTorch code"
    The serving layer must not know which runtime is executing the model.
    That is not architectural decoration — it is what makes the central
    experiment of this project possible: swapping TensorRT for PyTorch must be
    a config change, so that a benchmark comparing them is comparing *runtimes*
    and not two differently-written pipelines.

THE TWO DECISIONS THAT MATTER HERE

1. The interface speaks numpy, not torch.

   ONNX Runtime and TensorRT both accept numpy natively. If this contract were
   written in torch tensors, the TensorRT engine would need PyTorch installed
   purely to satisfy a type signature — the abstraction would depend on one of
   its own implementations. Instead the contract depends only on numpy, which
   all three need regardless. Dependency inversion, with a concrete payoff:
   `uv sync --extra trt` does not have to drag in torch.

2. predict() returns logits, not labels.

   Softmax, top-k and label lookup are identical for every backend. Putting
   them behind the interface would duplicate that code three times and let it
   drift. Worse, it would make cross-backend numerical comparison impossible —
   and "does TensorRT FP16 still agree with PyTorch FP32?" is a question this
   project has to be able to answer with numbers, not vibes.

WHAT IS DELIBERATELY *NOT* HERE
    benchmark(). The PRD lists it on this interface; it does not belong here.
    Benchmarking is something you do *to* an engine, not something an engine
    does. Three implementations would mean three timing loops that inevitably
    drift, and a benchmark implemented differently per backend cannot fairly
    compare backends. It lives in one place that accepts any InferenceEngine,
    which also means new backends get benchmarked for free (open/closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import numpy as np


class EngineError(RuntimeError):
    """Base for every failure originating in an inference backend.

    Exists so the API layer can catch one type and map it to a clean HTTP
    response, instead of leaking backend-specific exceptions (and stack traces
    naming internal paths) to clients.
    """


class EngineNotAvailableError(EngineError):
    """The backend cannot be used on this machine right now.

    Raised for missing optional dependencies (`import tensorrt` fails), missing
    artifacts (no model.onnx for this version), or an unusable device. Separate
    from a generic error because this one is actionable: the message tells you
    what to install or build.
    """


class EngineNotLoadedError(EngineError):
    """predict() was called before load(). A programming error, not user input."""


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Where the time inside an engine actually went, in milliseconds.

    These three stages are only observable from inside the engine — a caller
    holding a numpy array cannot see how long the host-to-device copy took. So
    the engine reports them. This is what turns "inference is slow" into
    "we are spending 60% of the request copying 2 MB across PCIe", which is a
    fixable statement.

    Every one of these numbers is taken with an explicit CUDA synchronize.
    Without that they measure kernel *launch* time and are meaningless.
    """

    h2d_ms: float  # host -> device: the input batch crossing PCIe
    compute_ms: float  # the forward pass itself
    d2h_ms: float  # device -> host: logits coming back

    @property
    def total_ms(self) -> float:
        return self.h2d_ms + self.compute_ms + self.d2h_ms


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Raw model output plus the cost of producing it."""

    logits: np.ndarray  # (N, num_classes) float32
    timings: StageTimings


@dataclass(frozen=True, slots=True)
class EngineMetadata:
    """What this engine is, for /ready, API responses and benchmark labelling.

    A benchmark row that says "12.4 ms" is worthless. One that says
    "tensorrt / fp16 / cuda:0 / batch 8 -> 12.4 ms" is a result.
    """

    backend: str
    model_name: str
    model_version: str
    precision: str
    device: str
    input_shape: tuple[int, int, int]  # (C, H, W) for a single sample
    max_batch_size: int
    num_classes: int


class InferenceEngine(ABC):
    """Lifecycle: load() -> warmup() -> predict()* -> unload().

    Loading is an explicit method rather than constructor work on purpose.
    Building a TensorRT engine or moving weights to the GPU is slow and can
    fail with CUDA OOM; a constructor that does either gives you an object that
    might not exist and an exception with nothing to clean up. Explicit load()
    also gives the server an honest "loading" state to report on /ready, which
    is the difference between a health check and a lie.
    """

    # --- lifecycle ------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Acquire everything needed to serve: weights, GPU context, buffers.

        Raises EngineNotAvailableError if the backend or its artifacts are
        missing on this machine.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release GPU memory and handles. Must be safe to call twice.

        Idempotent because it runs on the shutdown path, where a second call
        during a signal-handler race must not raise and abort cleanup.
        """

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def metadata(self) -> EngineMetadata: ...

    # --- inference ------------------------------------------------------

    def predict(self, batch: np.ndarray) -> InferenceResult:
        """Run one batch. (N, C, H, W) float32 in, (N, num_classes) float32 out.

        Concrete, not abstract: validation happens here so that no backend can
        skip it. Subclasses implement `_predict` and inherit the guarantee that
        anything reaching them has already been checked for shape and dtype.
        Template Method — the invariant is enforced once, in the place that
        cannot be forgotten.
        """
        if not self.is_loaded:
            raise EngineNotLoadedError(f"{type(self).__name__}.predict() called before load()")
        self._validate_batch(batch)
        return self._predict(batch)

    @abstractmethod
    def _predict(self, batch: np.ndarray) -> InferenceResult:
        """Backend-specific forward pass. `batch` is already validated."""

    def warmup(self, iterations: int) -> None:
        """Execute throwaway inferences so the first real request is not the
        one that pays initialisation.

        A cold engine's first forward pass pays lazy CUDA context creation,
        kernel module loading, cuDNN algorithm selection and allocator growth —
        hundreds of milliseconds, charged entirely to whoever arrived first.
        Phase 13 measures exactly how much.

        Default implementation works for every backend: feed zeros of the right
        shape. Overriding is allowed but rarely necessary.
        """
        if iterations <= 0:
            return
        c, h, w = self.metadata.input_shape
        dummy = np.zeros((1, c, h, w), dtype=np.float32)
        for _ in range(iterations):
            self.predict(dummy)

    # --- helpers --------------------------------------------------------

    def _validate_batch(self, batch: np.ndarray) -> None:
        """Reject malformed input at the boundary.

        These are PRD failure cases 19 and 20 (wrong dtype, wrong shape). They
        matter more than they look: a silently wrong dtype does not crash
        TensorRT, it reinterprets the bytes and returns confident garbage.
        """
        meta = self.metadata
        if not isinstance(batch, np.ndarray):
            raise EngineError(f"batch must be np.ndarray, got {type(batch).__name__}")
        if batch.ndim != 4:
            raise EngineError(f"batch must be 4-D (N, C, H, W), got {batch.ndim}-D {batch.shape}")
        expected = meta.input_shape
        if batch.shape[1:] != expected:
            raise EngineError(
                f"expected per-sample shape (C, H, W)={expected}, got {batch.shape[1:]}"
            )
        if batch.shape[0] < 1:
            raise EngineError("batch must contain at least one sample")
        if batch.shape[0] > meta.max_batch_size:
            raise EngineError(
                f"batch of {batch.shape[0]} exceeds max_batch_size={meta.max_batch_size}"
            )
        if batch.dtype != np.float32:
            raise EngineError(f"batch must be float32, got {batch.dtype}")

    # --- context manager ------------------------------------------------
    # Scripts leak GPU memory when they exit between load() and unload().
    # Four lines here removes that whole class of mistake.

    def __enter__(self) -> Self:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()
