"""PyTorch backend — the correctness baseline every other runtime is judged against.

WHAT
    Eager-mode PyTorch executing ResNet-18. No graph compilation, no fusion, no
    kernel autotuning beyond what cuDNN does by default.

WHY have it at all, if TensorRT will be faster
    Because "faster" is meaningless without a reference, and because this is
    the runtime whose numerical output we trust. Phases 7 and 9 ask "does ONNX
    Runtime agree with PyTorch to 1e-4?" and "does TensorRT FP16 still pick the
    same class?" — questions that need a baseline that is simple enough to be
    obviously correct. Optimising before you have one is how you end up with a
    fast pipeline that returns the wrong answer.

THE FOUR THINGS THAT HAPPEN ON EVERY CALL

    1. numpy -> torch      torch.from_numpy is zero-copy; it wraps the same
                           host buffer rather than duplicating it.
    2. host -> device      the batch crosses PCIe into VRAM. This is the only
                           point where "CPU memory" becomes "GPU memory".
    3. forward pass        ~50 CUDA kernels launched onto the stream: conv,
                           batchnorm, relu, pool, gemm. The GPU schedules their
                           blocks across the 20 SMs.
    4. device -> host      1000 logits (4 KB) come back.

TIMING, AND WHY EACH STAGE ENDS WITH A SYNCHRONIZE

    CUDA calls are asynchronous: `model(x)` returns as soon as the kernels are
    *queued*, not when they have run. Timing it with perf_counter alone
    measures launch overhead — scripts/check_gpu.py shows that reporting
    0.065 ms for work that took 5.7 ms.

    So every stage boundary calls torch.cuda.synchronize(), which blocks the
    host until the stream drains. The cost is real but tiny (microseconds on an
    already-idle stream), and it is the difference between a measurement and a
    fiction.

    Caveat worth stating: synchronising between stages also *prevents* overlap
    between them. For a single synchronous predict() there is nothing to
    overlap — it is one stream, executing in issue order — so this costs
    nothing here. Phase 18 (streams) and Phase 19 (pinned memory) are where
    that stops being true, and they are measured separately for exactly this
    reason.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from src.config import Precision, Settings
from src.inference.base import (
    EngineError,
    EngineMetadata,
    EngineNotAvailableError,
    InferenceEngine,
    InferenceResult,
    StageTimings,
)
from src.model.pytorch_model import ModelArtifactError, arch_spec, load_from_repository

_TORCH_DTYPE = {
    Precision.FP32: torch.float32,
    Precision.FP16: torch.float16,
}


class PyTorchEngine(InferenceEngine):
    """Settings are injected, never read from a global.

    That is what lets a benchmark run five engines with five different
    configurations inside one process — which Phase 9 needs, and which a
    module-level `settings` singleton would make impossible.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: torch.nn.Module | None = None
        self._device = torch.device(settings.device)
        self._dtype = _TORCH_DTYPE.get(settings.precision, torch.float32)
        self._num_classes = arch_spec(settings.model_name).num_classes

    # --- lifecycle ------------------------------------------------------

    def load(self) -> None:
        s = self._settings

        if s.is_cuda and not torch.cuda.is_available():
            raise EngineNotAvailableError(
                f"device={s.device} requested but CUDA is unavailable. "
                "Run scripts/check_gpu.py, or set DEVICE=cpu."
            )
        if s.is_cuda and s.device_index >= torch.cuda.device_count():
            raise EngineNotAvailableError(
                f"device={s.device} requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
        if self._dtype is torch.float16 and not s.is_cuda:
            # Not a hardware limit so much as a pointless one: CPU FP16 has no
            # vectorised kernel path for convolutions and runs slower than FP32
            # while also being less accurate. Failing loudly beats a confusing
            # benchmark result.
            raise EngineNotAvailableError(
                "PRECISION=fp16 requires a CUDA device; CPU FP16 convolution is "
                "unaccelerated and slower than FP32"
            )

        # Set before any kernel runs. These are process-global torch flags, not
        # per-model state -- an unavoidable wart of the PyTorch API. Applying
        # them here rather than at import means the setting is always whatever
        # the currently-loaded engine's config says, which is what a benchmark
        # constructing several engines in one process needs.
        if s.is_cuda:
            torch.backends.cudnn.allow_tf32 = s.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = s.allow_tf32

        if s.precision is Precision.INT8:
            raise EngineNotAvailableError(
                "PRECISION=int8 is TensorRT-only in this project. INT8 here is a "
                "calibrated QDQ graph, not a runtime cast, so the PyTorch backend "
                "cannot serve it. Use BACKEND=tensorrt."
            )

        try:
            model = load_from_repository(s.model_dir, s.model_name)
        except ModelArtifactError as exc:
            raise EngineNotAvailableError(str(exc)) from exc

        # eval() switches BatchNorm to its running statistics and disables
        # Dropout. Forgetting it is the classic inference bug: the model still
        # runs, still returns plausible logits, and is quietly wrong because
        # BatchNorm is normalising by the current batch instead of the
        # population statistics it learned.
        model.eval()

        try:
            self._model = model.to(device=self._device, dtype=self._dtype)
        except torch.cuda.OutOfMemoryError as exc:
            raise EngineNotAvailableError(
                f"CUDA OOM while moving weights to {s.device}. "
                f"{arch_spec(s.model_name).num_classes}-class {s.model_name} needs "
                "~45 MiB in FP32; something else is occupying VRAM."
            ) from exc

    def unload(self) -> None:
        """Idempotent: runs on the shutdown path where a signal race can call
        it twice, and a raising cleanup handler aborts the rest of cleanup."""
        if self._model is None:
            return
        self._model = None
        if self._settings.is_cuda:
            # Drops the weights, then returns the caching allocator's reserved
            # blocks to the driver. Without empty_cache() nvidia-smi keeps
            # showing the memory as in use, because torch is still holding it.
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> EngineMetadata:
        s = self._settings
        # Storage stays fp32 under TF32 -- only the convolution accumulation
        # narrows to 10 mantissa bits -- so the two must be reported separately.
        math_mode = s.precision.value
        if s.precision is Precision.FP32 and s.is_cuda and s.allow_tf32:
            math_mode = "tf32"

        return EngineMetadata(
            backend="pytorch",
            model_name=s.model_name,
            model_version=s.model_version,
            precision=s.precision.value,
            math_mode=math_mode,
            device=s.device,
            input_shape=(3, s.image_size, s.image_size),
            max_batch_size=s.max_batch_size,
            num_classes=self._num_classes,
        )

    # --- inference ------------------------------------------------------

    def _sync(self) -> None:
        """Block until the GPU has actually finished. No-op on CPU."""
        if self._settings.is_cuda:
            torch.cuda.synchronize(self._device)

    def _predict(self, batch: np.ndarray) -> InferenceResult:
        assert self._model is not None  # guaranteed by predict()'s is_loaded check

        # inference_mode is stricter than no_grad: it also skips version-counter
        # bookkeeping on every tensor. Nothing here will ever be backpropagated,
        # so both the autograd graph and its metadata are pure overhead.
        with torch.inference_mode():
            # Zero-copy view over the numpy buffer -- no host allocation.
            host_tensor = torch.from_numpy(batch)

            t0 = perf_counter()
            # non_blocking=False because `batch` is pageable memory. Asking for
            # an async copy out of pageable memory silently degrades to a
            # synchronous one anyway; pretending otherwise would make the H2D
            # number a lie. Phase 19 measures what pinned memory changes.
            device_tensor = host_tensor.to(
                device=self._device, dtype=self._dtype, non_blocking=False
            )
            self._sync()
            t1 = perf_counter()

            try:
                logits = self._model(device_tensor)
            except torch.cuda.OutOfMemoryError as exc:
                # Free the reserved blocks so the *next* request has a chance;
                # otherwise one oversized batch poisons the process.
                torch.cuda.empty_cache()
                raise EngineError(
                    f"CUDA out of memory on a batch of {batch.shape[0]}. "
                    "Activations scale linearly with batch size -- reduce "
                    "MAX_BATCH_SIZE."
                ) from exc

            # Cast back on the GPU while we are still in the compute stage, so
            # the interface contract (float32 logits) holds for every precision
            # and cross-backend comparison stays apples-to-apples. No-op in FP32.
            logits = logits.float()
            self._sync()
            t2 = perf_counter()

            host_logits = logits.cpu().numpy()
            self._sync()
            t3 = perf_counter()

        return InferenceResult(
            logits=host_logits,
            timings=StageTimings(
                h2d_ms=(t1 - t0) * 1000.0,
                compute_ms=(t2 - t1) * 1000.0,
                d2h_ms=(t3 - t2) * 1000.0,
            ),
        )
