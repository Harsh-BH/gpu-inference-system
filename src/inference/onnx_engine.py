"""ONNX Runtime backend.

ONNX IS NOT ONNX RUNTIME

    ONNX is the file: a graph description, inert, executing nothing. ONNX
    Runtime is a program that reads that file, decides how to run each
    operator, and runs it. The same model.onnx exported in Phase 6 is consumed
    unchanged here and by TensorRT in Phase 8 — different runtimes, one graph.
    That is the whole reason the interchange format exists.

WHAT ORT ADDS OVER EAGER PYTORCH

    PyTorch executes operator by operator as Python calls them. ORT loads the
    whole graph first, so it can see the future: it fuses adjacent operators,
    folds constants, eliminates dead nodes, and plans memory reuse across the
    entire forward pass before running anything. Then it assigns each node to
    an execution provider and runs a precompiled plan with no Python in the
    loop.

    Whether that wins is measured, not assumed — Phase 7's comparison exists
    for exactly that reason.

EXECUTION PROVIDERS, AND THE FAILURE MODE THEY HIDE

    An execution provider is a backend for operators: CUDA, TensorRT, CPU.
    Providers are supplied in priority order and ORT assigns each node to the
    first one that can run it, falling back down the list.

    That fallback is silent, and it is the single nastiest trap in this
    library. On this machine `ort.get_available_providers()` cheerfully lists
    CUDAExecutionProvider, but creating a session with it produced:

        ACTIVE PROVIDERS: ['CPUExecutionProvider']

    because libcublasLt.so.13 could not be loaded. No exception. The session
    works, returns correct logits, and runs an order of magnitude slower. Anyone
    benchmarking "ONNX Runtime CUDA vs PyTorch CUDA" without checking would
    publish a completely wrong conclusion and never know.

    So load() verifies that the provider it asked for is the provider it got,
    and raises if not. Being slow for a reason you cannot see is worse than
    failing to start.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from time import perf_counter

import numpy as np

from src.config import Settings
from src.inference.base import (
    EngineError,
    EngineMetadata,
    EngineNotAvailableError,
    InferenceEngine,
    InferenceResult,
    StageTimings,
)
from src.model.pytorch_model import onnx_filename

# Loaded before the CUDA provider so its dlopen resolves. See _preload_cuda_libraries.
_CUDA_SONAMES = (
    "libcudart.so.13",
    "libcublas.so.13",
    "libcublasLt.so.13",
    "libcurand.so.10",
    "libcudnn.so.9",
)

_preloaded = False


def _preload_cuda_libraries() -> list[str]:
    """Make torch's bundled CUDA libraries visible to ONNX Runtime.

    PyTorch ships its CUDA runtime inside the nvidia-* pip packages and loads
    it itself at import. ONNX Runtime expects to find the same libraries
    through the dynamic linker's normal search path, which does not include
    site-packages — so `libonnxruntime_providers_cuda.so` fails to dlopen with
    'libcublasLt.so.13: cannot open shared object file' and the CUDA provider
    is silently dropped.

    Loading each library with RTLD_GLOBAL first puts it in the process's global
    symbol table, and the linker then satisfies ORT's dependency from memory
    instead of searching the filesystem. Equivalent to setting LD_LIBRARY_PATH,
    but it works without asking anyone to configure their shell.

    Best-effort by design: if this does not fix it, the provider check in
    load() reports the real problem rather than this function guessing at it.
    """
    global _preloaded
    if _preloaded:
        return []
    loaded = []
    try:
        import nvidia
    except ImportError:
        _preloaded = True
        return []

    # `nvidia` is a namespace package -- several distributions (nvidia-cu13,
    # nvidia-cudnn, ...) share it, so it has no __init__.py and __file__ is
    # None. __path__ is the list of directories that actually contribute to it.
    roots = [Path(p) for p in getattr(nvidia, "__path__", [])]

    for soname in _CUDA_SONAMES:
        for root in roots:
            candidates = sorted(root.glob(f"*/lib/{soname}"))
            if not candidates:
                continue
            try:
                ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)
                loaded.append(soname)
            except OSError:
                pass
            break
    _preloaded = True
    return loaded


class ONNXRuntimeEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = None
        self._input_name = ""
        self._output_name = ""
        self._num_classes = 0
        self._input_dtype = np.float32
        self._preloaded_libs: list[str] = []

    # --- lifecycle ------------------------------------------------------

    def load(self) -> None:
        s = self._settings
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EngineNotAvailableError(
                "onnxruntime is not installed. Run: uv sync --extra onnx"
            ) from exc

        # Precision is a property of the graph, not a runtime switch: ORT never
        # converts at load time. FP16 means loading a different file.
        path = s.model_dir / onnx_filename(s.precision.value)
        if not path.is_file():
            raise EngineNotAvailableError(
                f"no ONNX model at {path}\n"
                f"  run: uv run python scripts/export_onnx.py "
                f"--precision {s.precision.value}"
            )

        providers: list = []
        if s.is_cuda:
            self._preloaded_libs = _preload_cuda_libraries()
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": s.device_index,
                        # ORT enables TF32 for convolutions by default, exactly
                        # as PyTorch does. Left alone it would reintroduce the
                        # Experiment 2 problem on a different backend, and a
                        # PyTorch-vs-ORT accuracy comparison would be measuring
                        # this flag rather than the runtimes.
                        "use_tf32": "1" if s.allow_tf32 else "0",
                    },
                )
            )
        # CPU is always last so a node CUDA cannot run still executes. That
        # fallback is desirable per-node and disastrous per-session, which is
        # why the check below looks at what was actually registered.
        providers.append("CPUExecutionProvider")

        try:
            session = ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            raise EngineNotAvailableError(
                f"could not create an ONNX Runtime session for {path}: {exc}"
            ) from exc

        active = session.get_providers()
        if s.is_cuda and "CUDAExecutionProvider" not in active:
            raise EngineNotAvailableError(
                f"requested CUDA but ONNX Runtime fell back to {active}.\n"
                "  This is silent by default: the session would have worked and run\n"
                "  roughly an order of magnitude slower.\n"
                f"  Preloaded CUDA libraries: {self._preloaded_libs or 'none found'}\n"
                "  Check that onnxruntime-gpu matches your CUDA and cuDNN major versions."
            )

        self._session = session
        self._read_io_metadata(path)

    def _read_io_metadata(self, path: Path) -> None:
        """Take shapes from the graph, not from config.

        The exported model is the authority on what it accepts. Trusting
        IMAGE_SIZE instead would let a config change silently disagree with the
        artifact and fail later, deeper, with a worse message.
        """
        assert self._session is not None
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise EngineNotAvailableError(
                f"expected 1 input and 1 output, got {len(inputs)} and {len(outputs)}"
            )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._input_dtype = np.float16 if "float16" in inputs[0].type else np.float32

        shape = inputs[0].shape  # e.g. ['batch', 3, 224, 224]
        if len(shape) != 4:
            raise EngineNotAvailableError(f"expected a 4-D input, got {shape} in {path}")
        if isinstance(shape[0], int):
            raise EngineNotAvailableError(
                f"{path} has a fixed batch dimension of {shape[0]}; dynamic batching "
                "needs a symbolic one. Re-export with scripts/export_onnx.py."
            )

        expected = (3, self._settings.image_size, self._settings.image_size)
        actual = tuple(shape[1:])
        if actual != expected:
            raise EngineNotAvailableError(
                f"{path} expects input {actual} but IMAGE_SIZE implies {expected}"
            )
        out_shape = outputs[0].shape
        self._num_classes = int(out_shape[1]) if isinstance(out_shape[1], int) else 0

    def unload(self) -> None:
        # ORT frees its CUDA allocations when the session is destroyed. Note
        # that this memory never appears in torch.cuda.memory_allocated() --
        # ORT has its own allocator, which is why cross-backend VRAM has to be
        # measured at the driver level.
        self._session = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def metadata(self) -> EngineMetadata:
        s = self._settings
        math_mode = "tf32" if (s.is_cuda and s.allow_tf32) else s.precision.value
        return EngineMetadata(
            backend="onnxruntime",
            model_name=s.model_name,
            model_version=s.model_version,
            precision=s.precision.value,
            math_mode=math_mode,
            device=s.device,
            input_shape=(3, s.image_size, s.image_size),
            max_batch_size=s.max_batch_size,
            num_classes=self._num_classes or 1000,
        )

    # --- inference ------------------------------------------------------

    def _predict(self, batch: np.ndarray) -> InferenceResult:
        import onnxruntime as ort

        assert self._session is not None
        s = self._settings

        if self._input_dtype != batch.dtype:
            # Cast to whatever the graph declares. The engine interface is
            # always float32 so backends stay interchangeable and comparable;
            # the conversion belongs here, not in the caller.
            batch = batch.astype(self._input_dtype)

        if not s.is_cuda:
            # No device, no transfers to attribute. session.run does everything.
            t0 = perf_counter()
            logits = self._session.run([self._output_name], {self._input_name: batch})[0]
            elapsed = (perf_counter() - t0) * 1000.0
            return InferenceResult(
                logits=np.ascontiguousarray(logits, dtype=np.float32),
                timings=StageTimings(h2d_ms=0.0, compute_ms=elapsed, d2h_ms=0.0),
            )

        # IOBinding rather than session.run() so the three stages are separable.
        # session.run() takes numpy in and gives numpy back, doing both copies
        # internally as one opaque call -- which would leave h2d and d2h as
        # zeros and make the stage breakdown incomparable with the PyTorch
        # backend. Binding also happens to be what a real ORT deployment uses.
        device_id = s.device_index or 0
        binding = self._session.io_binding()

        t0 = perf_counter()
        device_input = ort.OrtValue.ortvalue_from_numpy(batch, "cuda", device_id)
        binding.bind_ortvalue_input(self._input_name, device_input)
        binding.bind_output(self._output_name, "cuda", device_id)
        binding.synchronize_inputs()
        t1 = perf_counter()

        try:
            self._session.run_with_iobinding(binding)
            binding.synchronize_outputs()
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                raise EngineError(
                    f"ONNX Runtime ran out of GPU memory on a batch of {batch.shape[0]}. "
                    "Activations scale with batch size -- reduce MAX_BATCH_SIZE."
                ) from exc
            raise EngineError(f"ONNX Runtime inference failed: {message}") from exc
        t2 = perf_counter()

        logits = binding.copy_outputs_to_cpu()[0]
        t3 = perf_counter()

        return InferenceResult(
            logits=np.ascontiguousarray(logits, dtype=np.float32),
            timings=StageTimings(
                h2d_ms=(t1 - t0) * 1000.0,
                compute_ms=(t2 - t1) * 1000.0,
                d2h_ms=(t3 - t2) * 1000.0,
            ),
        )
