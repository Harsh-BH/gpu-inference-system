"""TensorRT backend: executing a prebuilt engine.

THE DIVISION OF LABOUR

    scripts/build_engine.py does the compiling — minutes of tactic timing,
    kernel selection, layer fusion and memory planning. This module does none
    of that. It deserialises the resulting plan (milliseconds), allocates
    buffers once, and runs.

    That split is the point. A server whose startup includes a TensorRT build
    has a multi-minute cold start and cannot boot on a machine unlike its build
    host. Build once, load fast.

WHAT AN EXECUTION CONTEXT IS

    The engine is immutable and holds the weights and the plan. A *context*
    holds the mutable state of one in-flight inference: the activation memory
    and the current input shape. One engine can serve several contexts, which
    is how TensorRT does concurrency — and why Phase 12's concurrency work has
    a natural home here later. One context per engine is enough for a
    single-threaded batch loop, which is what this is.

WHY DEVICE BUFFERS ARE TORCH TENSORS

    TensorRT wants raw device pointers. Getting them means either a CUDA
    binding library (pycuda, cuda-python) or something that already allocates
    device memory — and torch is already a core dependency, `tensor.data_ptr()`
    is a device pointer, and torch's caching allocator is better than anything
    written here would be.

    Note this does mean torch's allocator now owns the input/output buffers, so
    they *are* visible to torch.cuda.memory_allocated() while TensorRT's own
    activation arena is not. Cross-backend VRAM still has to come from the
    driver.

SHAPES

    The engine was built with an optimisation profile spanning [min, max]
    batch. Each call sets the concrete input shape on the context, which is
    what selects the specialised kernels for that shape. A batch outside the
    profile is a hard error, not a slow path — that is the trade TensorRT makes
    for its speed.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from src.config import Precision, Settings
from src.inference.base import (
    EngineError,
    EngineMetadata,
    EngineNotAvailableError,
    InferenceEngine,
    InferenceResult,
    StageTimings,
)

ENGINE_FILE = "model.engine"


class TensorRTEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine = None
        self._context = None
        self._runtime = None  # must outlive the engine it deserialised
        self._stream = None
        self._input_name = ""
        self._output_name = ""
        self._torch_dtype = None
        self._num_classes = 1000
        self._output_buffer = None

    # --- lifecycle ------------------------------------------------------

    def engine_path(self) -> Path:
        """models/<name>/<version>/<precision>/model.engine

        Precision is part of the path, not a flag, because a TensorRT engine is
        compiled for one precision. Two precisions are two artifacts.
        """
        return self._settings.model_dir / self._settings.precision.value / ENGINE_FILE

    def load(self) -> None:
        s = self._settings
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise EngineNotAvailableError(
                "tensorrt is not installed. Run: uv sync --extra trt"
            ) from exc
        import torch

        if not s.is_cuda:
            raise EngineNotAvailableError(
                "TensorRT is a CUDA-only runtime; DEVICE=cpu is not supported"
            )
        if not torch.cuda.is_available():
            raise EngineNotAvailableError("TensorRT requires CUDA, which is unavailable")

        path = self.engine_path()
        if not path.is_file():
            raise EngineNotAvailableError(
                f"no TensorRT engine at {path}\n"
                f"  run: uv run python scripts/build_engine.py "
                f"--precision {s.precision.value}"
            )

        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            # PRD failure case: engine incompatibility. An engine built for a
            # different GPU architecture, TensorRT version or driver fails
            # here, and the message has to say so because the file itself looks
            # perfectly fine on disk.
            raise EngineNotAvailableError(
                f"could not deserialise {path}.\n"
                "  TensorRT engines are tied to the GPU architecture, TensorRT version\n"
                f"  and driver they were built with (this is TensorRT {trt.__version__}).\n"
                "  Rebuild it: uv run python scripts/build_engine.py --force "
                f"--precision {s.precision.value}"
            )

        self._engine = engine
        self._context = engine.create_execution_context()
        if self._context is None:
            raise EngineNotAvailableError(
                "could not create a TensorRT execution context (out of GPU memory?)"
            )

        self._read_io_metadata(trt, torch)
        # A dedicated stream, so this engine's work is not serialised behind
        # whatever else is queued on torch's default stream.
        self._stream = torch.cuda.Stream(device=s.device)

    def _read_io_metadata(self, trt, torch) -> None:
        engine = self._engine
        inputs, outputs = [], []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise EngineNotAvailableError(
                f"expected 1 input and 1 output, got {len(inputs)} and {len(outputs)}"
            )
        self._input_name, self._output_name = inputs[0], outputs[0]

        trt_dtype = engine.get_tensor_dtype(self._input_name)
        self._torch_dtype = torch.float16 if trt_dtype == trt.DataType.HALF else torch.float32

        # The engine's declared dtype is the authority. If PRECISION says fp16
        # and the engine is fp32, the numbers would be right and every FP16
        # claim about them would be false.
        # An INT8 QDQ engine still takes fp32 in: the QuantizeLinear node is
        # inside the graph, so quantisation happens on the device rather than
        # in the caller.
        expected = torch.float16 if self._settings.precision is Precision.FP16 else torch.float32
        if self._torch_dtype is not expected:
            raise EngineNotAvailableError(
                f"PRECISION={self._settings.precision.value} but {self.engine_path()} "
                f"was built for {trt_dtype}. Rebuild it with --precision "
                f"{self._settings.precision.value}."
            )

        out_shape = engine.get_tensor_shape(self._output_name)
        if len(out_shape) == 2 and out_shape[1] > 0:
            self._num_classes = int(out_shape[1])

    def unload(self) -> None:
        # Ordered deliberately: context, then engine, then runtime. The runtime
        # owns the engine and the engine owns the context; releasing them the
        # other way round can crash inside TensorRT during interpreter
        # shutdown, which is a spectacular way to fail a graceful stop.
        self._output_buffer = None
        self._context = None
        self._engine = None
        self._runtime = None
        self._stream = None

    @property
    def is_loaded(self) -> bool:
        return self._context is not None

    @property
    def metadata(self) -> EngineMetadata:
        s = self._settings
        return EngineMetadata(
            backend="tensorrt",
            model_name=s.model_name,
            model_version=s.model_version,
            precision=s.precision.value,
            # TensorRT picks per-layer tactics, so the *engine* is the record of
            # what precision the arithmetic runs at. Reporting the precision it
            # was compiled for is the honest summary.
            math_mode=s.precision.value,
            device=s.device,
            input_shape=(3, s.image_size, s.image_size),
            max_batch_size=s.max_batch_size,
            num_classes=self._num_classes,
        )

    # --- inference ------------------------------------------------------

    def _predict(self, batch: np.ndarray) -> InferenceResult:
        import torch

        assert self._context is not None
        n = batch.shape[0]
        device = self._settings.device

        with torch.cuda.stream(self._stream):
            t0 = perf_counter()
            device_input = torch.from_numpy(batch).to(
                device=device, dtype=self._torch_dtype, non_blocking=False
            )
            self._stream.synchronize()
            t1 = perf_counter()

            # Setting the shape is what selects the kernels specialised for it
            # within the built profile. Outside [min, max] this fails.
            if not self._context.set_input_shape(self._input_name, tuple(device_input.shape)):
                raise EngineError(
                    f"batch of {n} is outside this engine's optimisation profile. "
                    "Rebuild with a larger --max-batch."
                )

            output = torch.empty((n, self._num_classes), dtype=self._torch_dtype, device=device)
            self._context.set_tensor_address(self._input_name, device_input.data_ptr())
            self._context.set_tensor_address(self._output_name, output.data_ptr())

            try:
                if not self._context.execute_async_v3(self._stream.cuda_stream):
                    raise EngineError("TensorRT execution failed")
                self._stream.synchronize()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise EngineError(f"CUDA out of memory on a batch of {n} (TensorRT).") from exc
            t2 = perf_counter()

            host = output.float().cpu().numpy()
            self._stream.synchronize()
            t3 = perf_counter()

        return InferenceResult(
            logits=np.ascontiguousarray(host, dtype=np.float32),
            timings=StageTimings(
                h2d_ms=(t1 - t0) * 1000.0,
                compute_ms=(t2 - t1) * 1000.0,
                d2h_ms=(t3 - t2) * 1000.0,
            ),
        )
