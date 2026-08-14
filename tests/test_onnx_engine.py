"""ONNX Runtime backend tests.

The most important one is `test_cuda_provider_is_actually_active`. ORT's
provider fallback is silent: ask for CUDA, get CPU, receive correct logits an
order of magnitude slower, and never be told. That failure happened on this
machine during development (libcublasLt.so.13 could not be loaded), and it is
the reason load() verifies what it got rather than what it asked for.
"""

import numpy as np
import pytest
import torch

from src.benchmark import compare_logits
from src.config import Backend, Precision, Settings
from src.inference import create_engine
from src.inference.base import EngineError, EngineNotAvailableError
from src.inference.pytorch_engine import PyTorchEngine
from src.model.pytorch_model import WEIGHTS_FILE

pytest.importorskip("onnxruntime")
from src.inference.onnx_engine import ONNXRuntimeEngine  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def settings(**kw) -> Settings:
    kw.setdefault("backend", Backend.ONNXRUNTIME)
    return Settings(_env_file=None, **kw)


@pytest.fixture(scope="module")
def exported():
    s = settings()
    if not (s.model_dir / "model.onnx").is_file():
        pytest.skip("no model.onnx: uv run python scripts/export_onnx.py")
    if not (s.model_dir / WEIGHTS_FILE).is_file():
        pytest.skip("model not provisioned")
    return s


@pytest.fixture
def batch(exported):
    return np.random.default_rng(0).standard_normal((4, 3, 224, 224), dtype=np.float32)


# --- selection and lifecycle ---------------------------------------------


def test_create_engine_selects_ort_from_config(exported):
    assert isinstance(create_engine(exported), ONNXRuntimeEngine)


def test_metadata_reports_the_right_backend(exported):
    meta = ONNXRuntimeEngine(exported).metadata
    assert meta.backend == "onnxruntime"
    assert meta.input_shape == (3, 224, 224)


def test_unload_is_idempotent(exported):
    engine = ONNXRuntimeEngine(exported)
    engine.load()
    engine.unload()
    engine.unload()
    assert not engine.is_loaded


def test_missing_onnx_file_points_at_the_fix(tmp_path):
    engine = ONNXRuntimeEngine(settings(model_repository=tmp_path))
    with pytest.raises(EngineNotAvailableError, match="export_onnx"):
        engine.load()


def test_fp16_is_refused_rather_than_silently_run_as_fp32(exported):
    # ORT does not convert precision at load the way torch's .half() does.
    # Quietly running fp32 while the report says fp16 is the failure this
    # project exists to prevent, so the limitation is explicit.
    engine = ONNXRuntimeEngine(settings(precision=Precision.FP16))
    with pytest.raises(EngineNotAvailableError, match="fp32 only"):
        engine.load()


# --- the silent-fallback guard -------------------------------------------


@cuda_only
def test_cuda_provider_is_actually_active(exported):
    """Requesting CUDA must mean getting CUDA, or failing loudly.

    Without this check the session still works and still returns correct
    answers, just on the CPU. Every 'ONNX Runtime CUDA' number in this repo
    would be a CPU number.
    """
    engine = ONNXRuntimeEngine(exported)
    engine.load()
    try:
        assert "CUDAExecutionProvider" in engine._session.get_providers()
    finally:
        engine.unload()


def test_cpu_device_runs_without_cuda(exported, batch):
    with ONNXRuntimeEngine(settings(device="cpu")) as engine:
        result = engine.predict(batch)
    assert result.logits.shape == (4, 1000)
    assert np.isfinite(result.logits).all()


# --- correctness ----------------------------------------------------------


def test_agrees_with_pytorch(exported, batch):
    """Same weights, different runtime. Disagreement here is kernel choice and
    summation order only -- float addition is not associative."""
    with PyTorchEngine(settings(backend=Backend.PYTORCH, device="cpu")) as torch_engine:
        expected = torch_engine.predict(batch).logits
    with ONNXRuntimeEngine(settings(device="cpu")) as ort_engine:
        got = ort_engine.predict(batch).logits

    agreement = compare_logits(expected, got)
    assert agreement.top1_agreement == 1.0
    assert agreement.max_abs_logit_diff < 1e-3


@cuda_only
def test_gpu_agrees_with_cpu(exported, batch):
    with ONNXRuntimeEngine(settings(device="cpu")) as engine:
        cpu_logits = engine.predict(batch).logits
    with ONNXRuntimeEngine(exported) as engine:
        gpu_logits = engine.predict(batch).logits
    assert compare_logits(cpu_logits, gpu_logits).top1_agreement == 1.0


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_dynamic_batch_sizes(exported, n):
    with ONNXRuntimeEngine(settings(max_batch_size=8, device="cpu")) as engine:
        out = engine.predict(np.zeros((n, 3, 224, 224), dtype=np.float32))
    assert out.logits.shape == (n, 1000)


def test_shared_validation_applies_to_this_backend_too(exported):
    # predict() is concrete on the ABC, so no backend can skip it.
    with ONNXRuntimeEngine(settings(device="cpu")) as engine:
        with pytest.raises(EngineError, match="float32"):
            engine.predict(np.zeros((1, 3, 224, 224), dtype=np.float64))


# --- timings --------------------------------------------------------------


@cuda_only
def test_stage_timings_are_separated_by_iobinding(exported, batch):
    """session.run() would report h2d and d2h as zero.

    IOBinding exists here so the stage breakdown is comparable with the
    PyTorch backend rather than one opaque number.
    """
    with ONNXRuntimeEngine(exported) as engine:
        engine.warmup(5)
        t = engine.predict(batch).timings
    assert t.h2d_ms > 0
    assert t.compute_ms > 0
    assert t.d2h_ms > 0
    assert t.d2h_ms < t.compute_ms  # 4 KB back vs a full forward pass
