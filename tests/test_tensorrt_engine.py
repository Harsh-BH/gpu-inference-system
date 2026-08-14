"""TensorRT backend tests. Skip cleanly when no engine has been built."""

import numpy as np
import pytest
import torch

from src.benchmark import compare_logits
from src.config import Backend, Precision, Settings
from src.inference import create_engine
from src.inference.base import EngineError, EngineNotAvailableError
from src.inference.pytorch_engine import PyTorchEngine

pytest.importorskip("tensorrt")
from src.inference.tensorrt_engine import TensorRTEngine  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def settings(**kw) -> Settings:
    kw.setdefault("backend", Backend.TENSORRT)
    return Settings(_env_file=None, **kw)


@pytest.fixture(scope="module")
def built():
    s = settings()
    if not TensorRTEngine(s).engine_path().is_file():
        pytest.skip("no engine: uv run python scripts/build_engine.py")
    return s


@pytest.fixture
def batch(built):
    return np.random.default_rng(0).standard_normal((4, 3, 224, 224), dtype=np.float32)


def test_create_engine_selects_tensorrt(built):
    assert isinstance(create_engine(built), TensorRTEngine)


def test_engine_path_includes_precision(built):
    # Two precisions are two artifacts, not one artifact plus a flag.
    assert TensorRTEngine(settings()).engine_path().parent.name == "fp32"
    assert TensorRTEngine(settings(precision=Precision.FP16)).engine_path().parent.name == "fp16"


def test_missing_engine_points_at_the_build_step(tmp_path):
    with pytest.raises(EngineNotAvailableError, match="build_engine"):
        TensorRTEngine(settings(model_repository=tmp_path)).load()


def test_cpu_is_refused(built):
    with pytest.raises(EngineNotAvailableError, match="CUDA-only"):
        TensorRTEngine(settings(device="cpu")).load()


@cuda_only
def test_unload_is_idempotent(built):
    engine = TensorRTEngine(built)
    engine.load()
    engine.unload()
    engine.unload()
    assert not engine.is_loaded


@cuda_only
def test_agrees_with_pytorch(built, batch):
    with PyTorchEngine(settings(backend=Backend.PYTORCH)) as ref:
        expected = ref.predict(batch).logits
    with TensorRTEngine(built) as engine:
        got = engine.predict(batch).logits
    agreement = compare_logits(expected, got)
    assert agreement.top1_agreement == 1.0
    assert agreement.max_abs_logit_diff < 1e-3


@cuda_only
@pytest.mark.parametrize("n", [1, 3, 8])
def test_dynamic_batch_within_the_profile(built, n):
    with TensorRTEngine(settings(max_batch_size=8)) as engine:
        assert engine.predict(np.zeros((n, 3, 224, 224), np.float32)).logits.shape == (n, 1000)


@cuda_only
def test_batch_beyond_the_profile_is_a_clear_error(built):
    # The engine was built with max_batch 32. Asking for more must say so
    # rather than fail somewhere inside TensorRT.
    with TensorRTEngine(settings(max_batch_size=64)) as engine:
        with pytest.raises(EngineError, match="optimisation profile|out of memory"):
            engine.predict(np.zeros((64, 3, 224, 224), np.float32))


@cuda_only
def test_fp16_engine_agrees_on_top1(built, batch):
    fp16 = settings(precision=Precision.FP16)
    if not TensorRTEngine(fp16).engine_path().is_file():
        pytest.skip("no fp16 engine")
    with PyTorchEngine(settings(backend=Backend.PYTORCH)) as ref:
        expected = ref.predict(batch).logits
    with TensorRTEngine(fp16) as engine:
        got = engine.predict(batch).logits
    assert got.dtype == np.float32  # interface contract holds regardless
    assert compare_logits(expected, got).top1_agreement == 1.0
