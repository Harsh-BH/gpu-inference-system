"""Integration tests for the PyTorch backend.

These need the model repository provisioned:

    uv run python scripts/fetch_model.py

GPU tests skip cleanly when CUDA is unavailable, so the suite still runs on a
CPU-only machine -- otherwise "the tests pass" would mean nothing on CI.
"""

import numpy as np
import pytest
import torch

from src.config import Precision, Settings
from src.inference import create_engine
from src.inference.base import EngineNotAvailableError
from src.inference.pytorch_engine import PyTorchEngine
from src.model.pytorch_model import WEIGHTS_FILE, load_from_repository

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def settings(**kw) -> Settings:
    # _env_file=None so a developer's local .env cannot change what the tests mean.
    return Settings(_env_file=None, **kw)


@pytest.fixture(scope="module")
def provisioned():
    s = settings()
    if not (s.model_dir / WEIGHTS_FILE).is_file():
        pytest.skip("model not provisioned: uv run python scripts/fetch_model.py")
    return s


@pytest.fixture
def sample_batch(provisioned):
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, 3, 224, 224), dtype=np.float32)


# --- lifecycle ------------------------------------------------------------


def test_metadata_available_before_load(provisioned):
    # /ready needs to describe the engine while it is still loading.
    meta = PyTorchEngine(provisioned).metadata
    assert meta.backend == "pytorch"
    assert meta.input_shape == (3, 224, 224)
    assert meta.num_classes == 1000


def test_unload_is_idempotent(provisioned):
    engine = PyTorchEngine(provisioned)
    engine.load()
    engine.unload()
    engine.unload()
    assert not engine.is_loaded


def test_missing_artifacts_give_an_actionable_error(tmp_path):
    engine = PyTorchEngine(settings(model_repository=tmp_path))
    with pytest.raises(EngineNotAvailableError, match="fetch_model"):
        engine.load()


def test_fp16_on_cpu_is_refused(provisioned):
    engine = PyTorchEngine(settings(device="cpu", precision=Precision.FP16))
    with pytest.raises(EngineNotAvailableError, match="fp16"):
        engine.load()


def test_create_engine_selects_by_config(provisioned):
    assert isinstance(create_engine(provisioned), PyTorchEngine)


# --- correctness ----------------------------------------------------------


def test_cpu_inference_works(provisioned, sample_batch):
    with PyTorchEngine(settings(device="cpu")) as engine:
        result = engine.predict(sample_batch)
    assert result.logits.shape == (2, 1000)
    assert result.logits.dtype == np.float32
    assert np.isfinite(result.logits).all()


@cuda_only
def test_gpu_matches_a_direct_torchvision_forward_pass(provisioned, sample_batch):
    """The engine must not be doing anything the naive reference does not.

    Chiefly this catches a forgotten model.eval(): with BatchNorm in training
    mode the model still runs and still returns plausible logits, it just
    normalises by the current batch instead of the learned population
    statistics. Nothing crashes; the answers are simply worse.
    """
    reference = load_from_repository(provisioned.model_dir, provisioned.model_name)
    reference.eval().cuda()
    with torch.inference_mode():
        expected = reference(torch.from_numpy(sample_batch).cuda()).cpu().numpy()

    with PyTorchEngine(provisioned) as engine:
        got = engine.predict(sample_batch).logits

    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=0)


@cuda_only
def test_inference_is_deterministic(provisioned, sample_batch):
    with PyTorchEngine(provisioned) as engine:
        a = engine.predict(sample_batch).logits
        b = engine.predict(sample_batch).logits
    np.testing.assert_array_equal(a, b)


@cuda_only
def test_cpu_and_gpu_agree(provisioned, sample_batch):
    with PyTorchEngine(settings(device="cpu")) as engine:
        cpu_logits = engine.predict(sample_batch).logits
    with PyTorchEngine(provisioned) as engine:
        gpu_logits = engine.predict(sample_batch).logits
    # Not bit-identical: cuDNN picks different convolution algorithms with
    # different summation orders, and float addition is not associative. But
    # with TF32 correctly disabled the disagreement is ~5e-06, not ~4e-03.
    # If this assertion starts failing, TF32 has leaked back on.
    np.testing.assert_allclose(gpu_logits, cpu_logits, atol=1e-4, rtol=0)
    assert (gpu_logits.argmax(1) == cpu_logits.argmax(1)).all()


@cuda_only
def test_tf32_is_off_by_default_and_costs_precision_when_on(provisioned, sample_batch):
    """Pins the finding that motivated the ALLOW_TF32 setting.

    PyTorch enables TF32 convolutions on Ampere by default, silently trading
    13 mantissa bits for Tensor Core throughput. That is a defensible
    production choice and an indefensible baseline, since Phases 5/7/9 all
    measure other runtimes against 'PyTorch FP32'.

    This test exists so that a future torch upgrade flipping the default back
    on is a red test rather than a quietly wrong benchmark.
    """
    with PyTorchEngine(settings(device="cpu")) as engine:
        cpu_logits = engine.predict(sample_batch).logits

    with PyTorchEngine(provisioned) as engine:
        assert torch.backends.cudnn.allow_tf32 is False
        honest = np.abs(engine.predict(sample_batch).logits - cpu_logits).max()

    with PyTorchEngine(settings(allow_tf32=True)) as engine:
        assert torch.backends.cudnn.allow_tf32 is True
        tf32 = np.abs(engine.predict(sample_batch).logits - cpu_logits).max()

    assert honest < 1e-4, "FP32 path should agree with CPU to ~1e-6"
    assert tf32 > honest * 10, "TF32 should visibly degrade numerical agreement"


@cuda_only
def test_fp16_picks_the_same_classes_as_fp32(provisioned, sample_batch):
    # Groundwork for Phase 5. Halving precision must not change *what* the
    # model predicts, only how precisely the logits are represented.
    with PyTorchEngine(provisioned) as engine:
        fp32 = engine.predict(sample_batch).logits
    with PyTorchEngine(settings(precision=Precision.FP16)) as engine:
        fp16 = engine.predict(sample_batch).logits

    assert fp16.dtype == np.float32  # interface contract holds regardless
    assert (fp32.argmax(1) == fp16.argmax(1)).all()
    np.testing.assert_allclose(fp16, fp32, atol=0.05, rtol=0)


# --- timings --------------------------------------------------------------


@cuda_only
def test_stage_timings_are_populated_and_ordered(provisioned, sample_batch):
    with PyTorchEngine(provisioned) as engine:
        engine.warmup(3)
        t = engine.predict(sample_batch).timings

    assert t.h2d_ms > 0 and t.compute_ms > 0 and t.d2h_ms > 0
    assert t.total_ms == pytest.approx(t.h2d_ms + t.compute_ms + t.d2h_ms)
    # 4 KB of logits coming back must be cheaper than ~50 conv kernels. If this
    # ever fails, the synchronize calls have been dropped and the numbers are
    # measuring launch overhead rather than execution.
    assert t.d2h_ms < t.compute_ms
