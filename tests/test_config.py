"""Config is the first thing that can silently ruin an experiment.

If `PRECISION=fp16` is misspelled and silently ignored, you spend an afternoon
benchmarking FP32 and writing up FP16 conclusions. These tests exist so that
class of mistake is impossible rather than merely unlikely.
"""

import pytest
from pydantic import ValidationError

from src.config import Backend, Precision, Settings


def test_defaults_are_the_documented_ones():
    s = Settings(_env_file=None)
    assert s.backend is Backend.PYTORCH
    assert s.precision is Precision.FP32
    assert s.image_size == 224
    assert s.max_batch_size == 8


def test_model_dir_is_composed_not_hardcoded():
    s = Settings(_env_file=None, model_name="mobilenetv3", model_version="v7")
    assert s.model_dir.as_posix() == "models/mobilenetv3/v7"


@pytest.mark.parametrize(
    ("device", "is_cuda", "index"),
    [("cuda:0", True, 0), ("cuda:1", True, 1), ("cuda", True, 0), ("cpu", False, None)],
)
def test_device_parsing(device, is_cuda, index):
    s = Settings(_env_file=None, device=device)
    assert s.is_cuda is is_cuda
    assert s.device_index == index


@pytest.mark.parametrize("bad", ["gpu", "cuda:", "cuda:x", "cuda0", ""])
def test_bad_device_fails_at_startup_not_at_inference(bad):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, device=bad)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, backend="tensorflow")


def test_typo_in_precision_is_rejected_rather_than_ignored():
    # The failure mode this guards: silently running FP32 while the report says FP16.
    with pytest.raises(ValidationError):
        Settings(_env_file=None, precision="float16")


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_batch_size", 0), ("queue_max_size", 0), ("request_timeout_ms", 0), ("image_size", 8)],
)
def test_nonsense_limits_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_env_vars_are_read(monkeypatch):
    monkeypatch.setenv("BACKEND", "tensorrt")
    monkeypatch.setenv("MAX_BATCH_SIZE", "32")
    s = Settings(_env_file=None)
    assert s.backend is Backend.TENSORRT
    assert s.max_batch_size == 32
