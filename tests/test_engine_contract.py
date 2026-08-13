"""Tests for the InferenceEngine contract itself, using a fake backend.

Two things are being verified, and the second is the important one:

1. Validation actually rejects malformed batches.
2. The contract is implementable with *only* numpy.

The FakeEngine below imports no torch, no onnxruntime, no tensorrt. If someone
later "helpfully" types the interface in torch.Tensor, this file stops
importing — which is exactly the alarm we want, because that change would
silently force a torch dependency into the TensorRT-only deployment.
"""

import numpy as np
import pytest

from src.inference.base import (
    EngineError,
    EngineMetadata,
    EngineNotLoadedError,
    InferenceEngine,
    InferenceResult,
    StageTimings,
)

SHAPE = (3, 224, 224)
CLASSES = 1000


class FakeEngine(InferenceEngine):
    """Minimal conforming backend. Counts calls so warmup is observable."""

    def __init__(self, max_batch_size: int = 8):
        self._loaded = False
        self._max_batch = max_batch_size
        self.calls = 0

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            backend="fake",
            model_name="fake",
            model_version="v1",
            precision="fp32",
            device="cpu",
            input_shape=SHAPE,
            max_batch_size=self._max_batch,
            num_classes=CLASSES,
        )

    def _predict(self, batch: np.ndarray) -> InferenceResult:
        self.calls += 1
        return InferenceResult(
            logits=np.zeros((batch.shape[0], CLASSES), dtype=np.float32),
            timings=StageTimings(h2d_ms=0.1, compute_ms=1.0, d2h_ms=0.05),
        )


def batch(n: int = 1, shape: tuple[int, ...] = SHAPE, dtype=np.float32) -> np.ndarray:
    return np.zeros((n, *shape), dtype=dtype)


@pytest.fixture
def engine():
    with FakeEngine() as e:
        yield e


def test_predict_before_load_is_a_clear_error():
    e = FakeEngine()
    with pytest.raises(EngineNotLoadedError):
        e.predict(batch())


def test_context_manager_loads_and_unloads():
    e = FakeEngine()
    assert not e.is_loaded
    with e:
        assert e.is_loaded
    assert not e.is_loaded


def test_unload_is_idempotent():
    # Runs on the shutdown path; a second call during a signal race must not raise.
    e = FakeEngine()
    e.load()
    e.unload()
    e.unload()


def test_happy_path_shape_and_timings(engine):
    result = engine.predict(batch(4))
    assert result.logits.shape == (4, CLASSES)
    assert result.timings.total_ms == pytest.approx(1.15)


def test_wrong_dtype_is_rejected(engine):
    # The failure this prevents: TensorRT does not crash on a wrong dtype, it
    # reinterprets the bytes and returns confident garbage.
    with pytest.raises(EngineError, match="float32"):
        engine.predict(batch(dtype=np.float64))


def test_wrong_spatial_shape_is_rejected(engine):
    with pytest.raises(EngineError, match="per-sample shape"):
        engine.predict(batch(1, (3, 256, 256)))


def test_wrong_channel_count_is_rejected(engine):
    with pytest.raises(EngineError, match="per-sample shape"):
        engine.predict(batch(1, (1, 224, 224)))


def test_missing_batch_dimension_is_rejected(engine):
    # A very common bug: forgetting to unsqueeze a single image to (1, C, H, W).
    with pytest.raises(EngineError, match="4-D"):
        engine.predict(np.zeros(SHAPE, dtype=np.float32))


def test_oversized_batch_is_rejected(engine):
    with pytest.raises(EngineError, match="max_batch_size"):
        engine.predict(batch(9))


def test_empty_batch_is_rejected(engine):
    with pytest.raises(EngineError):
        engine.predict(batch(0))


def test_non_array_input_is_rejected(engine):
    with pytest.raises(EngineError, match="np.ndarray"):
        engine.predict([[1.0]])


def test_warmup_runs_requested_iterations(engine):
    engine.warmup(5)
    assert engine.calls == 5


def test_warmup_zero_is_a_noop(engine):
    engine.warmup(0)
    assert engine.calls == 0
