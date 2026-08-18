"""The three concrete stages, each tested against the Stage contract.

The contract the pipeline runner relies on and will not re-check per stage:

    1. `len(process(items)) == len(items)`
    2. result i belongs to item i
    3. a per-item failure is *returned*, a batch-wide failure is *raised*

Every stage here is checked against all three, because the runner trusts them.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.inference.base import (
    EngineError,
    EngineMetadata,
    InferenceEngine,
    InferenceResult,
    StageTimings,
)
from src.postprocessing import Prediction
from src.preprocessing import ImagePreprocessor
from src.stages import ClassifyStage, ImageDecodeStage, InferenceStage

SHAPE = (3, 224, 224)
NUM_CLASSES = 10


def make_jpeg(w: int = 400, h: int = 300, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class MarkerEngine(InferenceEngine):
    """Logits derived from the input, so mis-mapping shows up as a wrong marker
    rather than as plausible-looking noise."""

    def __init__(self, *, fail: bool = False, wrong_rows: bool = False) -> None:
        self._loaded = False
        self.fail = fail
        self.wrong_rows = wrong_rows
        self.batch_sizes: list[int] = []
        self.loads = 0
        self.unloads = 0
        self.warmups = 0

    def load(self):
        self._loaded = True
        self.loads += 1

    def unload(self):
        self._loaded = False
        self.unloads += 1

    def warmup(self, iterations: int) -> None:
        self.warmups += iterations

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
            math_mode="fp32",
            device="cpu",
            input_shape=SHAPE,
            max_batch_size=64,
            num_classes=NUM_CLASSES,
        )

    def _predict(self, batch: np.ndarray) -> InferenceResult:
        self.batch_sizes.append(batch.shape[0])
        if self.fail:
            raise EngineError("engine exploded")
        markers = batch[:, 0, 0, 0]
        if self.wrong_rows:
            markers = markers[:-1]
        logits = np.repeat(markers[:, None], NUM_CLASSES, axis=1).astype(np.float32)
        return InferenceResult(
            logits=logits, timings=StageTimings(h2d_ms=0.1, compute_ms=1.0, d2h_ms=0.1)
        )


def marked(marker: float) -> np.ndarray:
    tensor = np.zeros(SHAPE, dtype=np.float32)
    tensor[0, 0, 0] = marker
    return tensor


# --- decode --------------------------------------------------------------


def test_decode_returns_one_tensor_per_input():
    stage = ImageDecodeStage(ImagePreprocessor(image_size=224))
    out = stage.process([make_jpeg(seed=i) for i in range(3)])
    assert len(out) == 3
    assert all(t.shape == SHAPE and t.dtype == np.float32 for t in out)


def test_decode_preserves_order():
    """Item i's tensor must come back in slot i, not merely somewhere."""
    stage = ImageDecodeStage(ImagePreprocessor(image_size=224))
    blobs = [make_jpeg(seed=i) for i in range(4)]
    out = stage.process(blobs)
    for i, blob in enumerate(blobs):
        np.testing.assert_array_equal(out[i], stage.process([blob])[0])


def test_one_corrupt_image_fails_only_its_own_slot():
    """A batched decoder must not fail three good images for one bad one."""
    stage = ImageDecodeStage(ImagePreprocessor(image_size=224))
    out = stage.process([make_jpeg(seed=0), b"not an image", make_jpeg(seed=1)])
    assert len(out) == 3
    assert isinstance(out[0], np.ndarray)
    assert isinstance(out[1], Exception)
    assert isinstance(out[2], np.ndarray)


def test_empty_payload_is_a_per_item_failure_not_a_crash():
    stage = ImageDecodeStage(ImagePreprocessor(image_size=224))
    (result,) = stage.process([b""])
    assert isinstance(result, Exception)


def test_decode_reports_its_output_shape():
    stage = ImageDecodeStage(ImagePreprocessor(image_size=224))
    assert stage.output_shape == SHAPE


# --- inference -----------------------------------------------------------


def test_inference_setup_loads_and_warms_then_teardown_unloads():
    engine = MarkerEngine()
    stage = InferenceStage(engine, warmup_requests=3)
    assert not engine.is_loaded

    stage.setup()
    assert engine.is_loaded
    assert engine.loads == 1
    assert engine.warmups == 3  # paid before any request can arrive

    stage.teardown()
    assert not engine.is_loaded
    assert engine.unloads == 1


def test_inference_maps_row_i_to_item_i():
    """The correctness-critical invariant, at the stage level."""
    stage = InferenceStage(MarkerEngine())
    stage.setup()
    out = stage.process([marked(float(i)) for i in range(6)])
    assert [float(row[0]) for row in out] == [float(i) for i in range(6)]


def test_inference_runs_the_whole_batch_in_one_call():
    engine = MarkerEngine()
    stage = InferenceStage(engine)
    stage.setup()
    stage.process([marked(float(i)) for i in range(5)])
    assert engine.batch_sizes == [5]  # one GPU call, not five


def test_engine_failure_raises_so_the_whole_batch_fails():
    """A dead engine means the batch genuinely did not happen; there is no
    partial success to report, so this raises rather than returning."""
    stage = InferenceStage(MarkerEngine(fail=True))
    stage.setup()
    with pytest.raises(EngineError, match="exploded"):
        stage.process([marked(1.0), marked(2.0)])


def test_a_row_count_mismatch_raises_rather_than_mis_mapping():
    stage = InferenceStage(MarkerEngine(wrong_rows=True))
    stage.setup()
    with pytest.raises(EngineError, match="rows"):
        stage.process([marked(1.0), marked(2.0), marked(3.0)])


def test_engine_timings_are_reported_for_metrics():
    seen: list[tuple] = []
    stage = InferenceStage(MarkerEngine(), on_timings=lambda t, n: seen.append((t, n)))
    stage.setup()
    stage.process([marked(1.0), marked(2.0)])
    assert len(seen) == 1
    timings, batch_size = seen[0]
    assert batch_size == 2
    assert timings.compute_ms > 0


def test_inference_exposes_engine_metadata():
    stage = InferenceStage(MarkerEngine())
    assert stage.metadata.backend == "fake"
    assert stage.engine.metadata.num_classes == NUM_CLASSES


# --- classify ------------------------------------------------------------


def test_classify_ranks_and_keeps_order():
    labels = [f"class{i}" for i in range(NUM_CLASSES)]
    stage = ClassifyStage(labels, k=3)
    # Row i's argmax is class i, so a mis-mapping is immediately visible.
    rows = [np.eye(NUM_CLASSES, dtype=np.float32)[i] * 10.0 for i in range(4)]
    out = stage.process(rows)

    assert len(out) == 4
    for i, ranked in enumerate(out):
        assert len(ranked) == 3
        assert isinstance(ranked[0], Prediction)
        assert ranked[0].label == f"class{i}"


def test_classify_confidences_are_a_distribution():
    labels = [f"class{i}" for i in range(NUM_CLASSES)]
    (ranked,) = ClassifyStage(labels, k=NUM_CLASSES).process(
        [np.arange(NUM_CLASSES, dtype=np.float32)]
    )
    assert sum(p.confidence for p in ranked) == pytest.approx(1.0, abs=1e-5)
    assert ranked == sorted(ranked, key=lambda p: -p.confidence)


# --- the three composed --------------------------------------------------


async def test_the_real_pipeline_end_to_end():
    """Bytes in, ranked predictions out, through the actual assembled stages."""
    import asyncio

    from src.pipeline import Pipeline, StageSpec

    labels = [f"class{i}" for i in range(NUM_CLASSES)]
    pipeline = Pipeline(
        [
            StageSpec(ImageDecodeStage(ImagePreprocessor(image_size=224)), workers=2),
            StageSpec(
                InferenceStage(MarkerEngine()), max_batch=4, max_batch_wait_ms=20.0
            ),
            StageSpec(ClassifyStage(labels), workers=0, max_batch=8),
        ]
    )
    pipeline.start()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(pipeline.submit(make_jpeg(seed=i)) for i in range(6)),
                return_exceptions=True,
            ),
            timeout=10.0,
        )
    finally:
        await pipeline.stop()

    assert len(results) == 6
    for completion in results:
        assert not isinstance(completion, BaseException), completion
        assert isinstance(completion.result[0], Prediction)
        # Every stage recorded itself on the way through.
        assert set(completion.stage_ms) == {"decode", "infer", "classify"}


async def test_a_corrupt_upload_fails_only_that_request():
    import asyncio

    from src.pipeline import Pipeline, StageSpec
    from src.preprocessing import PreprocessingError

    labels = [f"class{i}" for i in range(NUM_CLASSES)]
    pipeline = Pipeline(
        [
            StageSpec(ImageDecodeStage(ImagePreprocessor(image_size=224)), workers=2),
            StageSpec(InferenceStage(MarkerEngine()), max_batch=4, max_batch_wait_ms=20.0),
            StageSpec(ClassifyStage(labels), workers=0, max_batch=8),
        ]
    )
    pipeline.start()
    try:
        payloads = [make_jpeg(seed=0), b"garbage", make_jpeg(seed=1)]
        results = await asyncio.wait_for(
            asyncio.gather(
                *(pipeline.submit(p) for p in payloads), return_exceptions=True
            ),
            timeout=10.0,
        )
    finally:
        await pipeline.stop()

    assert isinstance(results[1], PreprocessingError)
    assert isinstance(results[0].result[0], Prediction)
    assert isinstance(results[2].result[0], Prediction)
