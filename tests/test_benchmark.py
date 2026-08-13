"""Benchmark harness tests.

The harness produces the numbers every later claim in this project rests on, so
the things worth testing are the ones that would silently corrupt a result:
percentile arithmetic, throughput definition, and the CSV/JSON round trip.

Uses the FakeEngine pattern rather than a GPU so these run anywhere.
"""

import csv
import json

import numpy as np
import pytest

from src.benchmark import (
    NOISE_FLOOR,
    P99_MIN_ITERATIONS,
    BenchmarkResult,
    benchmark_engine,
    compare_logits,
    sweep_batch_sizes,
    write_results,
)
from src.inference.base import (
    EngineError,
    EngineMetadata,
    InferenceEngine,
    InferenceResult,
    StageTimings,
)

SHAPE = (3, 224, 224)


class FakeEngine(InferenceEngine):
    """Conforming engine that can be told to fail above a batch size."""

    def __init__(self, max_batch_size: int = 64, oom_above: int | None = None):
        self._loaded = True
        self._max = max_batch_size
        self._oom_above = oom_above
        self.seen_batches: list[int] = []

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
            math_mode="fp32",
            device="cpu",
            input_shape=SHAPE,
            max_batch_size=self._max,
            num_classes=10,
        )

    def _predict(self, batch: np.ndarray) -> InferenceResult:
        n = batch.shape[0]
        self.seen_batches.append(n)
        if self._oom_above is not None and n > self._oom_above:
            raise EngineError(f"CUDA out of memory on a batch of {n}")
        return InferenceResult(
            logits=np.zeros((n, 10), dtype=np.float32),
            timings=StageTimings(h2d_ms=0.5, compute_ms=2.0, d2h_ms=0.25),
        )


def test_result_carries_enough_context_to_be_a_result():
    # A bare latency number is not a finding; it needs backend/precision/device.
    r = benchmark_engine(FakeEngine(), batch_size=4, iterations=10, warmup=2)
    assert (r.backend, r.precision, r.device, r.batch_size) == ("fake", "fp32", "cpu", 4)
    assert r.iterations == 10


def test_warmup_runs_are_not_measured():
    engine = FakeEngine()
    r = benchmark_engine(engine, batch_size=2, iterations=10, warmup=5)
    assert len(engine.seen_batches) == 15  # warmup + measured
    assert r.iterations == 10  # only the measured ones counted


def test_warmup_uses_the_same_batch_size_as_the_measurement():
    # cuDNN caches an algorithm per input shape. Warming at a different shape
    # would leave autotuning cost inside the first measured iteration.
    engine = FakeEngine()
    benchmark_engine(engine, batch_size=8, iterations=3, warmup=4)
    assert set(engine.seen_batches) == {8}


def test_percentiles_are_ordered():
    r = benchmark_engine(FakeEngine(), batch_size=1, iterations=100, warmup=2)
    assert r.min_ms <= r.p50_ms <= r.p90_ms <= r.p95_ms <= r.p99_ms <= r.max_ms


def test_throughput_matches_latency_and_batch_size():
    r = benchmark_engine(FakeEngine(), batch_size=8, iterations=50, warmup=2)
    # throughput = images / wall time, so images/sec ~= batch / mean_seconds.
    implied = r.batch_size / (r.mean_ms / 1000.0)
    assert r.throughput_ips == pytest.approx(implied, rel=0.25)


def test_latency_per_image_is_per_image():
    r = benchmark_engine(FakeEngine(), batch_size=16, iterations=20, warmup=2)
    assert r.latency_per_image_ms == pytest.approx(r.p50_ms / 16, rel=1e-6)


def test_stage_medians_are_reported():
    r = benchmark_engine(FakeEngine(), batch_size=2, iterations=20, warmup=2)
    assert (r.h2d_ms, r.compute_ms, r.d2h_ms) == (0.5, 2.0, 0.25)


def test_p99_is_flagged_when_the_sample_is_too_small():
    small = benchmark_engine(FakeEngine(), batch_size=1, iterations=50, warmup=1)
    assert not small.p99_is_meaningful
    assert P99_MIN_ITERATIONS >= 500


def test_sweep_survives_an_oom_and_keeps_earlier_rows():
    # Finding the ceiling is part of the result; it must not discard the rows
    # already collected below it.
    engine = FakeEngine(oom_above=8)
    results = sweep_batch_sizes(engine, [1, 4, 8, 16, 32], iterations=5, warmup=1)
    assert [r.batch_size for r in results] == [1, 4, 8]


def test_noise_floor_is_documented_and_nonzero():
    # Guards against someone "cleaning up" the constant that stops a 2% thermal
    # artefact from being reported as a finding.
    assert 0 < NOISE_FLOOR < 0.5


# --- persistence ----------------------------------------------------------


@pytest.fixture
def results():
    engine = FakeEngine()
    return sweep_batch_sizes(engine, [1, 2, 4], iterations=10, warmup=1)


def test_json_carries_the_environment(results, tmp_path):
    # A benchmark without its machine is a number without units.
    json_path, _ = write_results(results, tmp_path, "sweep")
    payload = json.loads(json_path.read_text())
    assert "torch" in payload["environment"]
    assert "timestamp" in payload["environment"]
    assert len(payload["results"]) == 3


def test_csv_round_trips_every_field(results, tmp_path):
    _, csv_path = write_results(results, tmp_path, "sweep")
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 3
    assert [int(r["batch_size"]) for r in rows] == [1, 2, 4]
    # Every dataclass field must survive to CSV, or the saved results are
    # quietly less useful than the console output.
    assert set(rows[0]) == set(BenchmarkResult.__dataclass_fields__)


def test_write_results_creates_missing_directories(results, tmp_path):
    out = tmp_path / "deep" / "nested"
    paths = write_results(results, out, "sweep")
    assert all(p.is_file() for p in paths)


# --- numerical agreement --------------------------------------------------


def _logits(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def test_identical_logits_agree_perfectly():
    a = _logits([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    r = compare_logits(a, a)
    assert r.top1_agreement == 1.0
    assert r.top5_agreement == 1.0
    assert r.max_abs_logit_diff == 0.0
    assert r.max_confidence_delta == 0.0
    assert r.samples == 2


def test_small_drift_keeps_top1_but_shows_in_logits():
    # The FP16 signature: identical predictions, non-zero numerical distance.
    # Reporting only top-1 would call this "no difference", which is how a
    # precision regression hides until the inputs change.
    ref = _logits([[5.0, 1.0, 0.0]])
    cand = ref + np.array([[0.01, -0.01, 0.005]], dtype=np.float32)
    r = compare_logits(ref, cand)
    assert r.top1_agreement == 1.0
    assert r.max_abs_logit_diff == pytest.approx(0.01, abs=1e-6)
    assert r.max_confidence_delta > 0


def test_flipped_prediction_is_caught():
    ref = _logits([[5.0, 4.9, 0.0]])
    cand = _logits([[4.9, 5.0, 0.0]])
    r = compare_logits(ref, cand)
    assert r.top1_agreement == 0.0
    # Still in the candidate's top 5, so the softer check does not fire --
    # which is the distinction the two metrics exist to draw.
    assert r.top5_agreement == 1.0


def test_top5_agreement_fails_when_the_class_falls_out_of_range():
    ref = _logits([[9.0] + [0.0] * 9])  # reference picks class 0
    cand = _logits([[-9.0] + [float(i) for i in range(9)]])  # class 0 now last
    r = compare_logits(ref, cand)
    assert r.top1_agreement == 0.0
    assert r.top5_agreement == 0.0


def test_partial_agreement_is_a_fraction():
    ref = _logits([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
    cand = _logits([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    assert compare_logits(ref, cand).top1_agreement == 0.75


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="cannot compare"):
        compare_logits(np.zeros((2, 10), np.float32), np.zeros((2, 8), np.float32))


def test_unbatched_logits_are_rejected():
    with pytest.raises(ValueError, match="N, num_classes"):
        compare_logits(np.zeros(10, np.float32), np.zeros(10, np.float32))
