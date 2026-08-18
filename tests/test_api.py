"""End-to-end API tests through the real app, including lifespan.

TestClient runs startup, so these exercise the actual engine load, warmup and
batch manager rather than mocks. Slower, and the only way to catch wiring bugs
between layers.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.config import Settings
from src.main import create_app
from src.model.pytorch_model import WEIGHTS_FILE


def encode(w=300, h=300, mode="RGB", fmt="JPEG") -> bytes:
    arr = np.random.default_rng(0).integers(
        0, 256, (h, w, 3 if mode == "RGB" else 1), dtype=np.uint8
    )
    img = Image.fromarray(arr.squeeze() if mode != "RGB" else arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    s = Settings(_env_file=None, warmup_requests=2, max_batch_size=4, queue_max_size=8)
    if not (s.model_dir / WEIGHTS_FILE).is_file():
        pytest.skip("model not provisioned: scripts/fetch_model.py")
    with TestClient(create_app(s)) as c:
        yield c


# --- health and readiness -------------------------------------------------


def test_health_is_independent_of_the_model(client):
    # Liveness must not depend on the GPU, or a slow load gets the container
    # killed mid-startup and looks like a crash loop.
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_what_is_actually_serving(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["model_name"] == "resnet18"
    assert body["backend"] in {"pytorch", "onnxruntime", "tensorrt"}
    assert body["math_mode"] in {"fp32", "tf32", "fp16"}


def test_ready_is_503_before_the_model_loads(tmp_path):
    """A missing artifact must not crash the process.

    The container has to stay up long enough for someone to read the reason,
    and readiness has to keep it out of rotation meanwhile.
    """
    s = Settings(_env_file=None, model_repository=tmp_path)
    with TestClient(create_app(s), raise_server_exceptions=False) as c:
        assert c.get("/health").status_code == 200
        r = c.get("/ready")
        assert r.status_code == 503
        assert r.json()["ready"] is False
        assert r.json()["detail"]


# --- prediction -----------------------------------------------------------


def test_predict_returns_a_ranked_prediction(client):
    r = client.post("/predict", files={"file": ("x.jpg", encode(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["predictions"]) == 5
    # Ranked descending.
    confs = [p["confidence"] for p in body["predictions"]]
    assert confs == sorted(confs, reverse=True)


def test_response_identifies_the_engine_that_answered(client):
    """Unauditable otherwise: after a deploy changes accuracy, the first
    question is which engine produced a given prediction."""
    body = client.post("/predict", files={"file": ("x.jpg", encode(), "image/jpeg")}).json()
    for field in ("request_id", "model_name", "model_version", "backend", "precision"):
        assert body[field]


def test_latency_breakdown_is_returned(client):
    """Keyed by stage name, so a pipeline that grows a stage reports it
    without a schema change."""
    latency = client.post("/predict", files={"file": ("x.jpg", encode(), "image/jpeg")}).json()[
        "latency"
    ]
    assert set(latency["stages"]) == {"decode", "infer", "classify"}
    assert set(latency["waits"]) == {"decode", "infer", "classify"}
    assert latency["stages"]["decode"] > 0
    assert latency["queued_ms"] >= 0
    assert latency["total_ms"] >= latency["pipeline_ms"]
    assert latency["pipeline_ms"] >= sum(latency["stages"].values()) * 0.5


def test_request_ids_are_unique(client):
    ids = {
        client.post("/predict", files={"file": ("x.jpg", encode(), "image/jpeg")}).json()[
            "request_id"
        ]
        for _ in range(3)
    }
    assert len(ids) == 3


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_common_formats(client, fmt):
    r = client.post("/predict", files={"file": (f"x.{fmt}", encode(fmt=fmt), "image/*")})
    assert r.status_code == 200


# --- failure handling -----------------------------------------------------


def test_garbage_bytes_are_400_not_500(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request"


def test_empty_upload_is_rejected(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_truncated_image_is_rejected(client):
    data = encode()
    r = client.post("/predict", files={"file": ("x.jpg", data[: len(data) // 3], "image/jpeg")})
    assert r.status_code == 400


def test_oversized_upload_is_413(client):
    # Counted as it arrives rather than trusting Content-Length.
    huge = b"\xff" * (11 * 1024 * 1024)
    r = client.post("/predict", files={"file": ("x.jpg", huge, "image/jpeg")})
    assert r.status_code == 413


def test_missing_file_field_is_422(client):
    r = client.post("/predict", data={"nope": "1"})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"


def test_errors_never_leak_internals(client):
    """No stack traces, no file paths, no exception class names."""
    body = client.post("/predict", files={"file": ("x.jpg", b"junk", "image/jpeg")}).json()
    blob = str(body).lower()
    assert "traceback" not in blob
    assert "/home/" not in blob
    assert set(body) <= {"error", "detail", "request_id"}


# --- metrics --------------------------------------------------------------


def test_metrics_exposes_the_pipeline_stages(client):
    client.post("/predict", files={"file": ("x.jpg", encode(), "image/jpeg")})
    text = client.get("/metrics").text
    for metric in (
        "inference_requests_total",
        "inference_latency_seconds",
        "inference_model_loaded",
        "inference_compute_seconds",
        # One instrument per kind of measurement, labelled by stage, so a new
        # stage is observable without touching the metrics module.
        "pipeline_stage_work_seconds",
        "pipeline_stage_wait_seconds",
        "pipeline_stage_batch_size",
        "pipeline_stage_items_total",
    ):
        assert metric in text, f"{metric} missing from /metrics"


def test_every_stage_is_labelled_in_the_metrics(client):
    """The point of the labelled instruments: `sum by (stage)` must be able to
    tell you which stage is the bottleneck, for any pipeline."""
    client.post("/predict", files={"file": ("x.jpg", encode(), "image/jpeg")})
    text = client.get("/metrics").text
    for stage in ("decode", "infer", "classify"):
        assert f'stage="{stage}"' in text, f"stage {stage} missing from /metrics"


def test_metrics_counts_errors_by_reason(client):
    client.post("/predict", files={"file": ("x.jpg", b"junk", "image/jpeg")})
    assert 'reason="bad_image"' in client.get("/metrics").text


# --- concurrency ----------------------------------------------------------


def test_concurrent_requests_each_get_their_own_answer(client):
    """Exercises the batch path: these should be grouped and demultiplexed."""
    from concurrent.futures import ThreadPoolExecutor

    payloads = [encode(w=300 + i, h=300) for i in range(8)]

    def send(data):
        return client.post("/predict", files={"file": ("x.jpg", data, "image/jpeg")})

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(send, payloads))

    assert all(r.status_code == 200 for r in responses)
    assert len({r.json()["request_id"] for r in responses}) == 8
