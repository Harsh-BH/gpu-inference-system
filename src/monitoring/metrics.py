"""Prometheus metrics (Phase 16, generalised to the pipeline in Phase 18).

WHY HISTOGRAMS AND NOT AVERAGES

    Average latency is the least useful number a serving system produces. If
    95 requests take 5 ms and 5 take 500 ms, the average is 30 ms -- a value no
    request experienced, describing an outage nobody would notice from the
    number. Users experience the tail, so p50/p95/p99 is what gets reported,
    and that means histograms with explicit buckets.

    Bucket edges matter: a histogram can only resolve latency where it has
    boundaries. These are chosen around what this system actually does --
    sub-millisecond GPU stages, single-digit-millisecond inference, tens of
    milliseconds for decoding and end-to-end.

WHY THE STAGE METRICS ARE LABELLED RATHER THAN NAMED

    They used to be `inference_preprocess_seconds`, `inference_postprocess_
    seconds`, `inference_queue_wait_seconds` -- one instrument per stage, hard
    coded. Adding a stage meant adding an instrument, a call site and a
    dashboard panel, so in practice stages went unmeasured.

    Now there is one instrument per *kind* of measurement, labelled by stage.
    Any pipeline gets complete per-stage observability with no code here at
    all, which is what makes `src/pipeline/` reusable rather than merely
    generic. `sum by (stage) (rate(...))` answers "which stage is the
    bottleneck" for a pipeline this module has never heard of.

WHY WAIT AND WORK ARE SEPARATE INSTRUMENTS

    They diagnose opposite problems. Rising *work* means the stage itself got
    slower -- a bigger image, a colder cache, a throttled GPU. Rising *wait*
    with flat work means the stage is saturated and the fix is capacity or
    concurrency, not a faster implementation. Phase 1 of this project had those
    two diagnoses pointing in opposite directions on the same workload, which
    is why they are never summed into one number.
"""

from __future__ import annotations

from functools import lru_cache

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from src.inference.base import StageTimings
from src.pipeline import StageReport

# Latency buckets in seconds. Deliberately dense between 1 ms and 100 ms, which
# is where every stage of this system lives.
_FAST = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
_SLOW = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class Metrics:
    """All instruments in one object.

    Constructed with an explicit registry rather than using the global default,
    so tests can build a fresh Metrics per case. The default registry is
    process-global and raises on duplicate registration, which makes any test
    that builds the app twice fail for reasons unrelated to what it is testing.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry

        # --- requests ---
        self.requests_total = Counter(
            "inference_requests_total", "Requests received", ["status"], registry=r
        )
        self.errors_total = Counter(
            "inference_errors_total", "Requests that failed", ["reason"], registry=r
        )
        self.total_latency = Histogram(
            "inference_latency_seconds",
            "End-to-end, as the client experiences it",
            buckets=_SLOW,
            registry=r,
        )

        # --- pipeline, per stage ---
        self.stage_work = Histogram(
            "pipeline_stage_work_seconds",
            "Time inside a stage's process(), per batch",
            ["stage"],
            buckets=_SLOW,
            registry=r,
        )
        self.stage_wait = Histogram(
            "pipeline_stage_wait_seconds",
            "Time an item spent queued before a stage ran it",
            ["stage"],
            buckets=_SLOW,
            registry=r,
        )
        self.stage_batch_size = Histogram(
            "pipeline_stage_batch_size",
            "Items per process() call",
            ["stage"],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
            registry=r,
        )
        self.stage_items = Counter(
            "pipeline_stage_items_total",
            "Items leaving a stage",
            ["stage", "outcome"],
            registry=r,
        )
        self.stage_depth = Gauge(
            "pipeline_stage_queue_depth",
            "Items waiting in front of a stage",
            ["stage"],
            registry=r,
        )

        # --- inside the engine ---
        # Not stage metrics: only an engine can see these, and only for the one
        # stage that runs a model. A caller holding a numpy array cannot
        # observe how long the PCIe copy took.
        self.h2d = Histogram(
            "inference_h2d_seconds", "Host to device copy", buckets=_FAST, registry=r
        )
        self.compute = Histogram(
            "inference_compute_seconds", "Forward pass", buckets=_FAST, registry=r
        )
        self.d2h = Histogram(
            "inference_d2h_seconds", "Device to host copy", buckets=_FAST, registry=r
        )

        # --- GPU ---
        self.gpu_memory_used = Gauge(
            "gpu_memory_used_bytes", "Device memory in use (driver view)", registry=r
        )
        self.gpu_memory_peak = Gauge(
            "gpu_memory_peak_bytes", "Peak allocated by this process", registry=r
        )
        self.gpu_utilization = Gauge("gpu_utilization_percent", "GPU utilisation", registry=r)
        self.gpu_power = Gauge("gpu_power_watts", "GPU power draw", registry=r)
        self.gpu_clock = Gauge("gpu_sm_clock_mhz", "SM clock", registry=r)

        # --- readiness ---
        self.model_loaded = Gauge(
            "inference_model_loaded", "1 when the engine is ready to serve", registry=r
        )

    # --- observers -------------------------------------------------------

    def record_stage(self, report: StageReport) -> None:
        """Wired to `Pipeline(on_stage=...)`. Called once per process() call.

        Cheap on purpose: this runs on the event loop after every batch of
        every stage, so it does arithmetic and nothing else. Anything that
        could block or fail belongs behind a scrape, not here.
        """
        stage = report.stage
        self.stage_work.labels(stage=stage).observe(report.work_ms / 1000.0)
        self.stage_wait.labels(stage=stage).observe(report.wait_ms / 1000.0)
        self.stage_batch_size.labels(stage=stage).observe(report.batch_size)
        if report.succeeded:
            self.stage_items.labels(stage=stage, outcome="ok").inc(report.succeeded)
        if report.failed:
            self.stage_items.labels(stage=stage, outcome="error").inc(report.failed)

    def record_engine_timings(self, timings: StageTimings, batch_size: int) -> None:
        """Wired to `InferenceStage(on_timings=...)`."""
        self.h2d.observe(timings.h2d_ms / 1000.0)
        self.compute.observe(timings.compute_ms / 1000.0)
        self.d2h.observe(timings.d2h_ms / 1000.0)

    def observe_depths(self, depths: dict[str, int]) -> None:
        """Sample queue depths. Called on scrape, not per request."""
        for stage, depth in depths.items():
            self.stage_depth.labels(stage=stage).set(depth)

    def observe_gpu(self, device: str) -> None:
        """Sample GPU state. Called on /metrics scrape, never per request.

        NVML costs 0.1-1 ms per read, which is fine at Prometheus scrape
        intervals and absurd on a request path serving 3000 img/s.
        """
        from src.gpu.memory import get_gpu_memory
        from src.gpu.telemetry import sample_telemetry

        snapshot = get_gpu_memory(device)
        if snapshot.available:
            self.gpu_memory_used.set(snapshot.used_on_device)
            self.gpu_memory_peak.set(snapshot.peak_allocated)

        telemetry = sample_telemetry()
        if telemetry is not None:
            self.gpu_utilization.set(telemetry.utilization_pct)
            self.gpu_power.set(telemetry.power_w)
            self.gpu_clock.set(telemetry.sm_clock_mhz)


@lru_cache(maxsize=1)
def get_metrics() -> Metrics:
    """Process-wide metrics. Cached, because Prometheus collectors may only be
    registered once per registry."""
    return Metrics()
