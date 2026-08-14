"""Prometheus metrics (Phase 16).

WHY HISTOGRAMS AND NOT AVERAGES

    Average latency is the least useful number a serving system produces. If
    95 requests take 5 ms and 5 take 500 ms, the average is 30 ms -- a value no
    request experienced, describing an outage nobody would notice from the
    number. Users experience the tail, so p50/p95/p99 is what gets reported,
    and that means histograms with explicit buckets.

    Bucket edges matter: a histogram can only resolve latency where it has
    boundaries. These are chosen around what this system actually does --
    sub-millisecond GPU stages, single-digit-millisecond inference, tens of
    milliseconds for preprocessing and end-to-end.

WHY EVERY STAGE IS MEASURED SEPARATELY

    "The API is slow" is not actionable. "Queue wait is 400 ms while inference
    is 3 ms" says the GPU is saturated and you need capacity or bigger batches.
    "Preprocess is 20 ms while inference is 3 ms" says buy CPU, not GPU. The
    split is what turns a complaint into a decision, and Phase 1 already showed
    those two diagnoses pointing in opposite directions on this workload.
"""

from __future__ import annotations

from functools import lru_cache

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

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

        # --- queue ---
        self.queue_depth = Gauge(
            "inference_queue_depth", "Requests waiting in the queue", registry=r
        )
        self.queue_wait = Histogram(
            "inference_queue_wait_seconds",
            "Time from enqueue to batch dispatch",
            buckets=_SLOW,
            registry=r,
        )

        # --- batching ---
        self.batch_size = Histogram(
            "inference_batch_size",
            "Images per dispatched batch",
            buckets=(1, 2, 4, 8, 16, 32, 64),
            registry=r,
        )
        self.batch_formation = Histogram(
            "inference_batch_formation_seconds",
            "Time spent assembling a batch",
            buckets=_FAST,
            registry=r,
        )

        # --- pipeline stages ---
        self.preprocess = Histogram(
            "inference_preprocess_seconds", "Image decode to tensor", buckets=_SLOW, registry=r
        )
        self.h2d = Histogram(
            "inference_h2d_seconds", "Host to device copy", buckets=_FAST, registry=r
        )
        self.compute = Histogram(
            "inference_compute_seconds", "Forward pass", buckets=_FAST, registry=r
        )
        self.d2h = Histogram(
            "inference_d2h_seconds", "Device to host copy", buckets=_FAST, registry=r
        )
        self.postprocess = Histogram(
            "inference_postprocess_seconds", "Softmax and top-k", buckets=_FAST, registry=r
        )
        self.total_latency = Histogram(
            "inference_latency_seconds",
            "End-to-end, as the client experiences it",
            buckets=_SLOW,
            registry=r,
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
