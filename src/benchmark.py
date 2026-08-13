"""Measurement harness. Takes any InferenceEngine and reports what it costs.

WHY THIS IS NOT A METHOD ON InferenceEngine

    The PRD puts benchmark() on the engine interface. It does not belong there.
    Benchmarking is something you do *to* an engine, not something an engine
    does, and three implementations would mean three timing loops that drift
    apart. A benchmark written differently per backend cannot fairly compare
    backends -- which is the entire reason this project has backends.

    One harness, one loop, every engine measured by identical code. Adding a
    backend gets it benchmarked for free.

METHODOLOGY, AND WHY EACH CHOICE MATTERS

    Warmup is per batch size, not once.
        cuDNN picks a convolution algorithm per input shape and caches it. A
        new batch size is a new shape, so the first call at each size pays
        autotuning. Warming up only once would charge batch 32's setup cost to
        batch 32's first measured iteration and make it look slow.

    empty_cache() between sizes.
        So each size's peak VRAM is its own, not a high-water mark inherited
        from the previous, larger run.

    Latency is wall-clock around predict(), not the sum of stage timings.
        Stage timings omit validation, the numpy->torch wrap, and Python
        overhead. Those are real costs a client pays. Stage medians are
        reported alongside, for attribution rather than for the headline.

    Throughput is total images / total wall time.
        Not batch_size / mean_latency, which quietly discards the gaps between
        iterations and overstates what the engine sustains.

    p99 needs samples.
        With 100 iterations, "p99" is the second-worst sample and moves wildly
        between runs. Anything below 500 iterations gets flagged in the output
        rather than silently reported as if it meant something.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from src.gpu.memory import empty_cache, get_peak_memory, reset_peak_memory
from src.gpu.telemetry import sample_telemetry
from src.inference.base import InferenceEngine

# Below this many iterations, p99 is an artefact of the sample size.
P99_MIN_ITERATIONS = 500

# Throughput differences smaller than this are not resolvable on a
# thermally-limited device and must not be reported as findings.
#
# Not a guess. Batch 16 and batch 32 measured 681.4 and 666.0 img/s inside one
# sequential sweep -- an apparent 2.3% regression. Re-run in isolation they
# measured 662.9 and 665.2, i.e. identical, and batch 16 itself moved 2.8%
# between the two runs. The cause is visible in the telemetry: this card is
# pinned at its 60 W cap with SM clocks swinging 1627-1725 MHz, so whichever
# batch size runs last in a long sweep is measured on a hotter, slower GPU.
NOISE_FLOOR = 0.05


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One (engine, batch size) measurement. Flat on purpose, for CSV."""

    # what was measured -- a latency number without this context is not a result
    backend: str
    model_name: str
    model_version: str
    precision: str
    device: str
    batch_size: int
    iterations: int

    # per-batch latency, milliseconds
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float

    # median stage breakdown, for attribution
    h2d_ms: float
    compute_ms: float
    d2h_ms: float

    # derived
    throughput_ips: float  # images per second, total images / total wall time
    latency_per_image_ms: float

    # memory
    peak_alloc_bytes: int

    # physical state at the end of the measured run. None when NVML is absent.
    # Present because a throughput difference between two rows is only real if
    # the GPU was in a comparable state for both -- see NOISE_FLOOR below.
    sm_clock_mhz: int | None
    power_w: float | None
    temperature_c: int | None

    @property
    def p99_is_meaningful(self) -> bool:
        return self.iterations >= P99_MIN_ITERATIONS


def _percentiles(samples: np.ndarray) -> dict[str, float]:
    p50, p90, p95, p99 = np.percentile(samples, [50, 90, 95, 99])
    return {
        "mean_ms": float(samples.mean()),
        "p50_ms": float(p50),
        "p90_ms": float(p90),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "min_ms": float(samples.min()),
        "max_ms": float(samples.max()),
        "stddev_ms": float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
    }


def benchmark_engine(
    engine: InferenceEngine,
    batch_size: int,
    *,
    iterations: int = 200,
    warmup: int = 20,
) -> BenchmarkResult:
    """Measure one engine at one batch size.

    The engine must already be loaded. Input is zeros: this measures the
    execution path, and convolution cost does not depend on pixel values.
    """
    meta = engine.metadata
    c, h, w = meta.input_shape
    batch = np.zeros((batch_size, c, h, w), dtype=np.float32)

    # Fresh pool, then re-warm at *this* shape so cuDNN's algorithm choice for
    # this batch size is made before the clock starts.
    empty_cache(meta.device)
    for _ in range(warmup):
        engine.predict(batch)

    reset_peak_memory(meta.device)
    latencies = np.empty(iterations, dtype=np.float64)
    stages = np.empty((iterations, 3), dtype=np.float64)

    wall_start = perf_counter()
    for i in range(iterations):
        t0 = perf_counter()
        result = engine.predict(batch)
        latencies[i] = (perf_counter() - t0) * 1000.0
        t = result.timings
        stages[i] = (t.h2d_ms, t.compute_ms, t.d2h_ms)
    wall_total = perf_counter() - wall_start
    # Sampled here, outside the timed window, while the GPU is still in the
    # thermal state it held during the run. Clocks do not recover in the
    # microseconds it takes to get here.
    telemetry = sample_telemetry()

    stage_medians = np.median(stages, axis=0)
    return BenchmarkResult(
        backend=meta.backend,
        model_name=meta.model_name,
        model_version=meta.model_version,
        precision=meta.precision,
        device=meta.device,
        batch_size=batch_size,
        iterations=iterations,
        **_percentiles(latencies),
        h2d_ms=float(stage_medians[0]),
        compute_ms=float(stage_medians[1]),
        d2h_ms=float(stage_medians[2]),
        # Total work over total time -- includes the gaps between iterations,
        # which batch_size/mean_latency would silently drop.
        throughput_ips=(iterations * batch_size) / wall_total,
        latency_per_image_ms=float(np.median(latencies)) / batch_size,
        peak_alloc_bytes=get_peak_memory(meta.device),
        sm_clock_mhz=telemetry.sm_clock_mhz if telemetry else None,
        power_w=telemetry.power_w if telemetry else None,
        temperature_c=telemetry.temperature_c if telemetry else None,
    )


def sweep_batch_sizes(
    engine: InferenceEngine,
    batch_sizes: list[int],
    *,
    iterations: int = 200,
    warmup: int = 20,
) -> list[BenchmarkResult]:
    """Run benchmark_engine across batch sizes, skipping any that OOM.

    An OOM at batch 64 must not discard the batch 1-32 rows already collected;
    on a 6 GB card, finding the ceiling *is* part of the result.
    """
    from src.inference.base import EngineError

    results = []
    for bs in batch_sizes:
        try:
            results.append(benchmark_engine(engine, bs, iterations=iterations, warmup=warmup))
        except EngineError as exc:
            print(f"  batch {bs}: skipped ({exc})")
            empty_cache(engine.metadata.device)
    return results


# --- persistence ----------------------------------------------------------


def environment() -> dict[str, str]:
    """Captured alongside results because a benchmark without its machine is
    a number without units."""
    import torch

    env = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env |= {
            "gpu": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "vram_bytes": str(props.total_memory),
            "driver_cuda": torch.version.cuda or "",
            "cudnn_allow_tf32": str(torch.backends.cudnn.allow_tf32),
        }
    return env


def write_results(results: list[BenchmarkResult], out_dir: Path, stem: str) -> list[Path]:
    """Write JSON (with environment) and CSV (for spreadsheets). Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps({"environment": environment(), "results": rows}, indent=2) + "\n"
    )

    csv_path = out_dir / f"{stem}.csv"
    if rows:
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return [json_path, csv_path]
