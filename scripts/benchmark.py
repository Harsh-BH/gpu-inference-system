"""Batch size sweep: what batching actually buys, and what it costs (Phase 4).

    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --batch-sizes 1,2,4,8,16,32,64 --iterations 500
    uv run python scripts/benchmark.py --precision fp16

Writes benchmarks/results/<stem>.json and .csv.

The claim under test is "bigger batch is better". It is half true, and the half
that is false is the half that matters for a latency SLO.
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from src.benchmark import NOISE_FLOOR, BenchmarkResult, sweep_batch_sizes, write_results
from src.config import Backend, Precision, Settings
from src.inference import EngineError, create_engine
from src.preprocessing import ImagePreprocessor
from src.utils.tensor_info import human_bytes

DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]
SAMPLE_IMAGE = Path("data/dog.jpg")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def measure_preprocessing(settings: Settings, runs: int = 30) -> tuple[float, str]:
    """Median milliseconds to turn one encoded image into a tensor.

    Needed to put engine throughput in context: a GPU that can serve 900 img/s
    is not a system that can serve 900 img/s if one CPU thread can only produce
    56 tensors per second.

    Uses the real sample photo when present. JPEG decode cost depends on image
    content, so a synthetic fallback is labelled as such rather than quietly
    substituted.
    """
    pre = ImagePreprocessor(image_size=settings.image_size)
    if SAMPLE_IMAGE.is_file():
        data, source = SAMPLE_IMAGE.read_bytes(), f"{SAMPLE_IMAGE}"
    else:
        from PIL import Image

        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 256, (1213, 1546, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        data, source = buf.getvalue(), "synthetic 1546x1213 JPEG (noise decodes unlike a photo)"

    times = []
    for _ in range(runs):
        t = perf_counter()
        pre.from_bytes(data)
        times.append((perf_counter() - t) * 1000.0)
    return statistics.median(times), source


def print_table(results: list[BenchmarkResult]) -> None:
    section("Batch size sweep")
    print(
        f"  {'batch':>5} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>9} "
        f"{'img/s':>7} {'ms/img':>7} {'peak VRAM':>10}  {'GPU at end of run':<20}"
    )
    for r in results:
        p99 = f"{r.p99_ms:.2f}" if r.p99_is_meaningful else f"{r.p99_ms:.2f}*"
        gpu = (
            f"{r.sm_clock_mhz} MHz  {r.power_w:.0f} W  {r.temperature_c} C"
            if r.sm_clock_mhz is not None
            else "-"
        )
        print(
            f"  {r.batch_size:>5} {r.p50_ms:>8.2f} {r.p95_ms:>8.2f} {p99:>9} "
            f"{r.throughput_ips:>7.1f} {r.latency_per_image_ms:>7.3f} "
            f"{human_bytes(r.peak_alloc_bytes):>10}  {gpu:<20}"
        )
    if results and not results[0].p99_is_meaningful:
        print(
            f"\n  * p99 from {results[0].iterations} iterations is roughly the "
            "worst few samples;\n    rerun with --iterations 500 for a stable tail."
        )


def print_marginal_analysis(results: list[BenchmarkResult]) -> None:
    """What each step actually bought, and what it charged for it."""
    if len(results) < 2:
        return
    section("What each step bought")
    print(f"  {'step':>12}  {'throughput':>12}  {'p50 latency':>13}  {'peak VRAM':>11}")
    for prev, cur in zip(results, results[1:], strict=False):
        tput = cur.throughput_ips / prev.throughput_ips
        lat = cur.p50_ms / prev.p50_ms
        mem = cur.peak_alloc_bytes / max(prev.peak_alloc_bytes, 1)
        flag = "" if abs(tput - 1.0) > NOISE_FLOOR else "   (within noise)"
        print(
            f"  {prev.batch_size:>4} -> {cur.batch_size:<4}  "
            f"{tput:>11.2f}x  {lat:>12.2f}x  {mem:>10.2f}x{flag}"
        )

    best = max(results, key=lambda r: r.throughput_ips)
    # Where the plateau starts: the first batch size statistically tied with the
    # best. Reporting the argmax alone would turn measurement noise into a
    # recommendation -- exactly the mistake this project got caught making.
    plateau = next(
        r for r in results if r.throughput_ips >= best.throughput_ips * (1 - NOISE_FLOOR)
    )
    print(
        f"\n  best measured   batch {best.batch_size:<4} "
        f"{best.throughput_ips:.0f} img/s, p50 {best.p50_ms:.2f} ms"
    )
    print(
        f"  plateau from    batch {plateau.batch_size:<4} "
        f"{plateau.throughput_ips:.0f} img/s, p50 {plateau.p50_ms:.2f} ms"
    )
    if plateau.batch_size != results[-1].batch_size:
        last = results[-1]
        print(
            f"\n  Past batch {plateau.batch_size} the GPU is saturated: batch {last.batch_size} "
            f"costs {last.p50_ms / plateau.p50_ms:.1f}x the latency and "
            f"{last.peak_alloc_bytes / plateau.peak_alloc_bytes:.1f}x the VRAM\n"
            f"  for {last.throughput_ips / plateau.throughput_ips:.2f}x the throughput. "
            "That is the answer to 'is bigger batch better':\n"
            "  better until the accelerator is full, pure cost afterwards."
        )
    first, last = results[0], results[-1]
    if first.sm_clock_mhz and last.sm_clock_mhz and last.sm_clock_mhz < first.sm_clock_mhz:
        drop = 1 - last.sm_clock_mhz / first.sm_clock_mhz
        print(
            f"\n  Note the clock column: {first.sm_clock_mhz} MHz at batch {first.batch_size} "
            f"down to {last.sm_clock_mhz} MHz at batch {last.batch_size}\n"
            f"  ({drop:.0%}), with temperature rising {first.temperature_c} -> "
            f"{last.temperature_c} C. The card is power-capped, so the sweep\n"
            "  measures each successive batch size on a slightly slower GPU. The bias runs\n"
            "  against large batches, meaning the scaling above is if anything understated."
        )
    print(
        f"\n  Differences under {NOISE_FLOOR:.0%} are not resolvable here. Confirm any close\n"
        "  call by re-running just those two sizes in isolation."
    )


def print_system_reality_check(
    results: list[BenchmarkResult], preprocess_ms: float, source: str
) -> None:
    """Engine throughput is not system throughput."""
    if not results:
        return
    section("Reality check: the engine is not the system")
    best = max(results, key=lambda r: r.throughput_ips)
    single_thread_ips = 1000.0 / preprocess_ms
    threads_needed = best.throughput_ips / single_thread_ips

    print(f"  preprocessing            {preprocess_ms:.2f} ms/image  ({source})")
    print(f"  one CPU thread sustains  {single_thread_ips:.1f} img/s")
    print(f"  engine peak              {best.throughput_ips:.1f} img/s at batch {best.batch_size}")
    print(f"  \033[1mratio                    {threads_needed:.1f}x\033[0m")
    print(
        f"\n  Saturating this GPU needs roughly {threads_needed:.0f} concurrent preprocessing\n"
        "  workers. With one, the batch manager will never assemble a full batch under\n"
        "  real load -- it will time out waiting for images that the CPU has not decoded\n"
        "  yet, and the sweep above becomes a description of hardware you cannot reach.\n"
        "\n  This is why Phases 10-12 (queue, dynamic batching, concurrency) come before\n"
        "  TensorRT. Making the 3 ms into 1 ms changes nothing while the 18 ms stands."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=[b.value for b in Backend], default="pytorch")
    ap.add_argument("--precision", choices=[p.value for p in Precision], default="fp32")
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-tf32", action="store_true", help="benchmark with TF32 enabled")
    ap.add_argument(
        "--batch-sizes",
        default=",".join(str(b) for b in DEFAULT_BATCHES),
        help="comma-separated, e.g. 1,2,4,8,16,32",
    )
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args()

    try:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    except ValueError:
        print(f"error: --batch-sizes must be integers, got {args.batch_sizes!r}", file=sys.stderr)
        return 2
    if not batch_sizes:
        print("error: no batch sizes given", file=sys.stderr)
        return 2

    overrides = {
        "backend": args.backend,
        "precision": args.precision,
        "allow_tf32": args.allow_tf32,
        # The engine validates against max_batch_size; the sweep sets its own ceiling.
        "max_batch_size": max(batch_sizes),
    }
    if args.device:
        overrides["device"] = args.device
    settings = Settings(_env_file=None, **overrides)

    print(
        f"\n\033[1mBenchmark\033[0m  {settings.backend.value} / {settings.precision.value} / "
        f"{settings.device} / tf32={settings.allow_tf32}"
    )
    print(f"  {args.iterations} iterations per batch size, {args.warmup} warmup")

    preprocess_ms, source = measure_preprocessing(settings)

    try:
        engine = create_engine(settings)
    except EngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        with engine:
            results = sweep_batch_sizes(
                engine, batch_sizes, iterations=args.iterations, warmup=args.warmup
            )
    except EngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("error: every batch size failed", file=sys.stderr)
        return 1

    print_table(results)
    print_marginal_analysis(results)
    print_system_reality_check(results, preprocess_ms, source)

    stem = f"batch_sweep_{settings.backend.value}_{settings.precision.value}"
    if settings.allow_tf32:
        stem += "_tf32"
    paths = write_results(results, args.output_dir, stem)
    print("\n  wrote " + "  ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
