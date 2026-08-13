"""FP32 vs TF32 vs FP16: what lower precision buys, and what it costs (Phase 5).

    uv run python scripts/benchmark_precision.py
    uv run python scripts/benchmark_precision.py --batch-sizes 1,8,16,32 --iterations 300

Writes benchmarks/results/precision_<backend>.json and .csv.

THREE MODES, NOT TWO

    fp32   weights float32, convolutions accumulate in float32. The reference.
    tf32   weights float32, convolutions accumulate on Tensor Cores with 10
           mantissa bits instead of 23. Storage is unchanged -- this is a
           *compute* precision, and it is what PyTorch does by default on
           Ampere unless told otherwise.
    fp16   weights and activations float16. Half the storage, half the
           bandwidth, Tensor Core eligible.

METHODOLOGY

    Phase 4 established that this card is power-capped and that a long
    sequential sweep measures each successive configuration on a hotter, slower
    GPU. Comparing precisions is exactly where that would do the most damage --
    a 10% "FP16 speedup" is worthless if FP16 simply ran first.

    So the sweep is interleaved: for each batch size all three precisions are
    measured back-to-back, and the order within each group rotates. Every
    precision therefore runs first, middle and last across the sweep, and any
    residual drift is spread evenly instead of landing on one mode.

    Agreement is measured on real image content, not noise. Convolution
    activations depend on spatial structure, and FP16's error behaviour depends
    on activation magnitude, so random tensors would understate or overstate
    drift in ways that do not transfer to production traffic.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.benchmark import (
    NOISE_FLOOR,
    BenchmarkResult,
    benchmark_engine,
    compare_logits,
    write_results,
)
from src.config import Backend, Precision, Settings
from src.gpu.memory import memory_scope
from src.inference import EngineError, create_engine
from src.preprocessing import ImagePreprocessor
from src.utils.tensor_info import human_bytes

SAMPLE_IMAGE = Path("data/dog.jpg")
EVAL_SAMPLES = 32


@dataclass(frozen=True, slots=True)
class Mode:
    label: str
    precision: Precision
    allow_tf32: bool


MODES = [
    Mode("fp32", Precision.FP32, False),
    Mode("tf32", Precision.FP32, True),
    Mode("fp16", Precision.FP16, False),
]


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def build_settings(args, mode: Mode, max_batch: int) -> Settings:
    overrides = {
        "backend": args.backend,
        "precision": mode.precision,
        "allow_tf32": mode.allow_tf32,
        "max_batch_size": max_batch,
    }
    if args.device:
        overrides["device"] = args.device
    return Settings(_env_file=None, **overrides)


def build_eval_batch(image_size: int, n: int = EVAL_SAMPLES) -> tuple[np.ndarray, str]:
    """N distinct samples with real image statistics.

    Random crops at varying scale from the sample photo: genuinely different
    inputs that still carry real texture and edge structure. Falls back to
    gaussian noise, clearly labelled, because a synthetic result presented as a
    real one is worse than no result.
    """
    pre = ImagePreprocessor(image_size=image_size)
    rng = np.random.default_rng(0)

    if SAMPLE_IMAGE.is_file():
        from PIL import Image, ImageOps

        img = ImageOps.exif_transpose(Image.open(SAMPLE_IMAGE)).convert("RGB")
        w, h = img.size
        samples = []
        for _ in range(n):
            scale = float(rng.uniform(0.35, 1.0))
            cw, ch = max(int(w * scale), image_size), max(int(h * scale), image_size)
            x = int(rng.integers(0, w - cw + 1))
            y = int(rng.integers(0, h - ch + 1))
            samples.append(pre._transform(img.crop((x, y, x + cw, y + ch))))
        return np.stack(samples), f"{n} random crops of {SAMPLE_IMAGE}"

    noisy = rng.standard_normal((n, 3, image_size, image_size), dtype=np.float32)
    return noisy, f"{n} gaussian tensors (NO real image found at {SAMPLE_IMAGE})"


def measure_weights_and_logits(
    args, eval_batch: np.ndarray
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Weights VRAM and reference logits for each mode, one engine at a time."""
    weights: dict[str, int] = {}
    logits: dict[str, np.ndarray] = {}

    for mode in MODES:
        settings = build_settings(args, mode, eval_batch.shape[0])
        engine = create_engine(settings)
        with memory_scope(f"load {mode.label}", settings.device) as scope:
            engine.load()
        try:
            weights[mode.label] = scope.allocated_delta
            engine.warmup(5)
            logits[mode.label] = engine.predict(eval_batch).logits
        finally:
            engine.unload()
    return weights, logits


def report_weights(weights: dict[str, int]) -> None:
    section("1. Weights in VRAM")
    base = weights.get("fp32", 1)
    for mode in MODES:
        w = weights[mode.label]
        note = ""
        if mode.label == "tf32":
            note = "  same storage as fp32 -- TF32 changes the maths, not the bytes"
        elif mode.label == "fp16":
            note = "  2 bytes per parameter instead of 4"
        print(f"  {mode.label:<6} {human_bytes(w):>12}   {w / base:.2f}x{note}")
    print(
        "\n  11,689,512 parameters. The storage argument for FP16 is exactly this and\n"
        "  nothing more: halve the bytes per weight, halve the bytes moved from VRAM\n"
        "  to the SMs on every layer. Whether that becomes *speed* depends on whether\n"
        "  the workload is bandwidth-bound, which is what section 3 measures."
    )


def report_agreement(logits: dict[str, np.ndarray], source: str) -> None:
    section("2. Numerical agreement vs FP32")
    print(f"  evaluated on {source}\n")
    print(
        f"  {'mode':<6} {'top-1':>8} {'top-5':>8} {'max|dlogit|':>13} "
        f"{'mean|dlogit|':>13} {'max dconf':>11}"
    )
    reference = logits["fp32"]
    for mode in MODES:
        if mode.label == "fp32":
            continue
        a = compare_logits(reference, logits[mode.label])
        print(
            f"  {mode.label:<6} {a.top1_agreement:>7.1%} {a.top5_agreement:>8.1%} "
            f"{a.max_abs_logit_diff:>13.2e} {a.mean_abs_logit_diff:>13.2e} "
            f"{a.max_confidence_delta:>11.2e}"
        )


def run_sweep(args, batch_sizes: list[int]) -> list[BenchmarkResult]:
    """Interleaved by batch size, precision order rotated per batch size.

    Grouping by batch size keeps the three precisions within seconds of each
    other thermally; rotating the order stops the same mode always paying the
    'measured last, on a hotter GPU' penalty.
    """
    results: list[BenchmarkResult] = []
    max_batch = max(batch_sizes)

    for i, bs in enumerate(batch_sizes):
        order = MODES[i % len(MODES) :] + MODES[: i % len(MODES)]
        print(f"  batch {bs}: {' -> '.join(m.label for m in order)}")
        for mode in order:
            settings = build_settings(args, mode, max_batch)
            engine = create_engine(settings)
            try:
                engine.load()
                results.append(
                    benchmark_engine(engine, bs, iterations=args.iterations, warmup=args.warmup)
                )
            except EngineError as exc:
                print(f"    {mode.label} batch {bs}: skipped ({exc})")
            finally:
                engine.unload()
    return results


def report_performance(results: list[BenchmarkResult]) -> None:
    section("4. Latency and throughput")
    print(
        f"  {'batch':>5} {'mode':<6} {'p50 ms':>8} {'p95 ms':>8} {'img/s':>8} "
        f"{'peak VRAM':>11} {'vs fp32':>8}  {'clock':>9}"
    )
    by_batch: dict[int, dict[str, BenchmarkResult]] = {}
    for r in results:
        by_batch.setdefault(r.batch_size, {})[r.math_mode] = r

    for bs in sorted(by_batch):
        row = by_batch[bs]
        base = row.get("fp32")
        for mode in MODES:
            r = row.get(mode.label)
            if r is None:
                continue
            speedup = r.throughput_ips / base.throughput_ips if base else float("nan")
            flag = "*" if base and abs(speedup - 1.0) <= NOISE_FLOOR else " "
            clock = f"{r.sm_clock_mhz} MHz" if r.sm_clock_mhz else "-"
            print(
                f"  {bs:>5} {mode.label:<6} {r.p50_ms:>8.2f} {r.p95_ms:>8.2f} "
                f"{r.throughput_ips:>8.1f} {human_bytes(r.peak_alloc_bytes):>11} "
                f"{speedup:>7.2f}x{flag} {clock:>9}"
            )
        print()
    print(f"  * difference from fp32 is within the {NOISE_FLOOR:.0%} noise floor")


def report_kernels(args, batch_size: int, top: int = 4) -> None:
    """Which CUDA kernels actually ran. Evidence, not inference.

    "FP16 is faster, so presumably Tensor Cores" is a guess. The kernel names
    say it outright, and they also explain the size of each gain:

      scudnn / winograd            CUDA cores, no Tensor Core involvement
      sm86_xmma_..._tf32f32_...    Tensor Core, TF32 fragments
      cutlass_tensorop_f16_s16816  Tensor Core, FP16 16x8x16 MMA fragments
      nchwToNhwcKernel             pure layout conversion, no maths at all
    """
    from torch.profiler import ProfilerActivity, profile

    section("5. Which kernels actually ran")
    print(f"  batch {batch_size}, 5 iterations, CUDA activity only\n")

    for mode in MODES:
        settings = build_settings(args, mode, batch_size)
        engine = create_engine(settings)
        try:
            engine.load()
            engine.warmup(10)
            data = np.zeros((batch_size, *engine.metadata.input_shape), dtype=np.float32)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    engine.predict(data)
        finally:
            engine.unload()

        events = sorted(
            (e for e in prof.key_averages() if e.device_time_total > 0),
            key=lambda e: -e.device_time_total,
        )
        total = sum(e.device_time_total for e in events)
        print(f"  \033[1m{mode.label}\033[0m  total GPU {total / 1000:.1f} ms")
        for e in events[:top]:
            print(f"    {e.device_time_total / total:5.1%}  {e.key[:70]}")
        print()


def report_verdict(results: list[BenchmarkResult]) -> None:
    section("6. Verdict")
    by_mode: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_mode.setdefault(r.math_mode, []).append(r)

    base_peak = max(by_mode.get("fp32", []), key=lambda r: r.throughput_ips, default=None)
    if base_peak is None:
        return

    for mode in MODES[1:]:
        rows = by_mode.get(mode.label)
        if not rows:
            continue
        best = max(rows, key=lambda r: r.throughput_ips)
        gain = best.throughput_ips / base_peak.throughput_ips
        verdict = (
            f"{gain:.2f}x throughput" if abs(gain - 1.0) > NOISE_FLOOR else "no measurable gain"
        )
        print(
            f"  {mode.label:<6} best {best.throughput_ips:>6.1f} img/s at batch "
            f"{best.batch_size:<3} vs fp32 {base_peak.throughput_ips:.1f} "
            f"at batch {base_peak.batch_size}   -> {verdict}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=[b.value for b in Backend], default="pytorch")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-sizes", default="1,8,16,32")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    ap.add_argument(
        "--profile-kernels",
        action="store_true",
        help="also dump the hottest CUDA kernels per mode (proves Tensor Core use)",
    )
    args = ap.parse_args()

    try:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    except ValueError:
        print(f"error: --batch-sizes must be integers, got {args.batch_sizes!r}", file=sys.stderr)
        return 2

    probe = build_settings(args, MODES[0], max(batch_sizes))
    if not probe.is_cuda:
        print("error: precision comparison requires a CUDA device", file=sys.stderr)
        return 1

    print(f"\n\033[1mPrecision comparison\033[0m  {probe.backend.value} / {probe.device}")
    print(f"  {args.iterations} iterations per (batch, mode), {args.warmup} warmup")

    eval_batch, source = build_eval_batch(probe.image_size)

    try:
        weights, logits = measure_weights_and_logits(args, eval_batch)
        report_weights(weights)
        report_agreement(logits, source)

        section("3. Sweep order (interleaved, rotated)")
        results = run_sweep(args, batch_sizes)
    except EngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("error: every configuration failed", file=sys.stderr)
        return 1

    report_performance(results)
    if args.profile_kernels:
        report_kernels(args, batch_size=max(batch_sizes))
    report_verdict(results)

    paths = write_results(results, args.output_dir, f"precision_{probe.backend.value}")
    print("\n  wrote " + "  ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
