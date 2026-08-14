"""PyTorch vs ONNX Runtime: same graph, different runtime (Phase 7).

    uv run python scripts/benchmark_backends.py
    uv run python scripts/benchmark_backends.py --batch-sizes 1,8,16,32 --iterations 300

Writes benchmarks/results/backends.json and .csv.

WHAT IS BEING COMPARED

    Not two models. The *same* ResNet-18 weights, executed two ways:

      pytorch      eager execution, one operator at a time as Python calls it
      onnxruntime  the Phase 6 model.onnx, loaded whole, optimised as a graph,
                   then run as a plan with no Python in the loop

    ORT can see the entire forward pass before running any of it, so it can
    fuse operators, fold constants and plan memory reuse across the whole
    graph. Whether that wins on a 51-node convnet is the question.

WHY VRAM IS READ FROM THE DRIVER

    torch.cuda.memory_allocated() only sees PyTorch's caching allocator. ONNX
    Runtime allocates through its own, so torch reports roughly zero for it.
    The only figure comparable across backends is the driver's own
    total-minus-free, which is what device_used_bytes reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.benchmark import NOISE_FLOOR, BenchmarkResult, compare_logits, write_results
from src.comparison import EngineConfig, collect_logits, result_key, run_interleaved_sweep
from src.config import Backend, Precision, Settings
from src.utils.tensor_info import human_bytes

SAMPLE_IMAGE = Path("data/dog.jpg")
EVAL_SAMPLES = 32

CONFIGS = [
    EngineConfig("pytorch/fp32", Backend.PYTORCH, Precision.FP32),
    EngineConfig("onnxruntime/fp32", Backend.ONNXRUNTIME, Precision.FP32),
]
BASELINE = "pytorch/fp32"


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def build_eval_batch(image_size: int, n: int = EVAL_SAMPLES) -> tuple[np.ndarray, str]:
    """Real image content, so agreement reflects production inputs."""
    from src.preprocessing import ImagePreprocessor

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

    return (
        rng.standard_normal((n, 3, image_size, image_size), dtype=np.float32),
        f"{n} gaussian tensors (no real image at {SAMPLE_IMAGE})",
    )


def report_agreement(logits: dict[str, np.ndarray], source: str) -> None:
    section(f"1. Numerical agreement vs {BASELINE}")
    if BASELINE not in logits:
        print("  baseline unavailable; skipping")
        return
    print(f"  evaluated on {source}\n")
    print(f"  {'backend':<20} {'top-1':>8} {'top-5':>8} {'max|dlogit|':>13} {'max dconf':>11}")
    reference = logits[BASELINE]
    for label, values in logits.items():
        if label == BASELINE:
            continue
        a = compare_logits(reference, values)
        print(
            f"  {label:<20} {a.top1_agreement:>7.1%} {a.top5_agreement:>8.1%} "
            f"{a.max_abs_logit_diff:>13.2e} {a.max_confidence_delta:>11.2e}"
        )
    print(
        "\n  Both runtimes execute the same weights, so disagreement here is entirely\n"
        "  down to kernel choice and operation order -- float addition is not\n"
        "  associative, and two runtimes summing a convolution differently will not\n"
        "  produce bit-identical results."
    )


def report_performance(results: list[BenchmarkResult]) -> None:
    section("2. Latency, throughput and VRAM")
    print(
        f"  {'batch':>5} {'backend':<20} {'p50 ms':>8} {'p95 ms':>8} {'img/s':>8} "
        f"{'torch VRAM':>11} {'device VRAM':>12} {'vs base':>8}"
    )
    by_batch: dict[int, dict[str, BenchmarkResult]] = {}
    for r in results:
        by_batch.setdefault(r.batch_size, {})[result_key(r)] = r

    for batch_size in sorted(by_batch):
        row = by_batch[batch_size]
        base = row.get(BASELINE)
        for label in (c.label for c in CONFIGS):
            r = row.get(label)
            if r is None:
                continue
            ratio = r.throughput_ips / base.throughput_ips if base else float("nan")
            flag = "*" if base and abs(ratio - 1.0) <= NOISE_FLOOR else " "
            print(
                f"  {batch_size:>5} {label:<20} {r.p50_ms:>8.2f} {r.p95_ms:>8.2f} "
                f"{r.throughput_ips:>8.1f} {human_bytes(r.peak_alloc_bytes):>11} "
                f"{human_bytes(r.device_used_bytes):>12} {ratio:>7.2f}x{flag}"
            )
        print()
    print(f"  * within the {NOISE_FLOOR:.0%} noise floor")
    print(
        "\n  'torch VRAM' is PyTorch's allocator and reads ~0 for ONNX Runtime, which\n"
        "  has its own. 'device VRAM' is the driver's total-minus-free and is the only\n"
        "  column comparable across backends -- it also includes the CUDA context and\n"
        "  anything else on the card."
    )


def report_stage_breakdown(results: list[BenchmarkResult], batch_size: int) -> None:
    section(f"3. Where the time goes at batch {batch_size}")
    rows = [r for r in results if r.batch_size == batch_size]
    if not rows:
        return
    print(f"  {'backend':<20} {'h2d ms':>9} {'compute ms':>12} {'d2h ms':>9} {'sum':>9}")
    for r in rows:
        total = r.h2d_ms + r.compute_ms + r.d2h_ms
        print(
            f"  {result_key(r):<20} {r.h2d_ms:>9.3f} {r.compute_ms:>12.3f} "
            f"{r.d2h_ms:>9.3f} {total:>9.3f}"
        )
    print(
        "\n  Both backends copy the batch to the device, run, and copy logits back.\n"
        "  ONNX Runtime uses IOBinding here rather than session.run() precisely so the\n"
        "  three stages are separable; session.run() does both copies inside one opaque\n"
        "  call and would leave h2d and d2h as zeros."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch-sizes", default="1,8,16,32")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args()

    try:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    except ValueError:
        print(f"error: --batch-sizes must be integers, got {args.batch_sizes!r}", file=sys.stderr)
        return 2

    probe = (
        Settings(_env_file=None)
        if not args.device
        else Settings(_env_file=None, device=args.device)
    )
    print(f"\n\033[1mBackend comparison\033[0m  {probe.device}")
    print(f"  {args.iterations} iterations per (batch, backend), {args.warmup} warmup")

    eval_batch, source = build_eval_batch(probe.image_size)
    logits = collect_logits(CONFIGS, eval_batch, device=args.device)
    report_agreement(logits, source)

    section("Sweep order (interleaved, rotated)")
    results = run_interleaved_sweep(
        CONFIGS,
        batch_sizes,
        iterations=args.iterations,
        warmup=args.warmup,
        device=args.device,
    )
    if not results:
        print("error: every configuration failed", file=sys.stderr)
        return 1

    report_performance(results)
    report_stage_breakdown(results, max(batch_sizes))

    paths = write_results(results, args.output_dir, "backends")
    print("\n  wrote " + "  ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
