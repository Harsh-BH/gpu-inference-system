"""Cold start vs warm inference (Phase 13).

    uv run python scripts/cold_start.py
    uv run python scripts/cold_start.py --backends pytorch,tensorrt

Measures, per backend, what the *first* request would cost if warmup did not
exist, and what it costs once warm.

WHAT A COLD ENGINE PAYS

    load()          reading weights, building or deserialising the plan,
                    moving parameters into VRAM
    first predict   lazy CUDA context creation, kernel module loading, cuDNN
                    algorithm selection for this input shape, allocator growth

    None of that is the model's arithmetic. It happens once, and somebody pays
    it. Warmup exists so that somebody is the server at boot rather than
    whichever user happens to arrive first.

WHY EACH BACKEND IS RUN IN ITS OWN PROCESS

    The CUDA context is per process and survives engine teardown, so the second
    backend measured in a shared process would find it already created and
    report a cold start that is not cold. Fresh subprocess per backend is the
    only way this number means anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def measure_one(backend: str, precision: str, warm_iterations: int) -> dict:
    """Run inside a fresh interpreter. Prints one JSON line."""
    import numpy as np

    from src.config import Settings
    from src.inference import create_engine

    settings = Settings(_env_file=None, backend=backend, precision=precision, max_batch_size=8)
    batch = np.zeros((1, 3, settings.image_size, settings.image_size), dtype=np.float32)

    t0 = perf_counter()
    engine = create_engine(settings)
    engine.load()
    load_ms = (perf_counter() - t0) * 1000.0

    # The very first inference. Everything lazy happens here.
    t0 = perf_counter()
    engine.predict(batch)
    first_ms = (perf_counter() - t0) * 1000.0

    # Second call: context exists, but cuDNN may still be settling.
    t0 = perf_counter()
    engine.predict(batch)
    second_ms = (perf_counter() - t0) * 1000.0

    for _ in range(warm_iterations):
        engine.predict(batch)
    warm = []
    for _ in range(50):
        t0 = perf_counter()
        engine.predict(batch)
        warm.append((perf_counter() - t0) * 1000.0)
    engine.unload()

    return {
        "backend": backend,
        "precision": precision,
        "load_ms": load_ms,
        "first_inference_ms": first_ms,
        "second_inference_ms": second_ms,
        "warm_median_ms": statistics.median(warm),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backends", default="pytorch,onnxruntime,tensorrt")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Child mode: one backend, fresh process, JSON on stdout.
    if args.child:
        print("JSON:" + json.dumps(measure_one(args.child, args.precision, args.warmup)))
        return 0

    print("\n\033[1mCold start vs warm\033[0m  (each backend in a fresh process)")
    print(
        f"  {'backend':<14} {'load':>10} {'1st infer':>11} {'2nd infer':>11} "
        f"{'warm p50':>10} {'cold penalty':>13}"
    )

    rows = []
    for backend in (b.strip() for b in args.backends.split(",") if b.strip()):
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--child",
                backend,
                "--precision",
                args.precision,
                "--warmup",
                str(args.warmup),
            ],
            capture_output=True,
            text=True,
        )
        line = next((line for line in result.stdout.splitlines() if line.startswith("JSON:")), None)
        if line is None:
            tail = (result.stderr or result.stdout).strip().splitlines()
            print(f"  {backend:<14} unavailable: {tail[-1][:60] if tail else 'no output'}")
            continue
        row = json.loads(line[5:])
        rows.append(row)
        penalty = row["first_inference_ms"] / max(row["warm_median_ms"], 1e-9)
        print(
            f"  {row['backend']:<14} {row['load_ms']:>9.0f}ms {row['first_inference_ms']:>10.1f}ms "
            f"{row['second_inference_ms']:>10.1f}ms {row['warm_median_ms']:>9.2f}ms "
            f"{penalty:>12.0f}x"
        )

    if rows:
        worst = max(rows, key=lambda r: r["first_inference_ms"] + r["load_ms"])
        print(
            f"\n  Worst cold path: {worst['backend']} at "
            f"{worst['load_ms'] + worst['first_inference_ms']:.0f} ms before it can answer,\n"
            f"  against a warm {worst['warm_median_ms']:.2f} ms. That entire cost lands on the\n"
            "  first request unless the server pays it at boot, which is what\n"
            "  WARMUP_REQUESTS and warmup-before-ready exist to do."
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "cold_start.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\n  wrote {args.output_dir}/cold_start.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
