"""Single-image inference with a full latency breakdown (Phases 1 and 2).

    uv run python scripts/infer.py path/to/image.jpg
    uv run python scripts/infer.py dog.jpg --trace --runs 50
    uv run python scripts/infer.py dog.jpg --no-warmup --runs 1   # cold start

This is the whole request path minus HTTP: preprocess -> tensor -> H2D ->
forward -> D2H -> postprocess. Everything the server does later, it does by
calling these same components.

The numbers it prints are the point. A prediction alone tells you the model
works; the breakdown tells you where a request's time actually goes, which is
the only basis on which any later optimisation can be justified.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter

from src.config import Backend, Precision, Settings
from src.inference import EngineError, create_engine
from src.model.pytorch_model import ModelArtifactError, load_labels
from src.postprocessing import top_k
from src.preprocessing import ImagePreprocessor, PreprocessingError
from src.utils.tensor_info import describe


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", help="path to a JPEG/PNG image")
    ap.add_argument("--backend", choices=[b.value for b in Backend])
    ap.add_argument("--precision", choices=[p.value for p in Precision])
    ap.add_argument("--device", help="cuda:0 or cpu")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--runs",
        type=int,
        default=10,
        help="repeat inference and report the median; one sample is noise",
    )
    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip warmup so run 1 shows what a cold engine actually costs",
    )
    ap.add_argument("--trace", action="store_true", help="print tensor info at each hop")
    return ap.parse_args()


def build_settings(args: argparse.Namespace) -> Settings:
    """CLI flags override .env, which overrides the defaults in config.py."""
    overrides = {
        k: v
        for k, v in (
            ("backend", args.backend),
            ("precision", args.precision),
            ("device", args.device),
        )
        if v is not None
    }
    return Settings(**overrides)


def ms(label: str, value: float, note: str = "") -> None:
    print(f"  {label:<18} {value:>8.3f} ms   {note}")


def main() -> int:
    args = parse_args()
    settings = build_settings(args)

    try:
        labels = load_labels(settings.model_dir)
    except ModelArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    preprocessor = ImagePreprocessor(
        image_size=settings.image_size,
        max_pixels=settings.max_image_pixels,
        # Honoured here so this CLI traces the same path the server takes.
        fast_decode=settings.fast_decode,
    )

    print("\n\033[1mConfiguration\033[0m")
    print(f"  {settings.backend.value} / {settings.precision.value} / {settings.device}")
    print(f"  model {settings.model_name} {settings.model_version} from {settings.model_dir}")

    # Read the file once. The server receives bytes over HTTP, never a path,
    # so disk I/O does not belong in the preprocessing measurement.
    try:
        image_bytes = Path(args.image).read_bytes()
    except OSError as exc:
        print(f"error: cannot read {args.image}: {exc}", file=sys.stderr)
        return 2

    try:
        engine = create_engine(settings)
    except EngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    runs = max(1, args.runs)
    pre_ms: list[float] = []
    post_ms: list[float] = []
    results = []

    try:
        with engine:  # load() on enter, unload() on exit -- no leaked VRAM
            if not args.no_warmup:
                t = perf_counter()
                engine.warmup(settings.warmup_requests)
                elapsed = (perf_counter() - t) * 1000.0
                print(f"  warmed up with {settings.warmup_requests} runs ({elapsed:.0f} ms)")

            # Every stage is measured the same number of times. Timing the GPU
            # over 50 runs while timing preprocessing once would compare a warm
            # median against a cold single sample -- and the headline of this
            # whole report is which of the two is bigger.
            for _ in range(runs):
                t = perf_counter()
                sample = preprocessor.from_bytes(image_bytes)
                batch = preprocessor.stack([sample])
                pre_ms.append((perf_counter() - t) * 1000.0)

                result = engine.predict(batch)
                results.append(result)

                t = perf_counter()
                predictions = top_k(result.logits, labels, k=args.top_k)[0]
                post_ms.append((perf_counter() - t) * 1000.0)

            if args.trace:
                print("\n\033[1mTensor lifecycle\033[0m")
                print(f"  {'after preprocess':<22} {describe(sample)}")
                print(f"  {'after stacking':<22} {describe(batch)}")
                print(f"  {'logits from engine':<22} {describe(result.logits)}")
                print(f"\n  {describe(batch).explain()}")
    except PreprocessingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except EngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    preprocess_ms = statistics.median(pre_ms)
    postprocess_ms = statistics.median(post_ms)

    # --- report ---------------------------------------------------------
    print("\n\033[1mPrediction\033[0m")
    for rank, p in enumerate(predictions, 1):
        bar = "#" * round(p.confidence * 40)
        print(f"  {rank}. {p.label:<28} {p.confidence:>7.2%}  {bar}")

    med = lambda f: statistics.median(f(r.timings) for r in results)  # noqa: E731
    inference_total = med(lambda t: t.total_ms)

    # On CPU there is no PCIe hop and no SM; saying otherwise would be the kind
    # of copied-label lie that makes a benchmark table untrustworthy.
    gpu = settings.is_cuda
    print(f"\n\033[1mLatency\033[0m  (median of {len(results)} run(s))")
    ms("preprocess", preprocess_ms, "CPU: decode, resize, crop, normalise")
    ms(
        "host -> device",
        med(lambda t: t.h2d_ms),
        "batch crosses PCIe into VRAM" if gpu else "no-op on CPU",
    )
    ms(
        "inference",
        med(lambda t: t.compute_ms),
        "CUDA kernels on the SMs" if gpu else "CPU forward pass",
    )
    ms("device -> host", med(lambda t: t.d2h_ms), "logits come back" if gpu else "no-op on CPU")
    ms("postprocess", postprocess_ms, "softmax + top-k on CPU")
    print(f"  {'-' * 58}")
    ms("engine total", inference_total, "")
    ms("end to end", preprocess_ms + inference_total + postprocess_ms, "")

    if len(results) > 1:
        firsts = results[0].timings.total_ms
        print(
            f"\n  run 1 {firsts:.3f} ms vs median {inference_total:.3f} ms"
            f"  ({'cold-start cost visible' if firsts > inference_total * 1.5 else 'warm'})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
