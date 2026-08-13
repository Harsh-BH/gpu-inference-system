"""Where the VRAM actually goes, and what happens when it runs out (Phase 3).

    uv run python scripts/gpu_memory_report.py
    uv run python scripts/gpu_memory_report.py --precision fp16
    uv run python scripts/gpu_memory_report.py --fraction 0.5   # OOM sooner

Reports memory at every stage of the engine lifecycle, measures how activation
memory scales with batch size, and then deliberately exhausts VRAM to show that
the process survives it.

SAFETY: this machine drives its display from the same GPU. The experiment caps
our allocator with torch.cuda.set_per_process_memory_fraction() so that OOM
hits our own ceiling and leaves the compositor its memory. Exhausting physical
VRAM on a desktop GPU can hang the display, which is not the lesson.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import numpy as np

from src.config import Precision, Settings
from src.gpu.memory import (
    empty_cache,
    get_gpu_memory,
    limit_process_memory,
    memory_scope,
)
from src.inference import EngineError, create_engine
from src.utils.tensor_info import human_bytes

# Doubling stops here regardless. Beyond this the *host* array alone is over a
# gigabyte, and we would be measuring numpy rather than the GPU.
HARD_BATCH_CAP = 1024


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def smi_memory() -> tuple[int, int] | None:
    """Free/total bytes according to the driver, before we create a context.

    Deliberately not torch: any torch CUDA call initialises a context, which is
    the very thing we are trying to measure the cost of.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        free, total = (int(x) for x in out.stdout.splitlines()[0].split(",")[:2])
        return free * 2**20, total * 2**20
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def report_context_cost() -> None:
    """The VRAM you pay before allocating a single tensor."""
    import torch

    section("1. CUDA context")
    baseline = smi_memory()
    if baseline is None:
        print("  nvidia-smi unavailable; skipping context measurement")
        return
    free_before, total = baseline

    torch.cuda.init()
    # Force the context to fully materialise.
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()

    print(f"  device total                  {human_bytes(total)}")
    print(f"  free before our process       {human_bytes(free_before)}")
    print(f"  free after CUDA context       {human_bytes(free_after)}")
    print(f"  \033[1mCUDA context cost             {human_bytes(free_before - free_after)}\033[0m")
    print(
        "\n  Charged the moment a process first touches CUDA -- driver kernel code,\n"
        "  constant memory and internal buffers. memory_allocated() reports 0 for\n"
        "  all of it. On a 6 GB card this is a real fraction of the budget, which\n"
        "  is why a server should hold ONE context and not fork per request."
    )


def report_lifecycle(settings: Settings, engine) -> None:
    section("2. Engine lifecycle")
    print(f"  {'stage':<26} {'allocated':>21}  {'reserved':>20}  {'transient peak':>16}")

    with memory_scope("model load", settings.device) as load_scope:
        engine.load()
    print(f"  {load_scope}")

    with memory_scope("warmup (10 x batch 1)", settings.device) as warm_scope:
        engine.warmup(10)
    print(f"  {warm_scope}")

    params = 11_689_512  # resnet18
    bytes_per = 2 if settings.precision is Precision.FP16 else 4
    print(
        f"\n  weights: {params:,} params x {bytes_per} B = "
        f"{human_bytes(params * bytes_per)} expected, "
        f"{human_bytes(load_scope.allocated_delta)} measured"
    )
    print(
        "  Warmup allocates almost nothing lasting -- activations are freed as the\n"
        "  forward pass proceeds. What it DOES do is grow the allocator pool, so\n"
        "  the first real request does not pay cudaMalloc."
    )


def report_batch_scaling(settings: Settings, engine, batches: list[int]) -> list[tuple[int, int]]:
    section("3. How batch size moves memory")
    print(f"  {'batch':>6}  {'input':>11}  {'transient peak':>15}  {'peak / image':>13}")

    rows: list[tuple[int, int]] = []
    c, h, w = engine.metadata.input_shape
    for bs in batches:
        data = np.zeros((bs, c, h, w), dtype=np.float32)
        try:
            with memory_scope(f"batch {bs}", settings.device) as scope:
                engine.predict(data)
        except EngineError as exc:
            print(f"  {bs:>6}  {'-':>11}  failed: {exc}")
            empty_cache(settings.device)
            continue
        peak = scope.peak_above_entry
        rows.append((bs, peak))
        print(
            f"  {bs:>6}  {human_bytes(data.nbytes):>11}  "
            f"{human_bytes(peak):>15}  {human_bytes(peak / bs):>13}"
        )

    if len(rows) >= 2:
        (b0, p0), (b1, p1) = rows[0], rows[-1]
        print(
            f"\n  {b0} -> {b1} images is {b1 // b0}x the work and {p1 / p0:.1f}x the peak memory."
        )
        print(
            "  Activation memory scales with batch size; the weights do not. That is\n"
            "  the whole memory argument for batching -- and the whole reason a batch\n"
            "  size that works in testing OOMs under a traffic spike."
        )
    return rows


def report_oom(settings: Settings, engine, start: int) -> None:
    section("4. Deliberate OOM, survived")
    c, h, w = engine.metadata.input_shape
    bs = start
    failed_at = None

    while bs <= HARD_BATCH_CAP:
        try:
            data = np.zeros((bs, c, h, w), dtype=np.float32)
        except MemoryError:
            print(f"  stopping at batch {bs}: host RAM, not VRAM, became the limit")
            break
        try:
            with memory_scope(f"batch {bs}", settings.device) as scope:
                engine.predict(data)
            print(f"  batch {bs:>5}  ok      peak {human_bytes(scope.peak_above_entry)}")
        except EngineError as exc:
            failed_at = bs
            print(f"  batch {bs:>5}  \033[31mOOM\033[0m")
            print(f"\n  the engine raised, cleanly:\n    {exc}")
            break
        bs *= 2

    if failed_at is None:
        print(f"  no OOM up to batch {HARD_BATCH_CAP}; rerun with a smaller --fraction")
        return

    # The point of the whole exercise: a request that fails must not take the
    # process with it. Recover and prove the engine still serves.
    empty_cache(settings.device)
    small = np.zeros((1, c, h, w), dtype=np.float32)
    result = engine.predict(small)

    print(
        f"\n  after empty_cache(), batch 1 still serves: logits {result.logits.shape}, "
        f"{result.timings.compute_ms:.2f} ms"
    )
    print(
        "\n  Why it OOMed: activations scale linearly with batch size while the\n"
        "  weights stay fixed, so a large enough batch needs a contiguous block\n"
        "  the allocator cannot supply. Why it did not crash: the engine catches\n"
        "  torch.cuda.OutOfMemoryError, calls empty_cache() so the next request\n"
        "  is not poisoned by the failed one, and re-raises as EngineError -- which\n"
        "  the API layer will map to 503 rather than a stack trace."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--precision", choices=[p.value for p in Precision], default="fp32")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--fraction",
        type=float,
        default=0.75,
        help="cap our allocator at this fraction of VRAM so the desktop keeps its memory",
    )
    args = ap.parse_args()

    overrides = {"precision": args.precision, "max_batch_size": HARD_BATCH_CAP}
    if args.device:
        overrides["device"] = args.device
    settings = Settings(_env_file=None, **overrides)

    if not settings.is_cuda:
        print("this report requires a CUDA device", file=sys.stderr)
        return 1

    print(f"\n\033[1mGPU memory report\033[0m  ({settings.precision.value}, {settings.device})")

    report_context_cost()
    limit_process_memory(args.fraction, settings.device)
    snap = get_gpu_memory(settings.device)
    print(
        f"\n  allocator capped at {args.fraction:.0%} of "
        f"{human_bytes(snap.total)} = {human_bytes(int(snap.total * args.fraction))}"
    )

    engine = create_engine(settings)
    try:
        report_lifecycle(settings, engine)
        rows = report_batch_scaling(settings, engine, [1, 2, 4, 8, 16, 32])
        report_oom(settings, engine, start=(rows[-1][0] * 2 if rows else 64))

        section("5. Teardown, and accounting for every last byte")
        with memory_scope("unload", settings.device) as scope:
            engine.unload()
        print(f"  {scope}")
        final = get_gpu_memory(settings.device)
        print(f"  after unload: {final}")
        print(
            f"\n  Weights are gone, but {human_bytes(final.allocated)} is still allocated and no\n"
            "  Python tensor holds it. It is the cuBLAS workspace: allocated on the first\n"
            "  GEMM (ResNet-18's final fc layer), cached per handle+stream, and kept for\n"
            "  the process lifetime. Confirmed by varying CUBLAS_WORKSPACE_CONFIG:\n"
            "      :4096:2:16:8  (torch default)  ->  8.12 MiB   = 4096 KiB x2 + 16 KiB x8\n"
            "      :1024:4                        ->  4.00 MiB\n"
            "      :16:8                          ->  128.00 KiB\n"
            "  Not a leak, and not worth shrinking -- a smaller workspace makes GEMMs\n"
            "  slower. It is worth *knowing about*, because 'memory_allocated() is\n"
            "  non-zero after I freed everything' is otherwise a day of hunting.\n"
            "\n  The CUDA context also survives until the process exits, so nvidia-smi\n"
            "  keeps showing this PID holding memory. Also not a leak."
        )
    finally:
        engine.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
