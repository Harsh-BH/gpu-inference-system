"""CUDA execution mechanics: streams, pinned memory, graphs (Phases 18-20).

    uv run python scripts/cuda_experiments.py
    uv run python scripts/cuda_experiments.py --only streams

These exist to be understood, not to be adopted. Each one is a technique that
is widely recommended, and each is measured here rather than assumed -- two of
the three turn out not to help this workload, which is the useful part.

The production path stays deliberately simple. An optimisation that does not
measurably help is complexity with no upside, and complexity on the request
path is paid for at 3am.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from time import perf_counter

import torch

from src.config import Settings
from src.model.pytorch_model import load_from_repository
from src.utils.tensor_info import human_bytes


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def median_ms(fn, iterations: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        t0 = perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


# --- Phase 18: streams ----------------------------------------------------


def experiment_streams(model, device: str, iterations: int) -> None:
    """Do two streams overlap, and does that help?

    A CUDA stream is an ordered queue of work. Operations in one stream run in
    issue order; operations in *different* streams may overlap, subject to the
    hardware having spare capacity.

    That last clause is the whole story, and it is the part usually left out.
    Streams do not create parallelism, they only permit it. If one stream
    already saturates the SMs, a second cannot run alongside it -- there is
    nothing left to run on.
    """
    section("Phase 18: CUDA streams")

    batch = torch.zeros(8, 3, 224, 224, device=device)
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)

    def sequential() -> None:
        with torch.inference_mode():
            model(batch)
            model(batch)

    def two_streams() -> None:
        with torch.inference_mode():
            with torch.cuda.stream(stream_a):
                model(batch)
            with torch.cuda.stream(stream_b):
                model(batch)
        torch.cuda.synchronize()

    seq = median_ms(sequential, iterations)
    par = median_ms(two_streams, iterations)

    print(f"  two forward passes, default stream   {seq:8.3f} ms")
    print(f"  two forward passes, two streams      {par:8.3f} ms")
    print(f"  speedup                              {seq / par:8.2f}x")
    print(
        "\n  Streams permit overlap; they do not manufacture it. A ResNet-18 forward\n"
        "  pass at batch 8 already occupies the SMs, so a second stream finds no idle\n"
        "  hardware to fill and the two serialise anyway.\n"
        "\n  Where streams genuinely pay is overlapping *different kinds* of work --\n"
        "  a host-to-device copy on one stream while another computes, since the copy\n"
        "  engine and the SMs are separate hardware. That needs pinned memory, which\n"
        "  is the next experiment."
    )

    # The other half of the lesson: without synchronize, the timing is fiction.
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = perf_counter()
        model(batch)
        launched = (perf_counter() - t0) * 1000.0
        torch.cuda.synchronize()
        completed = (perf_counter() - t0) * 1000.0
    print(
        f"\n  one forward pass, timed at launch    {launched:8.3f} ms  <- WRONG\n"
        f"  the same pass, timed after sync      {completed:8.3f} ms  <- actual\n"
        "  Kernel launches are asynchronous. Every timing in this project\n"
        "  synchronises for this reason."
    )


# --- Phase 19: pinned memory and non-blocking transfers -------------------


def experiment_transfers(device: str, iterations: int) -> None:
    """Does pinned memory actually speed up H2D on this workload?

    Pageable host memory can be swapped out, so the GPU's DMA engine cannot
    read it directly -- the driver stages it through an internal pinned buffer
    first, one extra copy. Pinned (page-locked) memory can be DMA'd directly,
    and only pinned memory can be copied asynchronously.

    The cost is real: pinned pages cannot be swapped, so they reduce memory the
    OS can reclaim, and allocating them is slow.
    """
    section("Phase 19: pinned memory and non-blocking transfers")
    print(
        f"  {'batch':>6} {'bytes':>11} {'pageable':>11} {'pinned':>11} "
        f"{'pinned+async':>13} {'gain':>7}"
    )

    for batch_size in (1, 8, 32):
        host = torch.zeros(batch_size, 3, 224, 224)
        pinned = torch.zeros(batch_size, 3, 224, 224).pin_memory()

        pageable_ms = median_ms(lambda h=host: h.to(device, non_blocking=False), iterations)
        pinned_ms = median_ms(lambda p=pinned: p.to(device, non_blocking=False), iterations)
        async_ms = median_ms(lambda p=pinned: p.to(device, non_blocking=True), iterations)

        print(
            f"  {batch_size:>6} {human_bytes(host.numel() * 4):>11} "
            f"{pageable_ms:>10.3f}ms {pinned_ms:>10.3f}ms {async_ms:>12.3f}ms "
            f"{pageable_ms / max(async_ms, 1e-9):>6.2f}x"
        )

    print(
        "\n  non_blocking=True on PAGEABLE memory is a no-op: the driver has to stage\n"
        "  the copy through its own pinned buffer, which forces it synchronous anyway.\n"
        "  Asking for an async copy out of pageable memory quietly gets a sync one.\n"
        "\n  Whether this matters depends on the ratio. Phase 1 measured H2D at 0.22 ms\n"
        "  against 2.98 ms of inference and 17.8 ms of preprocessing, so halving the\n"
        "  transfer would move end-to-end latency by well under one percent. The\n"
        "  production path therefore uses pageable memory and says so, rather than\n"
        "  carrying a pinned-buffer pool for a rounding error."
    )


# --- Phase 20: CUDA graphs ------------------------------------------------


def experiment_cuda_graph(model, device: str, iterations: int) -> None:
    """Replace per-call kernel launches with one replay.

    A forward pass here is ~50 kernel launches, each costing a few microseconds
    of CPU time to enqueue. A CUDA graph records that whole sequence once and
    replays it as a single submission, so the CPU-side launch cost collapses.

    The constraints are severe and are the reason this is not the default:
    shapes are frozen at capture time, input and output buffers are fixed
    addresses that must be written in place, and any control flow that depends
    on data is impossible. That suits a fixed-batch-size server and not a
    dynamic batcher, which is exactly what this project has.
    """
    section("Phase 20: CUDA graphs")

    batch_size = 8
    static_input = torch.zeros(batch_size, 3, 224, 224, device=device)

    # Capture must happen on a non-default stream after a warmup, or the
    # allocator's lazy initialisation gets recorded into the graph.
    try:
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            # 3 iterations was not enough: the first attempt captured before
            # cuDNN had settled on its algorithm and replay came out 13% slower
            # than eager. A graph freezes whatever kernels were chosen at
            # capture time, so capture must happen on a fully warm engine.
            for _ in range(25):
                model(static_input)
        torch.cuda.current_stream(device).wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                static_output = model(static_input)
    except Exception as exc:  # noqa: BLE001 - the experiment may not be supported
        print(f"  graph capture unavailable on this setup: {type(exc).__name__}: {exc}")
        return

    # Measured alternately, not sequentially. Experiment 8 established that a
    # sequential comparison on this power-capped card measures whichever ran
    # last on a hotter, slower GPU -- and the first attempt here duly reported
    # graphs as 13% SLOWER than eager, which is not a thing graphs can be.
    eager_samples, replay_samples = [], []
    with torch.inference_mode():
        for _ in range(10):
            model(static_input)
            graph.replay()
        torch.cuda.synchronize()
        for _ in range(iterations):
            t0 = perf_counter()
            model(static_input)
            torch.cuda.synchronize()
            eager_samples.append((perf_counter() - t0) * 1000.0)

            t0 = perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            replay_samples.append((perf_counter() - t0) * 1000.0)

    eager_ms = statistics.median(eager_samples)
    replay_ms = statistics.median(replay_samples)

    print(f"  eager, {batch_size} images            {eager_ms:8.3f} ms")
    print(f"  graph replay                    {replay_ms:8.3f} ms")
    print(f"  speedup                         {eager_ms / max(replay_ms, 1e-9):8.2f}x")
    print(f"  output still finite: {bool(static_output.sum().isfinite())}")
    print(
        "\n  The saving is CPU-side launch overhead, so it shows up when the GPU work\n"
        "  per launch is small -- small batches, small models, many tiny kernels. It\n"
        "  cannot speed up the arithmetic itself.\n"
        "\n  Not adopted here. A graph pins the batch size, and this server's whole\n"
        "  design is a batch manager that dispatches whatever happens to be waiting.\n"
        "  Supporting graphs would mean capturing one per batch size and pinning\n"
        "  input buffers, for a saving that Phase 4 shows is already amortised away\n"
        "  at batch 8 and above."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=["streams", "transfers", "graph"], default=None)
    ap.add_argument("--iterations", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("these experiments require a CUDA device", file=sys.stderr)
        return 1

    settings = Settings(_env_file=None)
    device = settings.device
    torch.backends.cudnn.allow_tf32 = settings.allow_tf32

    print(f"\n\033[1mCUDA experiments\033[0m  {torch.cuda.get_device_name(0)}")

    model = None
    if args.only in (None, "streams", "graph"):
        model = load_from_repository(settings.model_dir, settings.model_name).eval().to(device)

    if args.only in (None, "streams"):
        experiment_streams(model, device, args.iterations)
    if args.only in (None, "transfers"):
        experiment_transfers(device, args.iterations)
    if args.only in (None, "graph"):
        experiment_cuda_graph(model, device, args.iterations)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
