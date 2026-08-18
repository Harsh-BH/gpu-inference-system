"""Concurrency: what happens as clients pile up (Phase 12).

    uv run uvicorn src.main:app --port 8000        # in another shell
    uv run python scripts/stress_test.py
    uv run python scripts/stress_test.py --levels 1,10,50 --requests 200

Writes benchmarks/results/concurrency.json and .csv.

THE FOUR NUMBERS, AND WHY THEY ARE NOT THE SAME NUMBER

    concurrency     how many requests are in flight at once. An input.
    throughput      completed requests per second. Bounded by the slowest
                    stage, not by concurrency.
    latency         how long one request takes, from the client's side.
    queueing delay  how long it spent waiting rather than being worked on.

    Little's Law ties them together: concurrency = throughput x latency. Once
    throughput saturates, adding concurrency cannot increase it -- the extra
    requests simply queue, so latency rises in exact proportion. That is the
    signature to look for: flat throughput, linearly climbing p99.

    Beyond that point the honest response is to reject rather than accept, and
    the queue's 503 is what does it. A system that accepts everything and slows
    down for everyone is worse than one that serves what it can and says no to
    the rest.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

SAMPLE_IMAGE = Path("data/dog.jpg")


@dataclass(frozen=True, slots=True)
class LevelResult:
    concurrency: int
    requests: int
    ok: int
    rejected: int  # 503: the queue did its job
    errors: int
    wall_s: float
    throughput_rps: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    server_queue_wait_p50_ms: float
    server_preprocess_p50_ms: float


def payload() -> bytes:
    if SAMPLE_IMAGE.is_file():
        return SAMPLE_IMAGE.read_bytes()
    from PIL import Image

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


async def run_level(
    url: str, data: bytes, concurrency: int, total: int, timeout: float
) -> LevelResult:
    """Keep exactly `concurrency` requests in flight until `total` complete.

    A semaphore rather than firing all N at once: the point is to hold a steady
    concurrency level, not to measure a thundering herd.
    """
    import httpx

    latencies: list[float] = []
    queue_waits: list[float] = []
    preprocess: list[float] = []
    ok = rejected = errors = 0
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def one() -> None:
            nonlocal ok, rejected, errors
            async with semaphore:
                started = perf_counter()
                try:
                    response = await client.post(
                        url, files={"file": ("image.jpg", data, "image/jpeg")}
                    )
                except Exception:
                    errors += 1
                    return
                latencies.append((perf_counter() - started) * 1000.0)
                if response.status_code == 200:
                    ok += 1
                    body = response.json()["latency"]
                    # Stage-keyed since Phase 18: `queued_ms` is time spent
                    # waiting in front of any stage, `stages["decode"]` is the
                    # image-to-tensor cost that used to be `preprocess_ms`.
                    queue_waits.append(body["queued_ms"])
                    preprocess.append(body["stages"].get("decode", 0.0))
                elif response.status_code == 503:
                    # Backpressure working as designed, not a failure.
                    rejected += 1
                else:
                    errors += 1

        started = perf_counter()
        await asyncio.gather(*(one() for _ in range(total)))
        wall = perf_counter() - started

    def pct(values: list[float], q: float) -> float:
        return float(np.percentile(values, q)) if values else 0.0

    return LevelResult(
        concurrency=concurrency,
        requests=total,
        ok=ok,
        rejected=rejected,
        errors=errors,
        wall_s=wall,
        # Rejections are not work done, so they do not count toward throughput.
        throughput_rps=ok / wall if wall > 0 else 0.0,
        p50_ms=pct(latencies, 50),
        p90_ms=pct(latencies, 90),
        p95_ms=pct(latencies, 95),
        p99_ms=pct(latencies, 99),
        max_ms=max(latencies) if latencies else 0.0,
        server_queue_wait_p50_ms=statistics.median(queue_waits) if queue_waits else 0.0,
        server_preprocess_p50_ms=statistics.median(preprocess) if preprocess else 0.0,
    )


def report(results: list[LevelResult]) -> None:
    print("\n\033[1mConcurrency sweep\033[0m")
    print(
        f"  {'clients':>8} {'ok':>6} {'503':>5} {'err':>5} {'req/s':>8} "
        f"{'p50 ms':>8} {'p95 ms':>9} {'p99 ms':>9} {'queue p50':>10}"
    )
    for r in results:
        print(
            f"  {r.concurrency:>8} {r.ok:>6} {r.rejected:>5} {r.errors:>5} "
            f"{r.throughput_rps:>8.1f} {r.p50_ms:>8.1f} {r.p95_ms:>9.1f} "
            f"{r.p99_ms:>9.1f} {r.server_queue_wait_p50_ms:>10.1f}"
        )

    if len(results) < 2:
        return
    best = max(results, key=lambda r: r.throughput_rps)
    first = results[0]
    print(
        f"\n  peak throughput {best.throughput_rps:.1f} req/s at {best.concurrency} clients"
        f"  (p99 {best.p99_ms:.0f} ms)"
    )
    print(
        f"  from {first.concurrency} to {best.concurrency} clients: "
        f"{best.throughput_rps / max(first.throughput_rps, 1e-9):.2f}x throughput, "
        f"{best.p99_ms / max(first.p99_ms, 1e-9):.2f}x p99"
    )
    saturated = [r for r in results if r.throughput_rps >= best.throughput_rps * 0.95]
    if saturated:
        knee = saturated[0]
        print(
            f"  saturated from {knee.concurrency} clients onward; past that, extra\n"
            "  concurrency converts directly into queueing delay rather than throughput\n"
            "  (Little's Law: concurrency = throughput x latency, and throughput is fixed)."
        )
    if any(r.rejected for r in results):
        worst = max(results, key=lambda r: r.rejected)
        print(
            f"  {worst.rejected} requests rejected with 503 at {worst.concurrency} clients --\n"
            "  the bounded queue shedding load rather than letting the backlog grow."
        )


async def main_async(args) -> int:
    try:
        import httpx  # noqa: F401
    except ImportError:
        print("error: httpx is required. Run: uv sync", file=sys.stderr)
        return 1

    import httpx as _httpx

    url = args.url.rstrip("/")
    try:
        async with _httpx.AsyncClient(timeout=5.0) as client:
            ready = await client.get(f"{url}/ready")
        if ready.status_code != 200:
            print(f"error: server at {url} is not ready: {ready.text}", file=sys.stderr)
            return 1
        info = ready.json()
    except Exception as exc:
        print(
            f"error: cannot reach {url} ({exc}).\n"
            "  start it with: uv run uvicorn src.main:app --port 8000",
            file=sys.stderr,
        )
        return 1

    print(
        f"\n\033[1mTarget\033[0m {url}  "
        f"{info['backend']}/{info['precision']} on {info['device']}, "
        f"max_batch={info['max_batch_size']}"
    )

    data = payload()
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    results = []
    for level in levels:
        print(f"  running {level} concurrent clients ...", flush=True)
        results.append(await run_level(f"{url}/predict", data, level, args.requests, args.timeout))
        # Let the queue drain so the next level starts from a clean slate.
        await asyncio.sleep(1.0)

    report(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import csv
    import json

    rows = [asdict(r) for r in results]
    (args.output_dir / "concurrency.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output_dir / "concurrency.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {args.output_dir}/concurrency.json and .csv")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--levels", default="1,5,10,25,50,100")
    ap.add_argument("--requests", type=int, default=200, help="requests per level")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
