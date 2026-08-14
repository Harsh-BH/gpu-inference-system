"""Comparing engine configurations fairly.

Phase 5 needed FP32 vs TF32 vs FP16. Phase 7 needs PyTorch vs ONNX Runtime.
Phase 9 needs all five at once. Written three times, the sweep loop and the
tables drift, and a comparison whose two halves are measured differently is not
a comparison. So the machinery lives here once and the scripts supply only what
is actually different: which configurations, and what narrative to wrap around
the numbers.

THE ORDERING PROBLEM THIS SOLVES

    Experiment 8 established that this GPU is power-capped and that whatever
    runs last in a long sweep is measured on a hotter, slower device. Comparing
    *configurations* is where that does the most damage: a 15% win means
    nothing if the winner simply ran first.

    So the sweep is grouped by batch size — all configurations for one batch
    size run back-to-back, seconds apart rather than minutes — and the order
    within each group rotates, so every configuration takes every position
    across the sweep. Residual drift is then spread evenly instead of landing
    on one contestant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.benchmark import BenchmarkResult, benchmark_engine
from src.config import Backend, Precision, Settings
from src.inference import EngineError, create_engine


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """One thing worth measuring, and the settings that produce it."""

    label: str
    backend: Backend
    precision: Precision = Precision.FP32
    allow_tf32: bool = False

    def settings(self, **overrides) -> Settings:
        return Settings(
            _env_file=None,
            backend=self.backend,
            precision=self.precision,
            allow_tf32=self.allow_tf32,
            **overrides,
        )


def result_key(result: BenchmarkResult) -> str:
    """Identify which configuration a row came from.

    backend plus math_mode, because precision alone cannot distinguish FP32
    from TF32 -- both store float32 and differ only in how the arithmetic runs.
    """
    return f"{result.backend}/{result.math_mode}"


def run_interleaved_sweep(
    configs: list[EngineConfig],
    batch_sizes: list[int],
    *,
    iterations: int = 300,
    warmup: int = 20,
    device: str | None = None,
    report: Callable[[str], None] = print,
) -> list[BenchmarkResult]:
    """Benchmark every configuration at every batch size, order-balanced.

    Engines are loaded and unloaded around each measurement rather than held
    open. It costs about a second per configuration and buys two things: only
    one engine's memory is resident at a time, so a large batch does not OOM
    because of a rival still holding VRAM; and each measurement starts from a
    comparable allocator state.

    A configuration that fails is reported and skipped, never fatal -- finding
    that a backend cannot do something is a result, and it must not discard the
    rows already collected.
    """
    results: list[BenchmarkResult] = []
    overrides = {"max_batch_size": max(batch_sizes)}
    if device:
        overrides["device"] = device

    for i, batch_size in enumerate(batch_sizes):
        rotation = i % len(configs)
        order = configs[rotation:] + configs[:rotation]
        report(f"  batch {batch_size}: {' -> '.join(c.label for c in order)}")

        for config in order:
            engine = None
            try:
                engine = create_engine(config.settings(**overrides))
                engine.load()
                results.append(
                    benchmark_engine(engine, batch_size, iterations=iterations, warmup=warmup)
                )
            except EngineError as exc:
                report(f"    {config.label} batch {batch_size}: skipped ({exc})")
            finally:
                if engine is not None:
                    engine.unload()
    return results


def collect_logits(
    configs: list[EngineConfig],
    eval_batch: np.ndarray,
    *,
    device: str | None = None,
    warmup: int = 5,
    report: Callable[[str], None] = print,
) -> dict[str, np.ndarray]:
    """Run one identical batch through each configuration, keyed by label.

    Separate from the timing sweep because it answers a different question.
    Speed without agreement is meaningless: a backend that is twice as fast and
    picks a different class is a regression, not an optimisation.
    """
    overrides = {"max_batch_size": int(eval_batch.shape[0])}
    if device:
        overrides["device"] = device

    logits: dict[str, np.ndarray] = {}
    for config in configs:
        engine = None
        try:
            engine = create_engine(config.settings(**overrides))
            engine.load()
            engine.warmup(warmup)
            logits[config.label] = engine.predict(eval_batch).logits
        except EngineError as exc:
            report(f"  {config.label}: unavailable ({exc})")
        finally:
            if engine is not None:
                engine.unload()
    return logits
