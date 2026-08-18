"""Stage 2: N tensors -> N rows of logits, in one GPU call.

WHY THIS IS THE ONE STAGE THAT BATCHES

    Batching buys throughput and charges latency, and only here is the exchange
    rate favourable. Phase 4 measured this engine at 331 img/s / 2.50 ms p50 at
    batch 1, and 703 img/s / 23.01 ms at batch 16. The fixed costs a batch
    amortises -- kernel launches, the H2D round trip, the Python/CUDA boundary
    -- are per-*call*, not per-image, which is exactly the condition that makes
    batching pay.

    Decoding has no such fixed cost, which is why that stage does not batch.

WHY EXACTLY ONE WORKER

    One GPU, one CUDA context, one inference at a time. A second worker sharing
    this engine would serialise on the same context anyway while making the
    VRAM ceiling harder to reason about, and would let two batches allocate
    activation memory simultaneously on a 6 GB card. Scaling past one GPU is
    model replication across processes, not threads in this one.

WHY THE STAGE OWNS THE ENGINE'S LIFECYCLE

    `setup()` loads and warms; `teardown()` unloads. Putting that here rather
    than in application startup means the pipeline's `start()` is the single
    place a GPU is acquired, and its `stop()` the single place one is released
    -- including when a *later* stage fails to set up, which now unwinds this
    one automatically instead of leaking a loaded engine.

    Warmup before serving is not optional. A cold engine's first forward pass
    pays lazy CUDA context creation, kernel module loading, cuDNN algorithm
    selection and allocator growth -- hundreds of milliseconds, charged in full
    to whoever arrives first. Somebody has to pay it; it must not be a user.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from src.inference.base import EngineError, EngineMetadata, InferenceEngine, StageTimings
from src.pipeline import Stage, StageOutput

logger = logging.getLogger(__name__)


class InferenceStage(Stage[np.ndarray, np.ndarray]):
    """(C, H, W) float32 in, (num_classes,) float32 out, one row per item."""

    name = "infer"

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        warmup_requests: int = 0,
        on_timings: Callable[[StageTimings, int], None] | None = None,
    ) -> None:
        self._engine = engine
        self._warmup_requests = warmup_requests
        # h2d/compute/d2h are only observable from inside an engine, and only
        # meaningful for an engine. Reporting them through a callback keeps
        # that detail out of the generic StageReport, which knows nothing about
        # devices and should not learn.
        self._on_timings = on_timings

    @property
    def engine(self) -> InferenceEngine:
        """Exposed so readiness and prediction responses can report which model
        answered. A prediction without that context is unauditable."""
        return self._engine

    @property
    def metadata(self) -> EngineMetadata:
        return self._engine.metadata

    # --- lifecycle ------------------------------------------------------

    def setup(self) -> None:
        self._engine.load()
        logger.info(
            "engine loaded: backend=%s precision=%s math=%s device=%s",
            self._engine.metadata.backend,
            self._engine.metadata.precision,
            self._engine.metadata.math_mode,
            self._engine.metadata.device,
        )
        if self._warmup_requests > 0:
            self._engine.warmup(self._warmup_requests)
            logger.info("warmed up with %d iterations", self._warmup_requests)

    def teardown(self) -> None:
        self._engine.unload()

    # --- inference ------------------------------------------------------

    def process(self, items: list[np.ndarray]) -> StageOutput[np.ndarray]:
        """Stack, run once, split.

        Raises rather than returning per-item errors: a CUDA OOM or a dead
        engine means the batch genuinely did not happen, so every item in it
        must fail. There is no partial success to report.
        """
        # np.stack copies into fresh contiguous memory, and the copy is
        # required: these samples arrived from different requests at different
        # times and are scattered across the heap, while the H2D copy needs one
        # contiguous block to DMA.
        batch = np.stack(items, axis=0)
        result = self._engine.predict(batch)

        logits = result.logits
        if logits.shape[0] != len(items):
            # Rows cannot be mapped to items, so nobody gets a wrong answer.
            raise EngineError(
                f"engine returned {logits.shape[0]} rows for {len(items)} items"
            )

        if self._on_timings is not None:
            self._on_timings(result.timings, len(items))

        # Row i belongs to item i. np.stack preserved the order and nothing
        # between here and there has reordered.
        return [logits[i] for i in range(len(items))]
