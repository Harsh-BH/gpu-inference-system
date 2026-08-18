"""Stage 3: logits -> ranked, human-readable predictions.

WHY THIS RUNS INLINE (`workers=0`)

    Measured at 0.10 ms per request. A thread hop costs roughly 50 us of that,
    and a thread pool would add a queue, a context switch and a scheduling
    decision to a job that is one softmax and one argpartition. The stage runs
    on the event loop and the loop is back within a tenth of a millisecond.

    That is a *measured* exemption, not a guess. The rule this project follows
    is that anything long enough to stall the loop goes to a pool; this is not
    long enough, and `max_batch` keeps it vectorised so it stays that way as
    load rises.

WHY IT IS A SEPARATE STAGE AT ALL, RATHER THAN WORK IN THE HANDLER

    It used to live in the HTTP handler. Moving it here means the handler no
    longer knows that predictions come from logits, or that softmax exists --
    it awaits a result and serialises it. Every transformation is now in the
    pipeline, which is what makes the pipeline a description of the system
    rather than most of a description.
"""

from __future__ import annotations

import numpy as np

from src.pipeline import Stage, StageOutput
from src.postprocessing import Prediction, top_k


class ClassifyStage(Stage[np.ndarray, list[Prediction]]):
    """(num_classes,) logits -> the top-k classes, ranked."""

    name = "classify"

    def __init__(self, labels: list[str], *, k: int = 5) -> None:
        self._labels = labels
        self._k = k

    def process(self, items: list[np.ndarray]) -> StageOutput[list[Prediction]]:
        # Re-stacked so softmax and argpartition run once over (N, num_classes)
        # instead of N times over one row. At 4 KB per row the copy is free and
        # the vectorisation is not.
        rows = top_k(np.stack(items, axis=0), self._labels, k=self._k)
        return list(rows)
