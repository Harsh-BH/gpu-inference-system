"""Logits -> ranked, human-readable predictions.

WHY this is shared and backend-agnostic

    Softmax, top-k and label lookup are identical whether the logits came from
    PyTorch, ONNX Runtime or TensorRT. Implemented inside each engine they
    would be triplicated and would drift; implemented once here, a backend
    comparison differs only in the part that actually differs.

    numpy only — no torch. Same boundary the rest of the request path keeps.

WHAT A LOGIT IS

    The raw output of the final linear layer: an unbounded real number per
    class, where bigger means "more evidence for this class". Not a
    probability — logits can be negative and do not sum to anything.
    Softmax turns them into a distribution.

    Note that argmax(logits) == argmax(softmax(logits)), because softmax is
    monotonic. So the *predicted class* never depends on softmax; only the
    confidence number does. That matters in Phase 9: an FP16 engine can shift
    confidences slightly while still choosing the identical class, and those
    are two different kinds of "agreement" worth reporting separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Prediction:
    class_index: int
    label: str
    confidence: float

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence:.2%})"


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis.

    The `- max` shift is not an optimisation, it is a correctness requirement.
    exp(89) already overflows float32 to inf, and inf/inf is nan — so a
    confident prediction would return nan confidence for every class.
    Subtracting the row max leaves the result mathematically identical (the
    factor cancels) while keeping every exponent <= 0.

    This bites hardest on FP16 backends, where logits are noisier and the
    overflow threshold is far lower.
    """
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def top_k(logits: np.ndarray, labels: list[str], k: int = 5) -> list[list[Prediction]]:
    """Rank each row and return its k best classes.

    argpartition rather than a full argsort: it is O(n) instead of O(n log n)
    and we only need the top 5 of 1000. Marginal here, but this runs on every
    request and the habit is what keeps postprocessing off the critical path.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected (N, num_classes) logits, got shape {logits.shape}")
    if logits.shape[1] != len(labels):
        raise ValueError(
            f"logits have {logits.shape[1]} classes but {len(labels)} labels were "
            "loaded -- the model and its labels.json disagree"
        )

    probs = softmax(logits)
    k = min(k, probs.shape[1])

    # argpartition puts the k largest somewhere in the last k slots, unordered;
    # we then sort just those k.
    part = np.argpartition(probs, -k, axis=1)[:, -k:]
    rows = []
    for row_idx in range(probs.shape[0]):
        idx = part[row_idx]
        idx = idx[np.argsort(probs[row_idx, idx])[::-1]]
        rows.append(
            [
                Prediction(
                    class_index=int(i),
                    label=labels[int(i)],
                    confidence=float(probs[row_idx, i]),
                )
                for i in idx
            ]
        )
    return rows
