"""Postprocessing is shared by every backend, so a bug here shows up as all
three runtimes being 'consistently wrong together' -- the hardest kind to spot.
"""

import numpy as np
import pytest

from src.postprocessing.classification import softmax, top_k

LABELS = [f"class_{i}" for i in range(10)]


def test_softmax_rows_sum_to_one():
    probs = softmax(np.random.default_rng(0).normal(size=(4, 10)).astype(np.float32))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-6)


def test_softmax_survives_large_logits():
    # Without the max-shift this is exp(1000) -> inf -> inf/inf -> nan, and
    # every confidence in the response becomes nan. FP16 backends produce
    # logits large enough for this to be a real risk, not a theoretical one.
    probs = softmax(np.array([[1000.0, 999.0, 998.0]], dtype=np.float32))
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0, 0] > probs[0, 1] > probs[0, 2]


def test_softmax_is_shift_invariant():
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(softmax(logits), softmax(logits + 50.0), atol=1e-6)


def test_top_k_is_ranked_descending():
    logits = np.array([[0.1, 5.0, 0.3, 2.0, 0.2, 0, 0, 0, 0, 0]], dtype=np.float32)
    preds = top_k(logits, LABELS, k=3)[0]
    assert [p.class_index for p in preds] == [1, 3, 2]
    assert preds[0].confidence > preds[1].confidence > preds[2].confidence
    assert preds[0].label == "class_1"


def test_argmax_is_unaffected_by_softmax():
    # Softmax is monotonic, so the *predicted class* never depends on it.
    # Only the confidence does -- which is why Phase 9 reports class agreement
    # and confidence drift as two separate numbers.
    logits = np.random.default_rng(1).normal(size=(8, 10)).astype(np.float32)
    assert (softmax(logits).argmax(axis=1) == logits.argmax(axis=1)).all()


def test_each_row_ranked_independently():
    logits = np.array([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    rows = top_k(logits, LABELS[:3], k=1)
    assert rows[0][0].class_index == 0
    assert rows[1][0].class_index == 2


def test_k_larger_than_num_classes_is_clamped():
    logits = np.zeros((1, 3), dtype=np.float32)
    assert len(top_k(logits, LABELS[:3], k=99)[0]) == 3


def test_label_count_mismatch_is_caught():
    # models/<name>/<version>/labels.json disagreeing with the model head is a
    # silent-garbage bug otherwise: you get confident predictions with shifted names.
    with pytest.raises(ValueError, match="labels"):
        top_k(np.zeros((1, 10), dtype=np.float32), LABELS[:5])


def test_unbatched_logits_rejected():
    with pytest.raises(ValueError, match="N, num_classes"):
        top_k(np.zeros(10, dtype=np.float32), LABELS)
