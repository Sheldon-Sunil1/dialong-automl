"""Tests for evaluation metrics (Phase 5).

Covers:
- AUROC correct value
- AUROC returns None for single-class input
- AUPRC correct value
- AUPRC returns None for single-class input
- Accuracy
- Precision, recall, specificity exact values
- Specificity = TN / (TN + FP)
- F1
- Brier score
- Confusion matrix values
- evaluate() dict keys and types
- None returned for undefined metrics, never a fake value
"""

from __future__ import annotations

import numpy as np
import pytest

from dialong_automl.evaluation.metrics import (
    auroc,
    auprc,
    accuracy,
    precision,
    recall,
    specificity,
    f1_score,
    brier_score,
    confusion_matrix,
    evaluate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _perfect() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    return y_true, y_proba


def _all_correct_at_threshold() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.3, 0.7, 0.9])
    return y_true, y_proba


def _all_zeros_true() -> tuple[np.ndarray, np.ndarray]:
    """Single class — only negatives."""
    return np.array([0, 0, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4])


def _all_ones_true() -> tuple[np.ndarray, np.ndarray]:
    """Single class — only positives."""
    return np.array([1, 1, 1, 1]), np.array([0.6, 0.7, 0.8, 0.9])


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def test_auroc_perfect():
    y_true, y_proba = _perfect()
    score = auroc(y_true, y_proba)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_auroc_single_class_returns_none():
    y_true, y_proba = _all_zeros_true()
    assert auroc(y_true, y_proba) is None


def test_auroc_single_class_positives_returns_none():
    y_true, y_proba = _all_ones_true()
    assert auroc(y_true, y_proba) is None


def test_auroc_random_baseline():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_proba = rng.random(200)
    score = auroc(y_true, y_proba)
    # Should be near 0.5 for random
    assert 0.3 <= score <= 0.7


# ---------------------------------------------------------------------------
# AUPRC
# ---------------------------------------------------------------------------

def test_auprc_perfect():
    y_true, y_proba = _perfect()
    score = auprc(y_true, y_proba)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_auprc_single_class_returns_none():
    y_true, y_proba = _all_zeros_true()
    assert auprc(y_true, y_proba) is None


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def test_accuracy_all_correct():
    y_true, y_proba = _all_correct_at_threshold()
    acc = accuracy(y_true, y_proba, threshold=0.5)
    assert acc == pytest.approx(1.0)


def test_accuracy_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1])  # all wrong at 0.5
    acc = accuracy(y_true, y_proba, threshold=0.5)
    assert acc == pytest.approx(0.0)


def test_accuracy_half_correct():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.1, 0.9])  # 2/4 correct
    acc = accuracy(y_true, y_proba, threshold=0.5)
    assert acc == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Precision and recall
# ---------------------------------------------------------------------------

def test_precision_perfect():
    y_true, y_proba = _all_correct_at_threshold()
    p = precision(y_true, y_proba, threshold=0.5)
    assert p == pytest.approx(1.0)


def test_recall_perfect():
    y_true, y_proba = _all_correct_at_threshold()
    r = recall(y_true, y_proba, threshold=0.5)
    assert r == pytest.approx(1.0)


def test_precision_none_when_no_positives_predicted():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])  # all below 0.5 → no positive predictions
    p = precision(y_true, y_proba, threshold=0.5)
    assert p is None  # TP+FP = 0, undefined


def test_recall_none_when_no_actual_positives():
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    r = recall(y_true, y_proba, threshold=0.5)
    assert r is None  # TP+FN = 0, undefined


# ---------------------------------------------------------------------------
# Specificity  — TN / (TN + FP)
# ---------------------------------------------------------------------------

def test_specificity_correct_formula():
    # y_true=[0,0,1,1], y_pred=[0,1,0,1] → TN=1, FP=1, TP=1, FN=1
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.1, 0.9])
    spec = specificity(y_true, y_proba, threshold=0.5)
    # TN=1 (true_neg=index 0, pred=0), FP=1 (index 1, pred=1 but true 0)
    assert spec == pytest.approx(0.5, abs=1e-6)


def test_specificity_perfect():
    # All negatives correctly predicted as negative
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    spec = specificity(y_true, y_proba, threshold=0.5)
    assert spec == pytest.approx(1.0)


def test_specificity_none_when_no_actual_negatives():
    y_true = np.array([1, 1, 1, 1])
    y_proba = np.array([0.6, 0.7, 0.8, 0.9])
    spec = specificity(y_true, y_proba, threshold=0.5)
    assert spec is None  # TN+FP = 0, undefined


def test_specificity_is_tn_over_tn_plus_fp():
    """Explicit TN/(TN+FP) calculation matches the function."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9, 0.3, 0.8])
    # At threshold 0.5:
    #   preds = [0, 1, 0, 1, 0, 1]
    #   TN: y_true=0 & pred=0 → indices 0, 2 → TN=2
    #   FP: y_true=0 & pred=1 → index 1 → FP=1
    #   specificity = 2/(2+1) = 2/3
    spec = specificity(y_true, y_proba, threshold=0.5)
    assert spec == pytest.approx(2 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------

def test_f1_perfect():
    y_true, y_proba = _all_correct_at_threshold()
    f1 = f1_score(y_true, y_proba, threshold=0.5)
    assert f1 == pytest.approx(1.0)


def test_f1_none_when_undefined():
    # No positive predictions → precision undefined → f1 undefined
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    f1 = f1_score(y_true, y_proba, threshold=0.5)
    assert f1 is None


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def test_brier_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0])
    bs = brier_score(y_true, y_proba)
    assert bs == pytest.approx(0.0)


def test_brier_worst():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([1.0, 1.0, 0.0, 0.0])
    bs = brier_score(y_true, y_proba)
    assert bs == pytest.approx(1.0)


def test_brier_random():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.5, 0.5, 0.5, 0.5])
    bs = brier_score(y_true, y_proba)
    assert bs == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def test_confusion_matrix_values():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.1, 0.9])
    cm = confusion_matrix(y_true, y_proba, threshold=0.5)
    assert cm["TP"] == 1
    assert cm["TN"] == 1
    assert cm["FP"] == 1
    assert cm["FN"] == 1


def test_confusion_matrix_all_correct():
    y_true, y_proba = _all_correct_at_threshold()
    cm = confusion_matrix(y_true, y_proba, threshold=0.5)
    assert cm["FP"] == 0
    assert cm["FN"] == 0


# ---------------------------------------------------------------------------
# evaluate() combined
# ---------------------------------------------------------------------------

def test_evaluate_returns_all_keys():
    y_true, y_proba = _all_correct_at_threshold()
    result = evaluate(y_true, y_proba)
    for key in ["auroc", "auprc", "accuracy", "precision", "recall",
                "specificity", "f1", "brier", "confusion_matrix",
                "threshold", "n_samples", "n_positive", "n_negative"]:
        assert key in result, f"Missing key: {key}"


def test_evaluate_prefix():
    y_true, y_proba = _all_correct_at_threshold()
    result = evaluate(y_true, y_proba, prefix="val_")
    assert "val_auroc" in result
    assert "auroc" not in result


def test_evaluate_single_class_auroc_is_none():
    y_true, y_proba = _all_zeros_true()
    result = evaluate(y_true, y_proba)
    assert result["auroc"] is None


def test_evaluate_counts_correct():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.4, 0.7, 0.8])
    result = evaluate(y_true, y_proba)
    assert result["n_samples"] == 5
    assert result["n_positive"] == 2
    assert result["n_negative"] == 3
