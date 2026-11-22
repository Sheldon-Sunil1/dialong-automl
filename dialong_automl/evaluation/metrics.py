"""Classification metrics for DiaLong-AutoML.

All metrics are computed from actual model predictions — nothing is
fabricated.  When a metric is mathematically undefined (e.g. AUROC with
only one class in y_true), the function returns ``None`` rather than
raising or returning a fabricated value.

AUROC and AUPRC are implemented in pure NumPy to avoid dependency on
sklearn compiled extensions (which may be blocked by OS policy). Tied scores
are grouped into one threshold, so results do not depend on row order, and
AUPRC is the step-wise average precision. Both match scikit-learn's
``roc_auc_score`` / ``average_precision_score`` (Phase 13C, RE-7).

Default threshold: 0.5 for binary predictions.
Threshold selection must always use the validation set only.
The test set must never be used to select a threshold.

Usage::

    from dialong_automl.evaluation.metrics import evaluate

    result = evaluate(y_true, y_proba)
    print(result["auroc"], result["auprc"])
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-numpy AUC helpers (no sklearn dependency)
# ---------------------------------------------------------------------------

def _binary_clf_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative (FP, TP) counts at each DISTINCT score threshold, descending.

    Tied scores are grouped into a single threshold, so the result does not
    depend on the row order of tied samples (Phase 13C, audit item RE-7).
    Previously every sample was its own threshold, which made e.g. a constant
    predictor's AUROC vary between roughly 0.2 and 0.7 with the label order.
    Equivalent to scikit-learn's internal ``_binary_clf_curve``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_score = y_score[order]
    y_true = y_true[order].astype(np.float64)
    # Index of the last sample of each block of tied scores.
    distinct = np.where(np.diff(y_score))[0]
    threshold_idx = np.concatenate([distinct, [y_true.size - 1]])
    tps = np.cumsum(y_true)[threshold_idx]
    fps = (threshold_idx + 1) - tps
    return fps, tps


def _roc_curve_numpy(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """FPR and TPR at every distinct threshold, starting at (0, 0) (pure numpy).

    Ties are grouped (see :func:`_binary_clf_curve`); a block of tied scores
    contributes a single diagonal segment, i.e. half credit, as in the
    Mann-Whitney definition of AUROC.
    """
    P = int(np.asarray(y_true).sum())
    N = len(y_true) - P
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    fps, tps = _binary_clf_curve(y_true, y_score)
    fpr = np.concatenate([[0.0], fps / N])
    tpr = np.concatenate([[0.0], tps / P])
    return fpr, tpr


def _auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC — compatible with NumPy 1.x and 2.x.

    The sort is STABLE: ROC points with equal x keep their curve order, so a
    vertical step is never reordered (which would change the area).
    """
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trapz(y, x))


def _precision_recall_curve_numpy(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall at every distinct threshold, descending (pure numpy).

    Ties are grouped (see :func:`_binary_clf_curve`). Recall is
    non-decreasing along the returned arrays.
    """
    P = int(np.asarray(y_true).sum())
    if P == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    fps, tps = _binary_clf_curve(y_true, y_score)
    prec = tps / (tps + fps)
    rec = tps / P
    return prec, rec


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def auroc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    """Area Under the ROC Curve (pure numpy).

    Returns ``None`` when y_true contains only one class (undefined).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        logger.warning("AUROC undefined: only one class present in y_true.")
        return None
    try:
        fpr, tpr = _roc_curve_numpy(y_true, y_proba)
        return _auc_trapezoid(fpr, tpr)
    except Exception as exc:
        logger.warning("AUROC computation failed: %s", exc)
        return None


def auprc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    """Area under the precision-recall curve as AVERAGE PRECISION (pure numpy).

    ``AP = sum_k (R_k - R_{k-1}) * P_k`` over distinct thresholds, with tied
    scores grouped. This is the step-wise definition used by scikit-learn's
    ``average_precision_score``; it gives a constant predictor exactly the
    prevalence. (Phase 13C, RE-7: the earlier version interpolated the PR
    curve with the trapezoid rule, which is optimistic and order-dependent
    under ties.)

    Returns ``None`` when y_true contains only one class.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        logger.warning("AUPRC undefined: only one class present in y_true.")
        return None
    try:
        prec, rec = _precision_recall_curve_numpy(y_true, y_proba)
        rec_prev = np.concatenate([[0.0], rec[:-1]])
        return float(np.sum((rec - rec_prev) * prec))
    except Exception as exc:
        logger.warning("AUPRC computation failed: %s", exc)
        return None


def _binary_preds(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(y_proba, dtype=np.float64) >= threshold).astype(int)


def accuracy(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> float:
    """Accuracy = (TP + TN) / n."""
    y_pred = _binary_preds(y_proba, threshold)
    return float(np.mean(np.asarray(y_true) == y_pred))


def precision(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> float | None:
    """Precision = TP / (TP + FP). Returns None when denominator is 0."""
    y_pred = _binary_preds(y_proba, threshold)
    y_true = np.asarray(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    denom = tp + fp
    return float(tp / denom) if denom > 0 else None


def recall(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> float | None:
    """Recall / Sensitivity = TP / (TP + FN). Returns None when denom is 0."""
    y_pred = _binary_preds(y_proba, threshold)
    y_true = np.asarray(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = tp + fn
    return float(tp / denom) if denom > 0 else None


def specificity(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> float | None:
    """Specificity = TN / (TN + FP). Returns None when denom is 0."""
    y_pred = _binary_preds(y_proba, threshold)
    y_true = np.asarray(y_true)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    denom = tn + fp
    return float(tn / denom) if denom > 0 else None


def f1_score(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> float | None:
    """F1 = 2 * precision * recall / (precision + recall). None when undefined."""
    p = precision(y_true, y_proba, threshold)
    r = recall(y_true, y_proba, threshold)
    if p is None or r is None:
        return None
    denom = p + r
    return float(2 * p * r / denom) if denom > 0 else None


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score = mean((y_true - y_proba)^2). Lower is better."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    return float(np.mean((y_true - y_proba) ** 2))


def confusion_matrix(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, int]:
    """Return TP, FP, TN, FN as a dict."""
    y_pred = _binary_preds(y_proba, threshold)
    y_true = np.asarray(y_true)
    return {
        "TP": int(((y_pred == 1) & (y_true == 1)).sum()),
        "FP": int(((y_pred == 1) & (y_true == 0)).sum()),
        "TN": int(((y_pred == 0) & (y_true == 0)).sum()),
        "FN": int(((y_pred == 0) & (y_true == 1)).sum()),
    }


def select_threshold_by_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    candidates: np.ndarray | None = None,
) -> float | None:
    """Choose threshold by maximizing F1 on *validation* labels only.

    Never pass test labels to this function.
    Returns ``None`` if F1 is undefined at every candidate.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if candidates is None:
        candidates = np.linspace(0.05, 0.95, 19)
    best_threshold: float | None = None
    best_f1 = -1.0
    for thr in candidates:
        score = f1_score(y_true, y_proba, threshold=float(thr))
        if score is None:
            continue
        if score > best_f1:
            best_f1 = score
            best_threshold = float(thr)
    if best_threshold is None:
        logger.warning("No defined F1 on the provided labels; threshold not selected.")
        return None
    return best_threshold


def metrics_to_jsonable(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert metric values (including numpy scalars) to JSON-safe types."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            out[key] = metrics_to_jsonable(value)
        elif value is None:
            out[key] = None
        elif isinstance(value, (np.floating, float)):
            out[key] = float(value)
        elif isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict[str, Any]:
    """Compute all classification metrics in one call.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0/1).
    y_proba:
        Model predicted probabilities for the positive class.
    threshold:
        Classification cutoff for binary predictions. Default 0.5.
        Must be selected using validation data only — never the test set.
    prefix:
        Optional string prefix applied to every key (e.g. ``"val_"``).

    Returns
    -------
    dict[str, Any]
        Keys: auroc, auprc, accuracy, precision, recall, specificity,
              f1, brier, confusion_matrix (as sub-dict).
        Values are float or None where undefined.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    result: dict[str, Any] = {
        "auroc":            auroc(y_true, y_proba),
        "auprc":            auprc(y_true, y_proba),
        "accuracy":         accuracy(y_true, y_proba, threshold),
        "precision":        precision(y_true, y_proba, threshold),
        "recall":           recall(y_true, y_proba, threshold),
        "specificity":      specificity(y_true, y_proba, threshold),
        "f1":               f1_score(y_true, y_proba, threshold),
        "brier":            brier_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_proba, threshold),
        "threshold":        threshold,
        "n_samples":        len(y_true),
        "n_positive":       int(y_true.sum()),
        "n_negative":       int((y_true == 0).sum()),
    }

    if prefix:
        return {f"{prefix}{k}": v for k, v in result.items()}
    return result
