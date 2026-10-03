"""Evaluation sub-package for DiaLong-AutoML — Phase 5.

Usage::

    from dialong_automl.evaluation import evaluate, auroc, auprc
"""

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
    select_threshold_by_f1,
    metrics_to_jsonable,
)

__all__ = [
    "auroc", "auprc", "accuracy", "precision", "recall",
    "specificity", "f1_score", "brier_score", "confusion_matrix", "evaluate",
    "select_threshold_by_f1", "metrics_to_jsonable",
]
