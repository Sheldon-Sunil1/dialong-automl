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

from dialong_automl.evaluation.calibration import (
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)
from dialong_automl.evaluation.stability import (
    shared_key_count,
    spearman_rank_corr,
    stability_report,
    topk_jaccard,
)
from dialong_automl.evaluation.result_schema import (
    SCHEMA_VERSION,
    ExperimentCondition,
    RunResult,
    config_hash,
    new_run_id,
)
from dialong_automl.evaluation.perturbation import (
    PerturbationResult,
    remove_measurements,
    remove_visits,
    truncate_to_recent_visits,
    jitter_visit_timing,
    thin_followup,
)

__all__ = [
    # Existing metrics (unchanged)
    "auroc", "auprc", "accuracy", "precision", "recall",
    "specificity", "f1_score", "brier_score", "confusion_matrix", "evaluate",
    "select_threshold_by_f1", "metrics_to_jsonable",
    # Phase 11 — calibration
    "expected_calibration_error", "maximum_calibration_error",
    "reliability_bins", "calibration_report",
    # Phase 11 — explanation stability
    "topk_jaccard", "spearman_rank_corr", "shared_key_count", "stability_report",
    # Phase 11 — result schema
    "RunResult", "ExperimentCondition", "config_hash", "new_run_id",
    "SCHEMA_VERSION",
    # Phase 11 — perturbation interface
    "PerturbationResult", "remove_measurements", "remove_visits",
    "truncate_to_recent_visits", "jitter_visit_timing", "thin_followup",
]
