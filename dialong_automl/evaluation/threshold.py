"""Decision-threshold selection for the training pipeline (Phase 13B).

Only the strategies already documented in ``TrainingConfig.threshold_strategy``
are supported:

``fixed``
    Use the configured ``training.threshold``. No data is used.
``validation_f1``
    Choose the threshold that maximises F1 on the VALIDATION split, using the
    best search model's predictions made BEFORE the final retrain. The function
    deliberately accepts no test-set arguments, so test labels cannot influence
    the result.

Any other value is rejected with ``ValueError``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dialong_automl.evaluation.metrics import select_threshold_by_f1

logger = logging.getLogger(__name__)

SUPPORTED_THRESHOLD_STRATEGIES: tuple[str, ...] = ("fixed", "validation_f1")

APPLIED_TO = (
    "final retrained model; the threshold is derived from the pre-retrain model's "
    "probabilities, which may be distributed differently, and it is not re-tuned"
)
SELECTED_ON_FIXED = "none (fixed configured value; no data used)"
SELECTED_ON_VALIDATION = (
    "validation split: best-trial model fitted on train only (before final retrain)"
)


def select_decision_threshold(
    strategy: str,
    fixed_threshold: float,
    y_validation: np.ndarray | None = None,
    proba_validation: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select the decision threshold and describe how it was chosen.

    Parameters
    ----------
    strategy:
        ``"fixed"`` or ``"validation_f1"``.
    fixed_threshold:
        The configured threshold (used by ``fixed`` and as the documented
        fallback when F1 is undefined at every candidate).
    y_validation, proba_validation:
        Validation labels and the validation-split probabilities of the model
        fitted on the training split. Required for ``validation_f1``.

    Returns
    -------
    dict
        ``strategy``, ``threshold``, ``selected_on`` (data partition),
        ``n_samples``, ``n_positive``, ``fallback_reason`` and
        ``test_set_used`` (always ``False``).
    """
    if strategy not in SUPPORTED_THRESHOLD_STRATEGIES:
        raise ValueError(
            f"Unsupported threshold_strategy {strategy!r}; "
            f"supported: {SUPPORTED_THRESHOLD_STRATEGIES}."
        )
    if not 0.0 <= float(fixed_threshold) <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {fixed_threshold!r}.")

    info: dict[str, Any] = {
        "strategy": strategy,
        "threshold": float(fixed_threshold),
        "selected_on": SELECTED_ON_FIXED,
        "n_samples": None,
        "n_positive": None,
        "fallback_reason": None,
        "test_set_used": False,
        "applied_to": APPLIED_TO,
    }
    if strategy == "fixed":
        return info

    # --- validation_f1 -----------------------------------------------------
    if y_validation is None or proba_validation is None:
        raise ValueError(
            "threshold_strategy='validation_f1' requires validation labels and "
            "validation probabilities."
        )
    y = np.asarray(y_validation)
    p = np.asarray(proba_validation, dtype=np.float64)
    if y.shape[0] != p.shape[0]:
        raise ValueError(
            f"Validation labels ({y.shape[0]}) and probabilities ({p.shape[0]}) differ in length."
        )
    info["selected_on"] = SELECTED_ON_VALIDATION
    info["n_samples"] = int(y.shape[0])
    info["n_positive"] = int((y == 1).sum())

    chosen = select_threshold_by_f1(y, p)
    if chosen is None:
        info["fallback_reason"] = (
            "F1 undefined at every candidate threshold on the validation split; "
            "using the configured fixed threshold."
        )
        logger.warning(info["fallback_reason"])
    else:
        info["threshold"] = float(chosen)
    return info
