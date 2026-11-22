"""Fit, threshold, refit and test the reference baselines (Phase 13C).

The policy mirrors the AutoML-selected model so the comparison is fair:

1. ``fit_and_validate(train, validation)``
   Fit every baseline on the TRAIN split; score the validation split.
   Choose each decision threshold on the validation split. Report ``val_*``.
2. ``refit(train_val)``
   Refit every baseline on TRAIN + VALIDATION with the same fixed parameters
   (thresholds stay frozen). Report ``resub_*``: the refitted baseline on the
   validation patients, which are now part of its fit data (resubstitution,
   NOT held-out performance; same meaning as the model's ``resub_*``).
3. ``evaluate_test(test)``
   Score the untouched test split ONCE with the frozen thresholds. Report
   ``test_*``. Test data is accepted by no other method.

Threshold rules
---------------
* Probability baselines (B0, B3) use ``training.threshold_strategy`` exactly
  like the model (``fixed`` -> ``training.threshold``; ``validation_f1``).
* Raw-score baselines (B1, B2; HbA1c %) have no meaningful fixed threshold in
  [0, 1], so their threshold is always the F1-maximising cut-off on the
  validation scores (``validation_f1_raw_score``). If F1 is undefined
  (no validation positives) the cohort's HbA1c onset threshold is used and
  the fallback is recorded.

Like the model's, validation metrics are optimistic because the threshold is
chosen on the same split; only ``test_*`` is untouched by any choice.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dialong_automl.baselines.predictors import Baseline, BaselineInputs, default_baselines
from dialong_automl.evaluation.metrics import evaluate, f1_score, metrics_to_jsonable
from dialong_automl.evaluation.threshold import select_decision_threshold

logger = logging.getLogger(__name__)

POLICY: dict[str, Any] = {
    "fit": "train split",
    "threshold": "validation split, scored by the train-only fit",
    "refit": "train + validation, same fixed parameters, threshold frozen",
    "test": "test split scored once by the refitted baseline with the frozen threshold",
    "mirrors": "selected model's policy (search fit on train, threshold on validation, "
               "retrain on train + validation, single test evaluation)",
    "selectable_as_model": False,
    "optuna_candidate": False,
    "parameters_tuned": False,
}
METRIC_MEANING: dict[str, str] = {
    "val_": "train-only fit on the validation split (threshold also chosen there: optimistic)",
    "resub_": "train+validation refit on the validation patients (resubstitution, not held-out)",
    "test_": "train+validation refit on the untouched test split, evaluated once",
}
RAW_SCORE_STRATEGY = "validation_f1_raw_score"


def select_raw_score_threshold(
    y_validation: np.ndarray, scores_validation: np.ndarray, fallback_threshold: float
) -> dict[str, Any]:
    """F1-maximising threshold for a raw (non-probability) score, on validation only.

    Candidates are the smallest validation score and every midpoint between
    consecutive distinct validation scores (rule: positive if score >= t).
    Ties in F1 are broken towards the larger threshold. Deterministic.
    """
    y = np.asarray(y_validation)
    s = np.asarray(scores_validation, dtype=np.float64)
    if y.shape[0] != s.shape[0]:
        raise ValueError("Validation labels and scores differ in length.")
    info: dict[str, Any] = {
        "strategy": RAW_SCORE_STRATEGY,
        "threshold": float(fallback_threshold),
        "score_scale": "HbA1c %",
        "selected_on": "validation split, raw scores of the train-only fit",
        "n_samples": int(y.shape[0]),
        "n_positive": int((y == 1).sum()),
        "fallback_reason": None,
        "test_set_used": False,
    }
    u = np.unique(s[~np.isnan(s)])
    candidates = np.concatenate([u[:1], (u[1:] + u[:-1]) / 2.0]) if u.size else u
    best_t, best_f1 = None, -1.0
    for t in candidates:
        f1 = f1_score(y, s, threshold=float(t))
        if f1 is not None and f1 >= best_f1:
            best_t, best_f1 = float(t), f1
    if best_t is None:
        info["fallback_reason"] = (
            "F1 undefined at every candidate on the validation split; using the "
            "cohort HbA1c onset threshold."
        )
        logger.warning("Raw-score baseline threshold: %s", info["fallback_reason"])
    else:
        info["threshold"] = best_t
    return info


def _metrics(baseline: Baseline, y: np.ndarray, scores: np.ndarray, threshold: float,
             prefix: str) -> dict[str, Any]:
    res = evaluate(y, scores, threshold=threshold, prefix=prefix)
    out = {k: v for k, v in res.items() if not isinstance(v, dict)}
    if not baseline.is_probabilistic:
        # Raw HbA1c scores are not probabilities: Brier is undefined.
        out[f"{prefix}brier"] = None
    return metrics_to_jsonable(out)


class BaselineRunner:
    """Runs the fixed baselines through fit -> validate -> refit -> test once."""

    def __init__(
        self,
        baselines: list[Baseline],
        threshold_strategy: str,
        fixed_threshold: float,
        raw_score_fallback_threshold: float,
    ) -> None:
        names = [b.name for b in baselines]
        if len(set(names)) != len(names):
            raise ValueError(f"Baseline names must be unique, got {names}.")
        self.baselines = baselines
        self.threshold_strategy = threshold_strategy
        self.fixed_threshold = float(fixed_threshold)
        self.raw_score_fallback_threshold = float(raw_score_fallback_threshold)
        self._stage = "new"
        self._validation: BaselineInputs | None = None
        self._results: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_config(cls, cfg: Any) -> BaselineRunner:
        horizon = cfg.baselines.b2_horizon_days or cfg.cohort.prediction_horizon_days
        return cls(
            baselines=default_baselines(
                horizon_days=int(horizon),
                C=cfg.baselines.b3_c,
                max_iter=cfg.training.max_iter_sklearn,
                seed=cfg.training.seed,
            ),
            threshold_strategy=cfg.training.threshold_strategy,
            fixed_threshold=cfg.training.threshold,
            raw_score_fallback_threshold=cfg.cohort.diabetes_onset_hba1c_threshold,
        )

    # ------------------------------------------------------------------

    def fit_and_validate(self, train: BaselineInputs, validation: BaselineInputs) -> None:
        """Step 1: fit on train, choose thresholds and report ``val_*`` on validation."""
        if self._stage != "new":
            raise RuntimeError("fit_and_validate() may only be called once.")
        for data, part in ((train, "train"), (validation, "validation")):
            if data.y is None:
                raise ValueError(f"The {part} partition needs labels.")
        self._validation = validation
        for b in self.baselines:
            b.fit(train)
            val_scores = b.predict_scores(validation)
            if b.is_probabilistic:
                sel = select_decision_threshold(
                    self.threshold_strategy, self.fixed_threshold,
                    y_validation=validation.y, proba_validation=val_scores,
                )
                sel["score_scale"] = "probability"
                if sel["strategy"] == "validation_f1":
                    sel["selected_on"] = "validation split, probabilities of the train-only fit"
            else:
                sel = select_raw_score_threshold(
                    validation.y, val_scores, self.raw_score_fallback_threshold
                )
            sel.pop("applied_to", None)
            sel["applied_to"] = "train+validation refit, unchanged (resub_ and test_)"
            self._results[b.name] = {
                "code": b.code,
                "description": b.description,
                "is_probabilistic": b.is_probabilistic,
                "threshold": float(sel["threshold"]),
                "threshold_selection": sel,
                "params_train_fit": b.params(),
                "validation_metrics": _metrics(
                    b, validation.y, val_scores, sel["threshold"], "val_"
                ),
            }
        self._stage = "validated"

    def refit(self, train_val: BaselineInputs) -> None:
        """Step 2: refit on train + validation; report ``resub_*`` on the validation patients."""
        if self._stage != "validated":
            raise RuntimeError("refit() requires fit_and_validate() first and runs once.")
        if train_val.y is None:
            raise ValueError("The train+validation partition needs labels.")
        for b in self.baselines:
            b.fit(train_val)
            r = self._results[b.name]
            scores = b.predict_scores(self._validation)
            r["params_final_fit"] = b.params()
            r["retrain_resubstitution_metrics"] = _metrics(
                b, self._validation.y, scores, r["threshold"], "resub_"
            )
        self._stage = "refitted"

    def evaluate_test(self, test: BaselineInputs) -> None:
        """Step 3: score the test split once with the frozen thresholds."""
        if self._stage != "refitted":
            raise RuntimeError("evaluate_test() requires refit() first and runs exactly once.")
        if test.y is None:
            raise ValueError("The test partition needs labels.")
        for b in self.baselines:
            r = self._results[b.name]
            r["test_metrics"] = _metrics(
                b, test.y, b.predict_scores(test), r["threshold"], "test_"
            )
        self._stage = "tested"

    # ------------------------------------------------------------------

    @property
    def stage(self) -> str:
        return self._stage

    def results(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._results.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "stage": self._stage,
            "policy": dict(POLICY),
            "metric_prefixes": dict(METRIC_MEANING),
            "threshold_strategy_probability_baselines": self.threshold_strategy,
            "threshold_strategy_raw_score_baselines": RAW_SCORE_STRATEGY,
            "results": self.results(),
            "disclaimer": "Synthetic demonstration data. Reference baselines only; "
                          "not clinical evidence.",
        }


def comparison_rows(
    selected_model: str,
    model_threshold: float,
    model_val_metrics: dict[str, Any],
    model_test_metrics: dict[str, Any],
    baseline_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per predictor (selected model first) for ``baselines_comparison.csv``."""
    keys = ("auroc", "auprc", "brier", "f1", "recall", "precision")

    def row(name: str, kind: str, thr: float, val: dict, test: dict) -> dict[str, Any]:
        out: dict[str, Any] = {"predictor": name, "kind": kind, "threshold": thr}
        for k in keys:
            out[f"val_{k}"] = val.get(f"val_{k}")
        for k in keys:
            out[f"test_{k}"] = test.get(f"test_{k}")
        out["test_n_samples"] = test.get("test_n_samples")
        out["test_n_positive"] = test.get("test_n_positive")
        return out

    rows = [row(selected_model, "selected_model", model_threshold,
                model_val_metrics, model_test_metrics)]
    for name, r in baseline_results.items():
        rows.append(row(name, f"baseline_{r['code']}", r["threshold"],
                        r.get("validation_metrics", {}), r.get("test_metrics", {})))
    return rows
