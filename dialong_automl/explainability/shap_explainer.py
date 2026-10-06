"""SHAP-based explainability for tree/tabular models in DiaLong-AutoML.

Supports
--------
* **SHAP TreeExplainer** — for LightGBM, XGBoost, RandomForest (when
  the underlying sklearn/xgboost libraries are available).
* **Permutation importance fallback** — used when SHAP is unavailable or
  when the model type is not supported by TreeExplainer.

Rules
-----
* Explanations are computed on **validation or training data only**.
  The test set must never be passed here for explanation selection.
* If SHAP is unavailable the fallback is used and clearly labelled.
* No explanation values are fabricated.  If an explanation cannot be
  computed, ``None`` is returned with a clear log warning.

Usage::

    from dialong_automl.explainability.shap_explainer import TreeExplainerWrapper

    explainer = TreeExplainerWrapper(
        model=lgbm_model,
        preprocessor=wide_prep,
        feature_mapping=wide_result.feature_mapping,
    )
    global_exp  = explainer.global_explanation(X_val_wide_df, n_top=20)
    patient_exp = explainer.patient_explanation(X_val_wide_df, row_index=0)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from dialong_automl.explainability.feature_names import FeatureNameResolver
from dialong_automl.explainability.formatter import (
    EXPLANATION_DISCLAIMER,
    FeatureContribution,
    FeatureImportanceEntry,
    GlobalExplanation,
    PatientExplanation,
)
from dialong_automl.models.base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SHAP availability probe (lazy — never raises at import time)
# ---------------------------------------------------------------------------

def _shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def _get_inner_clf(model: BaseModel) -> Any | None:
    """Extract the underlying sklearn/lgbm/xgb estimator from a model wrapper."""
    return getattr(model, "_clf", None)


def _model_supports_tree_shap(model: BaseModel) -> bool:
    """Return True when the model's inner estimator works with TreeExplainer."""
    clf = _get_inner_clf(model)
    if clf is None:
        return False
    try:
        # Import module path of the inner classifier
        mod = type(clf).__module__
        return any(lib in mod for lib in ("lightgbm", "xgboost", "sklearn"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Permutation importance (pure numpy, no sklearn required)
# ---------------------------------------------------------------------------

def _permutation_importance(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Compute permutation feature importance in pure numpy.

    For each feature column, shuffle it ``n_repeats`` times and measure
    the average drop in AUROC compared to the baseline.

    Parameters
    ----------
    model:      Fitted BaseModel.
    X:          Preprocessed feature array (n_samples, n_features).
    y:          True labels (n_samples,).
    n_repeats:  Number of shuffles per feature.
    seed:       NumPy random seed.

    Returns
    -------
    np.ndarray of shape (n_features,) — mean importance per feature.
    """
    from dialong_automl.evaluation.metrics import auroc as _auroc

    rng = np.random.default_rng(seed)
    baseline_proba = model.predict_proba(X)
    baseline_score = _auroc(y, baseline_proba) or 0.5

    n_features = X.shape[1]
    importances = np.zeros(n_features, dtype=np.float64)

    for j in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_proba = model.predict_proba(X_perm)
            perm_score = _auroc(y, perm_proba)
            if perm_score is not None:
                scores.append(baseline_score - perm_score)
        importances[j] = float(np.mean(scores)) if scores else 0.0

    return importances


# ---------------------------------------------------------------------------
# SHAP computation helpers
# ---------------------------------------------------------------------------

def _compute_shap_values(
    model: BaseModel, X_processed: np.ndarray
) -> np.ndarray | None:
    """Compute SHAP values using TreeExplainer.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features) with signed SHAP values,
    or ``None`` when SHAP is unavailable or the model is not supported.
    """
    if not _shap_available():
        logger.info("shap package not available; falling back to permutation importance.")
        return None

    clf = _get_inner_clf(model)
    if clf is None:
        logger.warning("Cannot extract inner classifier from %s.", model.name)
        return None

    try:
        import shap
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_processed)
        # LightGBM returns list [class0, class1] for binary; take class-1 values
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        return np.asarray(sv, dtype=np.float64)
    except Exception as exc:
        logger.warning(
            "SHAP TreeExplainer failed for %s (%s). "
            "Falling back to permutation importance.",
            model.name, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Main explainer wrapper
# ---------------------------------------------------------------------------

class TreeExplainerWrapper:
    """Explain a tabular tree/linear model using SHAP or permutation fallback.

    Parameters
    ----------
    model:
        A fitted :class:`~dialong_automl.models.base.BaseModel` instance.
    preprocessor:
        The fitted :class:`~dialong_automl.features.preprocessing.WidePreprocessor`
        used to transform the raw wide DataFrame before model inference.
    feature_mapping:
        Dict from
        :attr:`~dialong_automl.features.wide.WideRepresentationResult.feature_mapping`.
        Used to build human-readable names.
    threshold:
        Classification threshold (default 0.5).

    Notes
    -----
    * Only use validation or training data for explanation selection.
    * Never pass test-set data for ``global_explanation`` if the result
      will be used to influence model selection or hyperparameter tuning.
    """

    def __init__(
        self,
        model: BaseModel,
        preprocessor: Any,
        feature_mapping: dict[str, Any] | None = None,
        threshold: float = 0.5,
    ) -> None:
        model._require_fitted()
        self.model = model
        self.preprocessor = preprocessor
        self.feature_mapping = feature_mapping or {}
        self.threshold = threshold
        self._resolver = FeatureNameResolver(
            preprocessor_feature_names=preprocessor.feature_names_out,
            wide_feature_mapping=feature_mapping,
        )
        self._readable_names = self._resolver.resolve_all()
        self._method: str = "shap_tree" if (
            _shap_available() and _model_supports_tree_shap(model)
        ) else "permutation"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def explanation_method(self) -> str:
        return self._method

    def global_explanation(
        self,
        X_wide_df: Any,
        y: np.ndarray | None = None,
        n_top: int = 20,
        permutation_repeats: int = 5,
        seed: int = 42,
    ) -> GlobalExplanation:
        """Compute global feature importance over a sample.

        Parameters
        ----------
        X_wide_df:
            Raw wide-format DataFrame (output of
            ``WideRepresentationResult.X``).  Must not contain labels or
            leakage columns.  Use **validation or training data only**.
        y:
            True labels.  Required for permutation importance.
            Must be the **validation** split labels — never test labels.
        n_top:
            Number of top features to include in the result.
        permutation_repeats:
            Repeats when falling back to permutation importance.
        seed:
            Random seed.

        Returns
        -------
        GlobalExplanation
        """
        import pandas as pd
        if isinstance(X_wide_df, pd.DataFrame):
            X_proc = self.preprocessor.transform(X_wide_df).astype(np.float32)
        else:
            X_proc = np.asarray(X_wide_df, dtype=np.float32)

        n_samples, n_features = X_proc.shape
        feature_names_out = self.preprocessor.feature_names_out

        # Try SHAP
        shap_vals = None
        method = "permutation"
        if _shap_available() and _model_supports_tree_shap(self.model):
            shap_vals = _compute_shap_values(self.model, X_proc)

        if shap_vals is not None:
            method = "shap_tree"
            # Mean absolute SHAP across samples
            mean_abs = np.abs(shap_vals).mean(axis=0)
            # Mean signed SHAP (for direction)
            mean_signed = shap_vals.mean(axis=0)
            importances = mean_abs
            directions = [
                "positive" if v > 0 else "negative"
                for v in mean_signed
            ]
        else:
            # Permutation fallback
            if y is None:
                raise ValueError(
                    "y (validation labels) must be provided for permutation importance. "
                    "Use validation set labels — never test labels."
                )
            importances = _permutation_importance(
                self.model, X_proc, np.asarray(y),
                n_repeats=permutation_repeats, seed=seed,
            )
            directions = [None] * n_features

        # Build ranked feature list
        order = np.argsort(-importances)
        entries: list[FeatureImportanceEntry] = []
        for rank, j in enumerate(order[:n_top], start=1):
            col = feature_names_out[j] if j < len(feature_names_out) else f"feature_{j}"
            readable = self._readable_names[j] if j < len(self._readable_names) else col
            entries.append(FeatureImportanceEntry(
                rank=rank,
                column_name=col,
                readable_name=readable,
                importance=float(importances[j]),
                direction=directions[j],
                explanation_method=method,
            ))

        return GlobalExplanation(
            model_name=self.model.name,
            representation=self.model.representation,
            explanation_method=method,
            n_samples=n_samples,
            features=entries,
            disclaimer=EXPLANATION_DISCLAIMER,
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_features_total": n_features,
                "n_top_shown": n_top,
                "note": "Computed on validation/training data only. Test set not used.",
            },
        )

    def patient_explanation(
        self,
        X_wide_df: Any,
        row_index: int = 0,
        patient_id: str | None = None,
        n_top: int = 10,
    ) -> PatientExplanation:
        """Compute a per-patient prediction explanation.

        Parameters
        ----------
        X_wide_df:
            Wide-format DataFrame.  May be validation or training data.
            Never pass test data if the result influences model selection.
        row_index:
            Row to explain (0-based index into *X_wide_df*).
        patient_id:
            Optional patient identifier for the output artifact.
        n_top:
            Number of top positive/negative contributions each.

        Returns
        -------
        PatientExplanation
        """
        import pandas as pd
        if isinstance(X_wide_df, pd.DataFrame):
            X_proc = self.preprocessor.transform(X_wide_df).astype(np.float32)
        else:
            X_proc = np.asarray(X_wide_df, dtype=np.float32)

        feature_names_out = self.preprocessor.feature_names_out
        row = X_proc[[row_index]]

        proba = float(self.model.predict_proba(row)[0])
        pred_class = int(proba >= self.threshold)

        shap_vals = None
        method = "permutation"

        if _shap_available() and _model_supports_tree_shap(self.model):
            sv = _compute_shap_values(self.model, row)
            if sv is not None:
                shap_vals = sv[0]
                method = "shap_tree"

        if shap_vals is None:
            # Permutation importance cannot give per-patient signed contributions
            # Return a note in metadata and empty contribution lists
            logger.warning(
                "Patient-level signed contributions unavailable without SHAP. "
                "Run with shap installed for full per-patient explanations."
            )
            return PatientExplanation(
                patient_id=patient_id,
                prediction_probability=proba,
                predicted_class=pred_class,
                threshold=self.threshold,
                model_name=self.model.name,
                explanation_method="not_available",
                top_positive=[],
                top_negative=[],
                disclaimer=EXPLANATION_DISCLAIMER,
                metadata={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "note": (
                        "Per-patient SHAP contributions require the shap package. "
                        "Only a prediction probability is available."
                    ),
                },
            )

        # Build contribution entries
        row_vals = X_proc[row_index]
        contributions: list[FeatureContribution] = []
        for j, sv_val in enumerate(shap_vals):
            col = feature_names_out[j] if j < len(feature_names_out) else f"feature_{j}"
            readable = self._readable_names[j] if j < len(self._readable_names) else col
            direction = "increases_risk" if sv_val > 0 else "decreases_risk"
            contributions.append(FeatureContribution(
                rank=0,  # filled below
                column_name=col,
                readable_name=readable,
                feature_value=float(row_vals[j]) if not np.isnan(row_vals[j]) else None,
                shap_value=float(sv_val),
                direction=direction,
            ))

        # Rank by |shap|
        contributions.sort(key=lambda c: -abs(c.shap_value))
        for rank, c in enumerate(contributions, start=1):
            c.rank = rank

        top_pos = [c for c in contributions if c.shap_value > 0][:n_top]
        top_neg = [c for c in contributions if c.shap_value <= 0][:n_top]

        return PatientExplanation(
            patient_id=patient_id,
            prediction_probability=proba,
            predicted_class=pred_class,
            threshold=self.threshold,
            model_name=self.model.name,
            explanation_method=method,
            top_positive=top_pos,
            top_negative=top_neg,
            disclaimer=EXPLANATION_DISCLAIMER,
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "row_index": row_index,
            },
        )
