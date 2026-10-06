"""Explainability sub-package for DiaLong-AutoML — Phase 7.

Provides:
- SHAP TreeExplainer (with permutation fallback) for tabular/tree models.
- Temporal occlusion for the GRU sequence model.
- Human-readable feature name resolver using Phase 4 mappings.
- Structured JSON-serialisable explanation containers.

DISCLAIMER
----------
Every persisted explanation artifact states:
    "Explanation values describe model behavior on the supplied data
     and do not establish medical causality."

Usage::

    # Tabular model (LightGBM / XGBoost / RandomForest)
    from dialong_automl.explainability import TreeExplainerWrapper

    explainer = TreeExplainerWrapper(model, preprocessor, feature_mapping)
    global_exp  = explainer.global_explanation(X_val_df, y_val)
    patient_exp = explainer.patient_explanation(X_val_df, row_index=0)

    # GRU
    from dialong_automl.explainability import TemporalOcclusionExplainer

    gru_exp = TemporalOcclusionExplainer(gru_model)
    result  = gru_exp.explain(X_seq, mask, patient_id="SYN_00001")
"""

from dialong_automl.explainability.feature_names import (
    FeatureNameResolver,
)
from dialong_automl.explainability.formatter import (
    EXPLANATION_DISCLAIMER,
    FeatureContribution,
    FeatureImportanceEntry,
    GlobalExplanation,
    PatientExplanation,
    TemporalExplanation,
    TimestepInfluence,
)
from dialong_automl.explainability.shap_explainer import (
    TreeExplainerWrapper,
)
from dialong_automl.explainability.temporal_occlusion import (
    TemporalOcclusionExplainer,
)

__all__ = [
    # Feature names
    "FeatureNameResolver",
    # Formatters
    "EXPLANATION_DISCLAIMER",
    "FeatureImportanceEntry",
    "FeatureContribution",
    "GlobalExplanation",
    "PatientExplanation",
    "TimestepInfluence",
    "TemporalExplanation",
    # Explainers
    "TreeExplainerWrapper",
    "TemporalOcclusionExplainer",
]
