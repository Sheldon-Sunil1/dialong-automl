"""Human-readable feature name resolver for DiaLong-AutoML explainability.

Uses Phase 4 :class:`~dialong_automl.features.wide.FeatureInfo` mappings so
that explanation outputs show names like ``"Latest HbA1c"`` rather than
opaque column identifiers like ``"hba1c_latest"``.

When a Phase 4 mapping does not exist for a preprocessor output column (e.g.
one-hot encoded categorical columns like ``sex_latest__M``), the resolver
constructs a readable name from the column string itself.

Usage::

    from dialong_automl.explainability.feature_names import FeatureNameResolver

    resolver = FeatureNameResolver(
        preprocessor_feature_names=preprocessor.feature_names_out,
        wide_feature_mapping=wide_result.feature_mapping,
    )
    readable = resolver.resolve_all()  # list[str] — one per preprocessor output col
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pretty-print vocabulary for raw variable names
# ---------------------------------------------------------------------------

_VAR_LABELS: dict[str, str] = {
    "hba1c":                    "HbA1c",
    "fasting_glucose":          "Fasting Glucose",
    "bmi":                      "BMI",
    "systolic_bp":              "Systolic Blood Pressure",
    "diastolic_bp":             "Diastolic Blood Pressure",
    "ldl_cholesterol":          "LDL Cholesterol",
    "hdl_cholesterol":          "HDL Cholesterol",
    "triglycerides":            "Triglycerides",
    "heart_rate":               "Heart Rate",
    "age_at_visit":             "Age at Visit",
    "sex":                      "Sex",
    "smoking_status":           "Smoking Status",
    "physical_activity_level":  "Physical Activity Level",
    "family_history_diabetes":  "Family History of Diabetes",
    "metformin_use":            "Metformin Use",
    "hypertension_diagnosis":   "Hypertension Diagnosis",
    "visit_count":              "Number of Observation Visits",
}

_TRANSFORMATION_LABELS: dict[str, str] = {
    "raw_observation":      "at wave {wave}",
    "latest":               "(latest)",
    "mean":                 "(mean)",
    "min":                  "(minimum)",
    "max":                  "(maximum)",
    "std":                  "(std dev)",
    "last_minus_first":     "(change / delta)",
    "linear_slope":         "(trend / slope)",
    "categorical_latest":   "(most recent)",
    "count":                "",
    "unknown":              "",
}


def _pretty_var(raw: str) -> str:
    return _VAR_LABELS.get(raw, raw.replace("_", " ").title())


def _pretty_transformation(transformation: str, wave: int | None = None) -> str:
    label = _TRANSFORMATION_LABELS.get(transformation, f"({transformation})")
    if "{wave}" in label and wave is not None:
        label = label.format(wave=wave)
    return label


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class FeatureNameResolver:
    """Map preprocessor output column names to human-readable strings.

    Parameters
    ----------
    preprocessor_feature_names:
        Ordered list of column names from
        :attr:`~dialong_automl.features.preprocessing.WidePreprocessor.feature_names_out`.
    wide_feature_mapping:
        Dict from
        :attr:`~dialong_automl.features.wide.WideRepresentationResult.feature_mapping`.
        May be ``None`` when the wide result is not available (falls back to
        string parsing).
    """

    def __init__(
        self,
        preprocessor_feature_names: list[str],
        wide_feature_mapping: dict[str, Any] | None = None,
    ) -> None:
        self.preprocessor_feature_names = preprocessor_feature_names
        self._wide_mapping = wide_feature_mapping or {}
        self._resolved: list[str] | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def resolve_all(self) -> list[str]:
        """Return one human-readable label per preprocessor output column."""
        if self._resolved is not None:
            return list(self._resolved)
        self._resolved = [self.resolve(col) for col in self.preprocessor_feature_names]
        return list(self._resolved)

    def resolve(self, preprocessor_col: str) -> str:
        """Resolve a single preprocessor output column name.

        Priority:
        1. Exact match in wide feature mapping.
        2. Match after stripping the OHE ``__category`` suffix.
        3. String-based heuristic fallback.
        """
        # 1. Exact match in wide mapping
        if preprocessor_col in self._wide_mapping:
            return self._from_feature_info(self._wide_mapping[preprocessor_col])

        # 2. OHE column: "sex_latest__M" → base = "sex_latest", cat = "M"
        if "__" in preprocessor_col:
            base, category = preprocessor_col.rsplit("__", 1)
            if base in self._wide_mapping:
                base_label = self._from_feature_info(self._wide_mapping[base])
                return f"{base_label} = {category}"
            # Fallback: parse base manually
            base_label = self._heuristic(base)
            return f"{base_label} = {category}"

        # 3. Heuristic fallback
        return self._heuristic(preprocessor_col)

    def _from_feature_info(self, info: Any) -> str:
        """Build a label from a FeatureInfo object or dict."""
        # Accept both dataclass and dict (for JSON-loaded mappings)
        if isinstance(info, dict):
            var = info.get("source_variable", "")
            transformation = info.get("transformation", "")
            wave = info.get("visit_wave", None)
            feature_type = info.get("feature_type", "")
        else:
            var = getattr(info, "source_variable", "")
            transformation = getattr(info, "transformation", "")
            wave = getattr(info, "visit_wave", None)
            feature_type = getattr(info, "feature_type", "")

        var_label = _pretty_var(var)
        trans_label = _pretty_transformation(transformation, wave=wave)

        if feature_type == "meta":
            return var_label
        if trans_label:
            return f"{var_label} {trans_label}".strip()
        return var_label

    def _heuristic(self, col: str) -> str:
        """Parse column names using naming conventions as a last resort."""
        # Wave: hba1c_w3
        m = re.match(r"^(.+)_w(\d+)$", col)
        if m:
            var, wave = m.group(1), int(m.group(2))
            return f"{_pretty_var(var)} at wave {wave}"

        # Temporal suffix: hba1c_latest, hba1c_mean, etc.
        for suffix in ("_latest", "_mean", "_min", "_max", "_std", "_delta", "_slope"):
            if col.endswith(suffix):
                var = col[: -len(suffix)]
                trans_key = suffix.lstrip("_")
                trans_label = _TRANSFORMATION_LABELS.get(trans_key, f"({trans_key})")
                return f"{_pretty_var(var)} {trans_label}".strip()

        # n_obs_visits
        if col == "n_obs_visits":
            return "Number of Observation Visits"

        # Generic
        return col.replace("_", " ").title()
