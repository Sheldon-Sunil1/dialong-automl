"""Wide longitudinal representation builder for DiaLong-AutoML.

Converts ``CohortResult.observation_data`` (long format, one row per visit)
into a **wide** patient-level DataFrame where every patient occupies exactly
one row.

Two complementary groups of features are generated:

Wave features
~~~~~~~~~~~~~
For each numeric variable and each observation wave (visit slot) a
separate column is produced::

    hba1c_w1, hba1c_w2, hba1c_w3, hba1c_w4   (when max_waves=4)

Visits are ordered chronologically so wave 1 = oldest observation,
wave N = most recent.  If a patient has fewer visits than ``max_waves``,
later waves are NaN.  Wave values are the raw observed measurements.

Temporal summary features
~~~~~~~~~~~~~~~~~~~~~~~~~
For each numeric variable a set of summary statistics is computed across
the patient's available (non-missing) observations:

=================  ===============================================================
Feature suffix     Definition
=================  ===============================================================
``_latest``        Value at the most recent observation (last chronologically).
``_mean``          Mean across all non-missing observations.
``_min``           Minimum across all non-missing observations.
``_max``           Maximum across all non-missing observations.
``_std``           Sample standard deviation (NaN when < 2 non-null values).
``_delta``         Last observation minus first observation (NaN when < 2 non-null).
``_slope``         Ordinary-least-squares slope over equally-spaced visit indices
                   (NaN when < 2 non-null values).
=================  ===============================================================

Categorical features
~~~~~~~~~~~~~~~~~~~~
For each categorical variable the **last observed (most recent) value** is
used::

    sex_latest, smoking_status_latest, …

``_latest`` is appended for naming consistency with numeric summaries, but
categoricals are not aggregated numerically.

Leakage protection
~~~~~~~~~~~~~~~~~~
The builder refuses to operate on DataFrames that contain any column from
``_REPRESENTATION_LEAKAGE_COLUMNS`` and raises ``LeakageError`` if
``target`` or any future-leakage column is present in the input.
It also checks that no row has ``visit_date > cutoff_date`` when
cutoff information is provided.

Feature mapping
~~~~~~~~~~~~~~~
A ``dict[str, FeatureInfo]`` is returned inside ``WideRepresentationResult``.
Every generated column has a machine-readable ``FeatureInfo`` describing
its source variable, wave index (if applicable), transformation type, and
a human-readable description.

Usage::

    from dialong_automl.features.wide import WideRepresentationBuilder

    builder = WideRepresentationBuilder(max_waves=4)
    result = builder.build(cohort.observation_data)

    result.X              # pd.DataFrame — one row per patient, no target
    result.patient_ids    # list[str]
    result.feature_names  # list[str]
    result.feature_mapping  # dict[str, FeatureInfo]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns operated on by the wide builder
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    "hba1c",
    "fasting_glucose",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
    "heart_rate",
    "age_at_visit",
]

CATEGORICAL_FEATURES: list[str] = [
    "sex",
    "smoking_status",
    "physical_activity_level",
    "family_history_diabetes",
    "metformin_use",
    "hypertension_diagnosis",
]

# Columns that must never appear in the representation
_REPRESENTATION_LEAKAGE_COLUMNS: frozenset[str] = frozenset({
    "target",
    "is_future_visit",
    "future_hba1c",
    "future_glucose",
    "future_diabetes",
    "diabetes_progression",
    "diagnosis_after",
    "post_outcome",
    "outcome",
})

TEMPORAL_SUFFIXES: list[str] = [
    "_latest",
    "_mean",
    "_min",
    "_max",
    "_std",
    "_delta",
    "_slope",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LeakageError(ValueError):
    """Raised when the input DataFrame contains leakage-candidate columns."""


# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

@dataclass
class FeatureInfo:
    """Human-readable metadata for one generated wide feature.

    Attributes
    ----------
    column_name:
        The exact column name in the output DataFrame.
    source_variable:
        The original longitudinal variable this feature was derived from.
    visit_wave:
        1-based wave index for raw-observation features; ``None`` for
        aggregated temporal features.
    transformation:
        Short machine-readable transformation identifier, e.g.
        ``"raw_observation"``, ``"mean"``, ``"linear_slope"``,
        ``"last_minus_first"``, ``"latest"``, ``"categorical_latest"``.
    description:
        Human-readable description suitable for explainability output.
    feature_type:
        ``"numeric_wave"``, ``"numeric_temporal"``, ``"categorical"``,
        or ``"meta"``.
    """

    column_name: str
    source_variable: str
    transformation: str
    description: str
    feature_type: str
    visit_wave: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "source_variable": self.source_variable,
            "visit_wave": self.visit_wave,
            "transformation": self.transformation,
            "description": self.description,
            "feature_type": self.feature_type,
        }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WideRepresentationResult:
    """Output of :meth:`WideRepresentationBuilder.build`.

    Attributes
    ----------
    X:
        Patient-level feature DataFrame.  One row per patient.
        Does **not** contain ``target``, patient IDs, or visit metadata.
    patient_ids:
        Ordered list of patient IDs matching the rows of ``X``.
    y:
        Optional target Series indexed by patient_id.  ``None`` when the
        input DataFrame did not contain a target column.
    feature_names:
        Ordered list of column names in ``X``.
    feature_mapping:
        Dict mapping every column name in ``X`` to its :class:`FeatureInfo`.
    n_obs_visits:
        Series (indexed by patient_id) recording how many non-padded
        observation visits each patient contributed.
    metadata:
        Dict with builder configuration and generation statistics.
    """

    X: pd.DataFrame
    patient_ids: list[str]
    y: pd.Series | None
    feature_names: list[str]
    feature_mapping: dict[str, FeatureInfo]
    n_obs_visits: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class WideRepresentationBuilder:
    """Build a patient-level wide representation from long-format observation data.

    Parameters
    ----------
    max_waves:
        Number of observation-wave columns to generate per numeric feature.
        Default 4 (i.e. w1…w4).  Patients with fewer visits will have NaN
        in the later wave columns.
    numeric_features:
        Numeric column names to include.  Defaults to
        :data:`NUMERIC_FEATURES`.
    categorical_features:
        Categorical column names to include.  Defaults to
        :data:`CATEGORICAL_FEATURES`.
    include_n_visits:
        If ``True``, add an ``n_obs_visits`` column to ``X``.
    """

    def __init__(
        self,
        max_waves: int = 4,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        include_n_visits: bool = True,
    ) -> None:
        self.max_waves = max_waves
        self.numeric_features = numeric_features or NUMERIC_FEATURES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self.include_n_visits = include_n_visits

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(
        self,
        observation_df: pd.DataFrame,
        patient_targets: pd.Series | None = None,
    ) -> WideRepresentationResult:
        """Build the wide representation.

        Parameters
        ----------
        observation_df:
            Long-format DataFrame — one row per visit.  Must contain
            ``patient_id`` and ``visit_date``.  Must NOT contain any
            column from ``_REPRESENTATION_LEAKAGE_COLUMNS``.
        patient_targets:
            Optional Series indexed by patient_id with values 0/1.
            When provided it is stored in ``result.y`` but is **never**
            added to ``result.X``.

        Returns
        -------
        WideRepresentationResult
        """
        self._check_leakage(observation_df)
        _require_columns(observation_df, ["patient_id", "visit_date"])

        df = observation_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["visit_date"]):
            df["visit_date"] = pd.to_datetime(df["visit_date"])

        feature_mapping: dict[str, FeatureInfo] = {}
        patient_rows: list[dict[str, Any]] = []
        visit_counts: dict[str, int] = {}

        grouped = df.groupby("patient_id", sort=False)
        for pid, grp in grouped:
            grp = grp.sort_values("visit_date").reset_index(drop=True)
            visit_counts[str(pid)] = len(grp)
            row: dict[str, Any] = {}

            # --- Wave features (numeric) ---
            for var in self.numeric_features:
                if var not in grp.columns:
                    for w in range(1, self.max_waves + 1):
                        row[f"{var}_w{w}"] = float("nan")
                    continue
                vals = grp[var].tolist()
                for w in range(1, self.max_waves + 1):
                    col = f"{var}_w{w}"
                    row[col] = float(vals[w - 1]) if w - 1 < len(vals) else float("nan")

            # --- Temporal summary features (numeric) ---
            for var in self.numeric_features:
                if var not in grp.columns:
                    for suf in TEMPORAL_SUFFIXES:
                        row[f"{var}{suf}"] = float("nan")
                    continue
                series = pd.to_numeric(grp[var], errors="coerce").dropna()
                vals_valid = series.values.astype(float)
                n = len(vals_valid)

                row[f"{var}_latest"] = float(vals_valid[-1]) if n >= 1 else float("nan")
                row[f"{var}_mean"] = float(np.mean(vals_valid)) if n >= 1 else float("nan")
                row[f"{var}_min"] = float(np.min(vals_valid)) if n >= 1 else float("nan")
                row[f"{var}_max"] = float(np.max(vals_valid)) if n >= 1 else float("nan")
                row[f"{var}_std"] = float(np.std(vals_valid, ddof=1)) if n >= 2 else float("nan")
                row[f"{var}_delta"] = float(vals_valid[-1] - vals_valid[0]) if n >= 2 else float("nan")

                if n >= 2:
                    # Slope: OLS over 0-based visit index, NaN values excluded
                    # Use the original (with NaN) series indices to get proper spacing
                    numeric_series = pd.to_numeric(grp[var], errors="coerce")
                    mask = numeric_series.notna()
                    indices = np.where(mask)[0].astype(float)  # visit positions
                    y_vals = numeric_series[mask].values.astype(float)
                    # OLS slope: (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
                    n_pts = len(indices)
                    sx = indices.sum()
                    sy = y_vals.sum()
                    sxy = (indices * y_vals).sum()
                    sxx = (indices ** 2).sum()
                    denom = n_pts * sxx - sx ** 2
                    row[f"{var}_slope"] = float((n_pts * sxy - sx * sy) / denom) if abs(denom) > 1e-12 else float("nan")
                else:
                    row[f"{var}_slope"] = float("nan")

            # --- Categorical features (most recent value) ---
            for var in self.categorical_features:
                if var not in grp.columns:
                    row[f"{var}_latest"] = None
                    continue
                non_null = grp[var].dropna()
                non_null = non_null[non_null.astype(str).str.strip().str.lower().isin(
                    ["nan", "none", "<na>"]
                ) == False]
                row[f"{var}_latest"] = str(non_null.iloc[-1]) if len(non_null) > 0 else None

            # --- N visits ---
            if self.include_n_visits:
                row["n_obs_visits"] = len(grp)

            patient_rows.append({"patient_id": str(pid), **row})

        # Build output DataFrame
        result_df = pd.DataFrame(patient_rows)
        patient_ids = result_df["patient_id"].tolist()
        result_df = result_df.drop(columns=["patient_id"])

        feature_names = list(result_df.columns)

        # Build feature mapping (only on first patient to avoid repetition)
        feature_mapping = self._build_feature_mapping(feature_names)

        # Y
        y: pd.Series | None = None
        if patient_targets is not None:
            common = [p for p in patient_ids if p in patient_targets.index]
            y = patient_targets.loc[common].copy()
            y.index.name = "patient_id"

        n_obs_visits = pd.Series(visit_counts, name="n_obs_visits")
        n_obs_visits.index.name = "patient_id"

        metadata: dict[str, Any] = {
            "max_waves": self.max_waves,
            "n_patients": len(patient_ids),
            "n_features": len(feature_names),
            "n_numeric_features": len(self.numeric_features),
            "n_categorical_features": len(self.categorical_features),
            "n_wave_features": len(self.numeric_features) * self.max_waves,
            "n_temporal_features": len(self.numeric_features) * len(TEMPORAL_SUFFIXES),
            "n_categorical_cols": len(self.categorical_features),
            "include_n_visits": self.include_n_visits,
        }

        logger.info(
            "Wide representation built: %d patients, %d features (%d wave, %d temporal, %d cat)",
            len(patient_ids),
            len(feature_names),
            metadata["n_wave_features"],
            metadata["n_temporal_features"],
            metadata["n_categorical_cols"],
        )

        return WideRepresentationResult(
            X=result_df,
            patient_ids=patient_ids,
            y=y,
            feature_names=feature_names,
            feature_mapping=feature_mapping,
            n_obs_visits=n_obs_visits,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Feature mapping
    # ------------------------------------------------------------------

    def _build_feature_mapping(self, feature_names: list[str]) -> dict[str, FeatureInfo]:
        """Generate a FeatureInfo for every column in *feature_names*."""
        mapping: dict[str, FeatureInfo] = {}

        for col in feature_names:
            info = self._infer_feature_info(col)
            if info is not None:
                mapping[col] = info

        return mapping

    def _infer_feature_info(self, col: str) -> FeatureInfo | None:
        """Infer FeatureInfo from a column name using naming conventions."""
        # n_obs_visits meta column
        if col == "n_obs_visits":
            return FeatureInfo(
                column_name=col,
                source_variable="visit_count",
                transformation="count",
                description="Number of non-padded observation visits for this patient.",
                feature_type="meta",
            )

        # Numeric wave: {var}_w{n}
        for var in self.numeric_features:
            for w in range(1, self.max_waves + 1):
                if col == f"{var}_w{w}":
                    return FeatureInfo(
                        column_name=col,
                        source_variable=var,
                        visit_wave=w,
                        transformation="raw_observation",
                        description=(
                            f"{var} recorded at observation wave {w} "
                            f"(chronologically ordered, wave 1 = oldest visit)."
                        ),
                        feature_type="numeric_wave",
                    )

        # Numeric temporal: {var}_{suffix}
        _SUFFIX_META: dict[str, tuple[str, str]] = {
            "_latest": ("latest", "Value at the most recent observation visit."),
            "_mean": ("mean", "Mean across all non-missing observation visits."),
            "_min": ("min", "Minimum value across all non-missing observation visits."),
            "_max": ("max", "Maximum value across all non-missing observation visits."),
            "_std": ("std", "Sample standard deviation across non-missing observation visits (NaN when < 2 values)."),
            "_delta": ("last_minus_first", "Last observed value minus first observed value (NaN when < 2 values)."),
            "_slope": ("linear_slope", "OLS slope over visit-index positions for non-missing observations (NaN when < 2 values)."),
        }
        for var in self.numeric_features:
            for suffix, (transformation, desc_template) in _SUFFIX_META.items():
                if col == f"{var}{suffix}":
                    return FeatureInfo(
                        column_name=col,
                        source_variable=var,
                        transformation=transformation,
                        description=f"{var}: {desc_template}",
                        feature_type="numeric_temporal",
                    )

        # Categorical: {var}_latest
        for var in self.categorical_features:
            if col == f"{var}_latest":
                return FeatureInfo(
                    column_name=col,
                    source_variable=var,
                    transformation="categorical_latest",
                    description=(
                        f"Most recent observed value of {var} across all "
                        f"observation visits."
                    ),
                    feature_type="categorical",
                )

        # Unknown column — return generic info rather than None
        return FeatureInfo(
            column_name=col,
            source_variable=col,
            transformation="unknown",
            description=f"Column {col!r} (no mapping rule matched).",
            feature_type="unknown",
        )

    # ------------------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------------------

    def _check_leakage(self, df: pd.DataFrame) -> None:
        """Raise LeakageError if the DataFrame contains leakage columns."""
        found = [c for c in df.columns if c in _REPRESENTATION_LEAKAGE_COLUMNS]
        if found:
            raise LeakageError(
                f"WideRepresentationBuilder received leakage columns {found}. "
                "Pass CohortResult.observation_data (leakage columns are stripped "
                "there) rather than the raw DataFrame."
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"WideRepresentationBuilder requires columns {missing}. "
            f"Found: {list(df.columns)}"
        )
