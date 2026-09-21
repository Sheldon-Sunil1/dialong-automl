"""Dataset profiler for DiaLong-AutoML longitudinal datasets.

Produces a machine-readable :class:`DataProfile` containing:
- row/patient/visit counts
- numeric summary statistics
- categorical value distributions
- missingness per column
- date range
- target distribution (when target column is present)

All values come from actual data — nothing is fabricated or estimated.

Usage::

    from dialong_automl.data.profiler import DataProfiler

    profiler = DataProfiler()
    profile = profiler.profile(df)
    print(profile.to_dict())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "sex",
    "smoking_status",
    "physical_activity_level",
    "family_history_diabetes",
    "metformin_use",
    "hypertension_diagnosis",
]


@dataclass
class NumericSummary:
    """Summary statistics for one numeric column."""

    column: str
    count: int
    missing: int
    missing_rate: float
    mean: float
    std: float
    min: float
    p25: float
    median: float
    p75: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "count": self.count,
            "missing": self.missing,
            "missing_rate": self.missing_rate,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "p25": self.p25,
            "median": self.median,
            "p75": self.p75,
            "max": self.max,
        }


@dataclass
class CategoricalSummary:
    """Summary for one categorical column."""

    column: str
    n_unique: int
    missing: int
    missing_rate: float
    value_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "n_unique": self.n_unique,
            "missing": self.missing,
            "missing_rate": self.missing_rate,
            "value_counts": self.value_counts,
        }


@dataclass
class DataProfile:
    """Full profile of a loaded longitudinal dataset.

    All counts and statistics are derived from the actual data passed to
    :meth:`DataProfiler.profile`.
    """

    # Basic counts
    n_rows: int
    n_columns: int
    n_patients: int

    # Visit statistics
    visits_per_patient_min: int
    visits_per_patient_max: int
    visits_per_patient_median: float
    visits_per_patient_mean: float

    # Date range
    date_min: str | None
    date_max: str | None

    # Numeric summaries
    numeric_summaries: list[NumericSummary] = field(default_factory=list)

    # Categorical summaries
    categorical_summaries: list[CategoricalSummary] = field(default_factory=list)

    # Target distribution
    target_present: bool = False
    n_positive_patients: int | None = None
    n_negative_patients: int | None = None
    n_unknown_target_patients: int | None = None
    positive_rate: float | None = None

    # Overall missingness per column
    missingness: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the full profile as a plain Python dict."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_patients": self.n_patients,
            "visits_per_patient": {
                "min": self.visits_per_patient_min,
                "max": self.visits_per_patient_max,
                "median": self.visits_per_patient_median,
                "mean": self.visits_per_patient_mean,
            },
            "date_range": {
                "min": self.date_min,
                "max": self.date_max,
            },
            "numeric_summaries": [s.to_dict() for s in self.numeric_summaries],
            "categorical_summaries": [s.to_dict() for s in self.categorical_summaries],
            "target": {
                "present": self.target_present,
                "n_positive_patients": self.n_positive_patients,
                "n_negative_patients": self.n_negative_patients,
                "n_unknown_target_patients": self.n_unknown_target_patients,
                "positive_rate": self.positive_rate,
            },
            "missingness": self.missingness,
        }

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print(f"=== DataProfile ===")
        print(f"Rows: {self.n_rows:,}  |  Patients: {self.n_patients:,}  |  Columns: {self.n_columns}")
        print(
            f"Visits/patient: min={self.visits_per_patient_min} "
            f"median={self.visits_per_patient_median} "
            f"max={self.visits_per_patient_max}"
        )
        if self.date_min:
            print(f"Date range: {self.date_min} → {self.date_max}")
        if self.target_present:
            print(
                f"Target: {self.n_positive_patients} positive, "
                f"{self.n_negative_patients} negative, "
                f"rate={self.positive_rate:.3f}"
            )
        if self.missingness:
            print("Missingness (columns with missing values):")
            for col, rate in sorted(self.missingness.items(), key=lambda x: -x[1]):
                print(f"  {col}: {rate:.2%}")
        print("Numeric summaries:")
        for ns in self.numeric_summaries:
            print(
                f"  {ns.column:25s}  mean={ns.mean:7.2f}  std={ns.std:6.2f}  "
                f"[{ns.min:.1f}, {ns.max:.1f}]  missing={ns.missing_rate:.1%}"
            )


class DataProfiler:
    """Compute a structured :class:`DataProfile` from a longitudinal DataFrame."""

    def profile(self, df: pd.DataFrame) -> DataProfile:
        """Profile *df* and return a :class:`DataProfile`.

        Parameters
        ----------
        df:
            Loaded longitudinal DataFrame.

        Returns
        -------
        DataProfile
        """
        n_rows = len(df)
        n_cols = len(df.columns)

        # Patient count
        n_patients = int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0

        # Visits per patient
        if "patient_id" in df.columns:
            vpp = df.groupby("patient_id").size()
            vpp_min = int(vpp.min())
            vpp_max = int(vpp.max())
            vpp_median = float(vpp.median())
            vpp_mean = round(float(vpp.mean()), 2)
        else:
            vpp_min = vpp_max = 0
            vpp_median = vpp_mean = 0.0

        # Date range
        if "visit_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["visit_date"]):
            date_min = str(df["visit_date"].min().date())
            date_max = str(df["visit_date"].max().date())
        else:
            date_min = date_max = None

        # Numeric summaries
        numeric_summaries: list[NumericSummary] = []
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            n_miss = int(series.isna().sum())
            n_valid = int(series.notna().sum())
            if n_valid == 0:
                continue
            numeric_summaries.append(
                NumericSummary(
                    column=col,
                    count=n_valid,
                    missing=n_miss,
                    missing_rate=round(n_miss / n_rows, 4) if n_rows > 0 else 0.0,
                    mean=round(float(series.mean()), 4),
                    std=round(float(series.std()), 4),
                    min=round(float(series.min()), 4),
                    p25=round(float(series.quantile(0.25)), 4),
                    median=round(float(series.median()), 4),
                    p75=round(float(series.quantile(0.75)), 4),
                    max=round(float(series.max()), 4),
                )
            )

        # Categorical summaries
        categorical_summaries: list[CategoricalSummary] = []
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            series = df[col].astype(str)
            null_mask = df[col].isna() | (series.str.lower().isin(["nan", "none", "<na>"]))
            n_miss = int(null_mask.sum())
            clean = series[~null_mask]
            vc = clean.value_counts().to_dict()
            categorical_summaries.append(
                CategoricalSummary(
                    column=col,
                    n_unique=int(clean.nunique()),
                    missing=n_miss,
                    missing_rate=round(n_miss / n_rows, 4) if n_rows > 0 else 0.0,
                    value_counts={str(k): int(v) for k, v in vc.items()},
                )
            )

        # Target distribution (patient-level)
        target_present = "target" in df.columns
        n_pos = n_neg = n_unk = None
        pos_rate = None
        if target_present and "patient_id" in df.columns:
            pt = df.drop_duplicates("patient_id")[["patient_id", "target"]]
            n_pos = int((pt["target"] == 1).sum())
            n_neg = int((pt["target"] == 0).sum())
            n_unk = int(pt["target"].isna().sum())
            labeled = n_pos + n_neg
            pos_rate = round(n_pos / labeled, 4) if labeled > 0 else None

        # Overall missingness
        missingness: dict[str, float] = {}
        for col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                missingness[col] = round(n_null / n_rows, 4)

        profile = DataProfile(
            n_rows=n_rows,
            n_columns=n_cols,
            n_patients=n_patients,
            visits_per_patient_min=vpp_min,
            visits_per_patient_max=vpp_max,
            visits_per_patient_median=vpp_median,
            visits_per_patient_mean=vpp_mean,
            date_min=date_min,
            date_max=date_max,
            numeric_summaries=numeric_summaries,
            categorical_summaries=categorical_summaries,
            target_present=target_present,
            n_positive_patients=n_pos,
            n_negative_patients=n_neg,
            n_unknown_target_patients=n_unk,
            positive_rate=pos_rate,
            missingness=missingness,
        )

        logger.info(
            "Profiled %d rows, %d patients, %d numeric summaries",
            n_rows, n_patients, len(numeric_summaries),
        )
        return profile
