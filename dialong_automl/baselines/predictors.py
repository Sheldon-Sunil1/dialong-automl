"""Fixed reference baselines B0-B3 for DiaLong-AutoML (Phase 13C).

These are deliberately simple comparators, reported next to the AutoML-selected
model. They are NOT registered in ``dialong_automl.models.REGISTRY``, so they
can never become Optuna candidates or the selected model.

All baselines consume the wide representation BEFORE ``WidePreprocessor``
(``WideRepresentationResult.X``), so named columns such as ``hba1c_latest``
keep their raw clinical units. B2 additionally reads the long-format
observation visits to estimate an HbA1c slope per calendar day.

Every learned quantity (prevalence, imputation medians, preprocessing
statistics, regression coefficients) comes only from the data passed to
``fit``. Nothing is tuned: all parameters are fixed in advance.

Baselines
---------
B0 ``b0_prevalence``          constant = prevalence of the fit data.
B1 ``b1_last_hba1c``          score = most recent observed HbA1c (%).
B2 ``b2_trend_extrapolation`` score = last HbA1c + (HbA1c slope per day) * horizon days.
B3 ``b3_logreg_*``            L2 logistic regression, fixed C, on simple aggregates
                              (variants: ``all``, ``hba1c_only``, ``no_hba1c``).

B1 and B2 return raw HbA1c-scale scores, not probabilities
(``is_probabilistic = False``); the Brier score is not defined for them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dialong_automl.features.preprocessing import WidePreprocessor
from dialong_automl.features.wide import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from dialong_automl.models.logistic_regression import LogisticRegressionModel

HBA1C_LATEST = "hba1c_latest"

#: Numeric aggregates used by B3 for every lab / vital (not for age).
B3_NUMERIC_AGGREGATES: tuple[str, ...] = ("_latest", "_mean", "_min", "_max", "_slope")
#: Age is a demographic: only its most recent value is used.
B3_AGE_COLUMNS: tuple[str, ...] = ("age_at_visit_latest",)
B3_VARIANTS: tuple[str, ...] = ("all", "hba1c_only", "no_hba1c")


@dataclass
class BaselineInputs:
    """Everything a baseline needs for one patient partition.

    ``wide_X`` rows, ``patient_ids`` and ``y`` must be in the same order
    (the order produced by ``WideRepresentationBuilder``).
    """

    wide_X: pd.DataFrame
    patient_ids: list[str]
    obs_df: pd.DataFrame
    y: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.wide_X) != len(self.patient_ids):
            raise ValueError(
                f"wide_X has {len(self.wide_X)} rows but {len(self.patient_ids)} patient ids."
            )
        if self.y is not None:
            self.y = np.asarray(self.y).astype(np.int32)
            if len(self.y) != len(self.patient_ids):
                raise ValueError(
                    f"y has {len(self.y)} labels but {len(self.patient_ids)} patient ids."
                )


class Baseline(ABC):
    """Common interface. ``fit`` learns only from its own inputs."""

    name: str = ""
    code: str = ""
    description: str = ""
    is_probabilistic: bool = True

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, data: BaselineInputs) -> Baseline:
        """Fit on *data* (labels required)."""

    @abstractmethod
    def predict_scores(self, data: BaselineInputs) -> np.ndarray:
        """Return one score per patient, in ``data.patient_ids`` order."""

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """JSON-safe description of fixed and fitted parameters."""

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"Baseline {self.name!r} must be fitted before scoring.")

    @staticmethod
    def _require_labels(data: BaselineInputs) -> np.ndarray:
        if data.y is None:
            raise ValueError("Baseline fit requires labels (BaselineInputs.y).")
        return data.y


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _require_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"Wide representation is missing required column {col!r}.")


def _train_median(df: pd.DataFrame, col: str) -> float:
    _require_column(df, col)
    vals = _numeric(df[col]).dropna()
    if vals.empty:
        raise ValueError(f"Cannot fit: column {col!r} has no observed values in the fit data.")
    return float(vals.median())


# ---------------------------------------------------------------------------
# B0
# ---------------------------------------------------------------------------

class PrevalenceBaseline(Baseline):
    """B0: predicts the prevalence of the fit data for every patient."""

    name = "b0_prevalence"
    code = "B0"
    description = "Constant predictor: prevalence of the data it was fitted on."
    is_probabilistic = True

    def __init__(self) -> None:
        super().__init__()
        self.prevalence_: float | None = None

    def fit(self, data: BaselineInputs) -> PrevalenceBaseline:
        y = self._require_labels(data)
        if len(y) == 0:
            raise ValueError("Cannot fit B0 on an empty partition.")
        self.prevalence_ = float(np.mean(y))
        self._fitted = True
        return self

    def predict_scores(self, data: BaselineInputs) -> np.ndarray:
        self._require_fitted()
        return np.full(len(data.patient_ids), self.prevalence_, dtype=np.float64)

    def params(self) -> dict[str, Any]:
        return {"prevalence": self.prevalence_}


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------

class LastHbA1cBaseline(Baseline):
    """B1: score = most recent observed HbA1c (``hba1c_latest``, in %).

    Patients with no observed HbA1c receive the median ``hba1c_latest`` of the
    fit data, so no patient is dropped.
    """

    name = "b1_last_hba1c"
    code = "B1"
    description = "Rule: most recent observed HbA1c (%), thresholded on validation."
    is_probabilistic = False

    def __init__(self) -> None:
        super().__init__()
        self.fill_value_: float | None = None

    def fit(self, data: BaselineInputs) -> LastHbA1cBaseline:
        self._require_labels(data)
        self.fill_value_ = _train_median(data.wide_X, HBA1C_LATEST)
        self._fitted = True
        return self

    def predict_scores(self, data: BaselineInputs) -> np.ndarray:
        self._require_fitted()
        _require_column(data.wide_X, HBA1C_LATEST)
        return _numeric(data.wide_X[HBA1C_LATEST]).fillna(self.fill_value_).to_numpy(np.float64)

    def params(self) -> dict[str, Any]:
        return {"score": HBA1C_LATEST, "missing_hba1c_fill_value": self.fill_value_}


# ---------------------------------------------------------------------------
# B2
# ---------------------------------------------------------------------------

def hba1c_slope_per_day(obs_df: pd.DataFrame, patient_ids: list[str]) -> np.ndarray:
    """OLS slope of observed HbA1c against elapsed days, per patient.

    Uses only non-missing HbA1c readings. Returns 0.0 for a patient with fewer
    than two readings or all readings on the same day (slope undefined).
    Output order follows *patient_ids*; patients absent from *obs_df* get 0.0.
    """
    for col in ("patient_id", "visit_date", "hba1c"):
        if col not in obs_df.columns:
            raise ValueError(f"Observation data is missing required column {col!r}.")
    df = obs_df[["patient_id", "visit_date", "hba1c"]].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df["hba1c"] = _numeric(df["hba1c"])
    df = df.dropna(subset=["hba1c"])

    slopes: dict[str, float] = {}
    for pid, grp in df.groupby("patient_id", sort=False):
        grp = grp.sort_values("visit_date")
        elapsed = grp["visit_date"] - grp["visit_date"].iloc[0]
        days = elapsed.dt.total_seconds().to_numpy() / 86400.0
        vals = grp["hba1c"].to_numpy(np.float64)
        if len(vals) < 2:
            slopes[pid] = 0.0
            continue
        dx = days - days.mean()
        sxx = float(np.sum(dx * dx))
        slopes[pid] = float(np.sum(dx * (vals - vals.mean())) / sxx) if sxx > 0 else 0.0
    return np.array([slopes.get(str(p), 0.0) for p in patient_ids], dtype=np.float64)


class TrendExtrapolationBaseline(Baseline):
    """B2: score = last HbA1c + (OLS HbA1c slope per calendar day) * horizon_days.

    The slope estimator (OLS on elapsed days over non-missing readings) and the
    horizon are fixed in advance. Missing last HbA1c uses the fit-data median.
    """

    name = "b2_trend_extrapolation"
    code = "B2"
    description = (
        "Rule: last HbA1c + OLS HbA1c slope per day x horizon days (%), "
        "thresholded on validation."
    )
    is_probabilistic = False

    def __init__(self, horizon_days: int) -> None:
        super().__init__()
        if horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {horizon_days}.")
        self.horizon_days = int(horizon_days)
        self.fill_value_: float | None = None

    def fit(self, data: BaselineInputs) -> TrendExtrapolationBaseline:
        self._require_labels(data)
        self.fill_value_ = _train_median(data.wide_X, HBA1C_LATEST)
        self._fitted = True
        return self

    def predict_scores(self, data: BaselineInputs) -> np.ndarray:
        self._require_fitted()
        _require_column(data.wide_X, HBA1C_LATEST)
        last = _numeric(data.wide_X[HBA1C_LATEST]).fillna(self.fill_value_).to_numpy(np.float64)
        slope = hba1c_slope_per_day(data.obs_df, data.patient_ids)
        return last + slope * self.horizon_days

    def params(self) -> dict[str, Any]:
        return {
            "score": "hba1c_latest + slope_per_day * horizon_days",
            "slope_estimator": "OLS of observed HbA1c on elapsed days (0 if < 2 readings)",
            "horizon_days": self.horizon_days,
            "missing_hba1c_fill_value": self.fill_value_,
        }


# ---------------------------------------------------------------------------
# B3
# ---------------------------------------------------------------------------

def b3_columns(variant: str) -> tuple[list[str], list[str]]:
    """Pre-specified (numeric, categorical) wide columns for a B3 variant."""
    if variant not in B3_VARIANTS:
        raise ValueError(f"Unknown B3 variant {variant!r}; expected one of {B3_VARIANTS}.")
    labs = [v for v in NUMERIC_FEATURES if v != "age_at_visit"]
    hba1c = [f"hba1c{s}" for s in B3_NUMERIC_AGGREGATES]
    if variant == "hba1c_only":
        return hba1c, []
    numeric = [f"{v}{s}" for v in labs for s in B3_NUMERIC_AGGREGATES] + list(B3_AGE_COLUMNS)
    categorical = [f"{c}_latest" for c in CATEGORICAL_FEATURES]
    if variant == "no_hba1c":
        numeric = [c for c in numeric if not c.startswith("hba1c")]
    return numeric, categorical


class AggregateLogisticBaseline(Baseline):
    """B3: L2 logistic regression with fixed C on pre-specified wide aggregates.

    Preprocessing reuses the project's ``WidePreprocessor`` (median impute,
    standard-scale, one-hot with ``<MISSING>``) restricted to the variant's
    columns, fitted on the fit data only. The classifier is the project's
    ``LogisticRegressionModel`` with fixed hyperparameters (no search).
    """

    code = "B3"
    is_probabilistic = True

    def __init__(self, variant: str = "all", C: float = 1.0, max_iter: int = 2000,
                 seed: int = 42) -> None:
        super().__init__()
        self.variant = variant
        self.numeric_columns, self.categorical_columns = b3_columns(variant)
        if not self.numeric_columns and not self.categorical_columns:
            raise ValueError(f"B3 variant {variant!r} selects no columns.")
        if not C > 0:
            raise ValueError(f"C must be positive, got {C}.")
        self.hyperparams = {
            "C": float(C), "class_weight": None, "solver": "lbfgs",
            "max_iter": int(max_iter), "random_state": int(seed),
        }
        self.name = f"b3_logreg_{variant}"
        self.description = {
            "all": "Logistic regression (fixed C) on HbA1c, other-lab, age and "
                   "categorical aggregates.",
            "hba1c_only": "Logistic regression (fixed C) on HbA1c aggregates only (ablation).",
            "no_hba1c": "Logistic regression (fixed C) on all aggregates except HbA1c (ablation).",
        }[variant]
        self._prep: WidePreprocessor | None = None
        self._model: LogisticRegressionModel | None = None

    @property
    def columns(self) -> list[str]:
        return self.numeric_columns + self.categorical_columns

    def _frame(self, data: BaselineInputs) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in data.wide_X.columns]
        if missing:
            raise ValueError(f"Wide representation is missing B3 columns {missing}.")
        return data.wide_X[self.columns]

    def fit(self, data: BaselineInputs) -> AggregateLogisticBaseline:
        y = self._require_labels(data)
        if len(np.unique(y)) < 2:
            raise ValueError(f"{self.name}: fit data must contain both classes.")
        self._prep = WidePreprocessor(
            numeric_columns=list(self.numeric_columns),
            categorical_columns=list(self.categorical_columns),
        )
        X = self._prep.fit_transform(self._frame(data))
        self._model = LogisticRegressionModel(dict(self.hyperparams)).fit(X, y)
        self._fitted = True
        return self

    def predict_scores(self, data: BaselineInputs) -> np.ndarray:
        self._require_fitted()
        X = self._prep.transform(self._frame(data))
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    def params(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "hyperparameters": dict(self.hyperparams),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "preprocessing": "WidePreprocessor fitted on the fit data only",
        }


def default_baselines(horizon_days: int, C: float = 1.0, max_iter: int = 2000,
                      seed: int = 42) -> list[Baseline]:
    """The pre-specified baseline set B0, B1, B2, B3 (all, hba1c_only, no_hba1c)."""
    return [
        PrevalenceBaseline(),
        LastHbA1cBaseline(),
        TrendExtrapolationBaseline(horizon_days=horizon_days),
        *[AggregateLogisticBaseline(variant=v, C=C, max_iter=max_iter, seed=seed)
          for v in B3_VARIANTS],
    ]
