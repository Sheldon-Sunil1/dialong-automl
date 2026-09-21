"""Cohort builder for DiaLong-AutoML.

The most critical invariant enforced here:

    NO FUTURE DATA MAY ENTER OBSERVATION FEATURES.

For every patient the cohort builder guarantees::

    OBSERVATION DATA:   visit_1, visit_2, ..., visit_N   (≤ cutoff)
                                                  ↑
                                               cutoff
    FUTURE DATA:        visit_N+1, ...                   (> cutoff)
      └─ used ONLY to derive target; NEVER returned as observation rows.

Canonical mode
--------------
**FUTURE_OUTCOME is the canonical mode for the longitudinal research
experiment.**  It must be used whenever targets are derived from real
(or realistic synthetic) follow-up observations, because it:

* constructs the target solely from post-cutoff visits,
* applies the ``prediction_horizon_days`` window relative to each
  patient's individual observation cutoff date,
* and guarantees that no future observation ever enters the feature space.

PRE_LABELED mode is provided only for:

* quick unit-testing with pre-constructed targets, and
* backward-compatibility with datasets that already carry a stable
  patient-level label.

It must NOT be used as a substitute for FUTURE_OUTCOME on the synthetic
full timeline, because the pre-labeled ``target`` column in the
observation-only CSV is derived from *all* future visits without a
horizon cutoff, whereas FUTURE_OUTCOME applies the configurable
``prediction_horizon_days`` window.  Silently using PRE_LABELED targets
on a dataset built for FUTURE_OUTCOME would introduce a
horizon-mismatch that corrupts the experiment.

Target definition (FUTURE_OUTCOME, canonical)
---------------------------------------------
For each eligible patient::

    cutoff_date   = visit_date of the last observation-window visit
    horizon_end   = cutoff_date + prediction_horizon_days   (calendar days)

    target = 1  iff any future visit v satisfies:
                    v.visit_date  >  cutoff_date
                AND v.visit_date  <=  horizon_end
                AND (v.diabetes_diagnosis == 1
                     OR v.hba1c >= diabetes_onset_hba1c_threshold)

    target = 0  when sufficient follow-up exists within the horizon and
                neither criterion is met in any future visit.

    patient excluded  when no future visit falls within the horizon
                      (outcome cannot be determined).

Temporal leakage protection
---------------------------
The following mechanisms are in place and are covered by the test suite
in ``tests/test_no_temporal_leakage.py``:

1. Future rows are identified (via ``is_future_visit`` flag or by
   positional split at ``max_observation_visits``) before any
   target derivation occurs.
2. A hard runtime check raises ``RuntimeError`` if any observation row
   has ``visit_date > cutoff_date``.
3. :func:`_strip_leakage_columns` removes ``target``,
   ``is_future_visit``, and all ``_FUTURE_LEAKAGE_COLUMNS`` from
   ``CohortResult.observation_data`` before it is returned.
4. :func:`get_safe_observation_columns` provides an explicit whitelist
   of columns safe for model input.

Two modes are supported:

Mode A — PRE-LABELED
    Input already contains a stable patient-level ``target`` column.
    The builder verifies consistency and assembles the cohort.
    Use only for testing or datasets with pre-existing stable labels.

Mode B — FUTURE-OUTCOME CONSTRUCTION  ← canonical for this project
    Derives targets from post-cutoff visit data within the prediction
    horizon.  Strictly separates observation rows from future rows in
    the returned objects.

Usage::

    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode

    # Canonical usage — future-outcome mode
    cfg = CohortConfig(mode=CohortMode.FUTURE_OUTCOME)
    builder = CohortBuilder(cfg)
    result = builder.build(full_df)   # full_df from load_longitudinal_csv

    result.observation_data    # pd.DataFrame — model features (leakage-free)
    result.patient_targets     # pd.Series   — patient_id -> target (0 or 1)
    result.future_outcome_audit  # pd.DataFrame — diagnostics only, not features
    result.audit               # CohortAudit — exclusion counts & config echo
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leakage-sensitive columns that must NEVER appear in observation features
# ---------------------------------------------------------------------------

_FUTURE_LEAKAGE_COLUMNS: frozenset[str] = frozenset({
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


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CohortMode(str, Enum):
    """Cohort construction mode."""

    PRE_LABELED = "pre_labeled"
    FUTURE_OUTCOME = "future_outcome"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CohortConfig:
    """Configuration for :class:`CohortBuilder`.

    Parameters
    ----------
    mode:
        ``pre_labeled`` or ``future_outcome``.
    min_observation_visits:
        Minimum number of observation-window visits required.
    max_observation_visits:
        Maximum observation-window visits used (earlier visits truncated
        from the start when a patient has more).
    prediction_horizon_days:
        Days after the cutoff date during which future outcomes count.
    diabetes_onset_hba1c_threshold:
        HbA1c value (%) at or above which a future visit is treated as a
        diabetes-onset event.
    exclude_known_diabetes_at_baseline:
        When ``True``, patients who already have ``diabetes_diagnosis=1``
        in any observation visit are excluded.
    require_followup_after_cutoff:
        When ``True``, patients with no future visits beyond the cutoff are
        excluded (outcome cannot be determined).
    target_inconsistency_as_error:
        For pre-labeled mode: whether inconsistent targets raise an error
        (``True``) or just emit a warning (``False``).
    """

    mode: CohortMode = CohortMode.FUTURE_OUTCOME
    min_observation_visits: int = 3
    max_observation_visits: int = 4
    prediction_horizon_days: int = 365
    diabetes_onset_hba1c_threshold: float = 6.5
    exclude_known_diabetes_at_baseline: bool = True
    require_followup_after_cutoff: bool = True
    target_inconsistency_as_error: bool = True


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@dataclass
class CohortAudit:
    """Audit trail for a cohort-building run.

    All counts reflect actual patients processed, not estimates.
    """

    mode: str
    input_patients: int
    eligible_patients: int
    excluded_prior_diabetes: int
    excluded_insufficient_visits: int
    excluded_insufficient_followup: int
    excluded_inconsistent_target: int
    positive_patients: int
    negative_patients: int
    positive_rate: float

    # Cohort configuration echoed back for traceability
    min_observation_visits: int
    max_observation_visits: int
    prediction_horizon_days: int
    diabetes_onset_hba1c_threshold: float
    exclude_known_diabetes_at_baseline: bool

    # Per-patient diagnostics
    observation_visit_counts: dict[str, int] = field(default_factory=dict)
    cutoff_dates: dict[str, str] = field(default_factory=dict)

    # Target construction criteria (human-readable)
    target_construction_criteria: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "input_patients": self.input_patients,
            "eligible_patients": self.eligible_patients,
            "excluded_prior_diabetes": self.excluded_prior_diabetes,
            "excluded_insufficient_visits": self.excluded_insufficient_visits,
            "excluded_insufficient_followup": self.excluded_insufficient_followup,
            "excluded_inconsistent_target": self.excluded_inconsistent_target,
            "positive_patients": self.positive_patients,
            "negative_patients": self.negative_patients,
            "positive_rate": self.positive_rate,
            "min_observation_visits": self.min_observation_visits,
            "max_observation_visits": self.max_observation_visits,
            "prediction_horizon_days": self.prediction_horizon_days,
            "diabetes_onset_hba1c_threshold": self.diabetes_onset_hba1c_threshold,
            "exclude_known_diabetes_at_baseline": self.exclude_known_diabetes_at_baseline,
            "target_construction_criteria": self.target_construction_criteria,
        }

    def print_summary(self) -> None:
        print("=== CohortAudit ===")
        print(f"  Mode                       : {self.mode}")
        print(f"  Input patients             : {self.input_patients}")
        print(f"  Eligible patients          : {self.eligible_patients}")
        print(f"  Excluded prior diabetes    : {self.excluded_prior_diabetes}")
        print(f"  Excluded insufficient visits: {self.excluded_insufficient_visits}")
        print(f"  Excluded insufficient follow-up: {self.excluded_insufficient_followup}")
        print(f"  Excluded inconsistent target: {self.excluded_inconsistent_target}")
        print(f"  Positive patients          : {self.positive_patients}")
        print(f"  Negative patients          : {self.negative_patients}")
        print(f"  Positive rate              : {self.positive_rate:.4f}")
        print(f"  Target criteria            : {self.target_construction_criteria}")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CohortResult:
    """Output of :meth:`CohortBuilder.build`.

    Attributes
    ----------
    observation_data:
        DataFrame containing **only** observation-window rows (visit_date
        <= cutoff_date).  All leakage-sensitive columns (``target``,
        ``is_future_visit``, and every name in ``_FUTURE_LEAKAGE_COLUMNS``)
        are stripped before this object is returned.

        **This is the only DataFrame that should be passed to a model.**
        It contains no information about future events.

    patient_targets:
        Series indexed by ``patient_id`` with integer target (0 or 1).
        In FUTURE_OUTCOME mode these are derived exclusively from
        post-cutoff visits within ``prediction_horizon_days`` of the
        cutoff date.  In PRE_LABELED mode they are read from the input
        ``target`` column after consistency verification.

    future_outcome_audit:
        DataFrame with per-patient future-visit details used for
        diagnostic inspection.  **Must never be used as model features.**
        Contains columns such as ``future_hba1c_max`` and
        ``future_diabetes_diagnosis`` that would constitute leakage.

    audit:
        :class:`CohortAudit` with exclusion counts, configuration echo,
        and the human-readable ``target_construction_criteria`` string
        that states exactly which rule was applied.
    """

    observation_data: pd.DataFrame
    patient_targets: pd.Series
    future_outcome_audit: pd.DataFrame
    audit: CohortAudit


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class CohortBuilder:
    """Build a model-ready cohort from a longitudinal dataset.

    Parameters
    ----------
    config:
        Cohort construction configuration.
    """

    def __init__(self, config: CohortConfig | None = None) -> None:
        self.cfg = config or CohortConfig()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> CohortResult:
        """Build the cohort from *df*.

        Parameters
        ----------
        df:
            Output of :func:`~dialong_automl.data.loader.load_longitudinal_csv`.
            Must contain ``patient_id`` and ``visit_date``.

        Returns
        -------
        CohortResult
        """
        _require_columns(df, ["patient_id", "visit_date"])

        # Ensure visit_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["visit_date"]):
            df = df.copy()
            df["visit_date"] = pd.to_datetime(df["visit_date"])

        logger.info(
            "Building cohort: mode=%s, patients=%d",
            self.cfg.mode.value,
            df["patient_id"].nunique(),
        )

        if self.cfg.mode == CohortMode.PRE_LABELED:
            return self._build_pre_labeled(df)
        else:
            return self._build_future_outcome(df)

    # ------------------------------------------------------------------
    # Mode A: pre-labeled
    # ------------------------------------------------------------------

    def _build_pre_labeled(self, df: pd.DataFrame) -> CohortResult:
        """Build cohort from a dataset with a pre-existing patient-level target.

        .. warning::
            PRE_LABELED mode reads the ``target`` column directly from the
            input DataFrame.  It does **not** apply ``prediction_horizon_days``
            or any temporal cutoff to derive the target.

            Do not use PRE_LABELED mode on the synthetic full-timeline CSV
            as a shortcut for FUTURE_OUTCOME.  The pre-labeled ``target``
            column in that file is derived from *all* future visits without
            a horizon window; FUTURE_OUTCOME applies the configurable
            ``prediction_horizon_days`` relative to each patient's cutoff.
            Mixing modes silently introduces a horizon mismatch.

            Use PRE_LABELED only for:
            * unit-testing with hand-crafted targets, or
            * real-world datasets that carry a clinically verified
              patient-level label already.
        """
        _require_columns(df, ["target"])

        input_patients = df["patient_id"].nunique()
        n_excluded_prior_diabetes = 0
        n_excluded_visits = 0
        n_excluded_followup = 0
        n_excluded_inconsistent = 0

        obs_rows: list[pd.DataFrame] = []
        targets: dict[str, int] = {}
        audit_rows: list[dict[str, Any]] = []

        for pid, grp in df.groupby("patient_id", sort=False):
            grp = grp.sort_values("visit_date").reset_index(drop=True)

            # --- Check observation visit count ---
            if len(grp) < self.cfg.min_observation_visits:
                n_excluded_visits += 1
                continue

            # --- Target consistency ---
            non_null_targets = grp["target"].dropna().unique()
            if len(non_null_targets) > 1:
                msg = (
                    f"Patient {pid!r} has inconsistent target values "
                    f"{non_null_targets.tolist()} across visits."
                )
                if self.cfg.target_inconsistency_as_error:
                    raise ValueError(msg)
                logger.warning(msg)
                n_excluded_inconsistent += 1
                continue

            if len(non_null_targets) == 0:
                # No target label at all — skip
                n_excluded_followup += 1
                continue

            patient_target = int(non_null_targets[0])

            # --- Exclude known diabetes at baseline ---
            if self.cfg.exclude_known_diabetes_at_baseline and "diabetes_diagnosis" in grp.columns:
                obs_grp = grp.iloc[: self.cfg.max_observation_visits]
                if (obs_grp["diabetes_diagnosis"] == 1).any():
                    n_excluded_prior_diabetes += 1
                    continue

            # --- Select observation visits ---
            obs_grp = grp.iloc[: self.cfg.max_observation_visits].copy()

            # Strip leakage columns
            obs_grp = _strip_leakage_columns(obs_grp)

            targets[str(pid)] = patient_target
            obs_rows.append(obs_grp)
            audit_rows.append({
                "patient_id": str(pid),
                "target": patient_target,
                "n_obs_visits": len(obs_grp),
                "cutoff_date": str(obs_grp["visit_date"].max().date()),
                "mode": "pre_labeled",
            })

        obs_df = pd.concat(obs_rows, ignore_index=True) if obs_rows else pd.DataFrame()
        target_series = pd.Series(targets, name="target")
        target_series.index.name = "patient_id"
        audit_df = pd.DataFrame(audit_rows)

        eligible = len(targets)
        n_pos = int((target_series == 1).sum())
        n_neg = int((target_series == 0).sum())
        pos_rate = round(n_pos / (n_pos + n_neg), 4) if (n_pos + n_neg) > 0 else 0.0

        audit = CohortAudit(
            mode="pre_labeled",
            input_patients=input_patients,
            eligible_patients=eligible,
            excluded_prior_diabetes=n_excluded_prior_diabetes,
            excluded_insufficient_visits=n_excluded_visits,
            excluded_insufficient_followup=n_excluded_followup,
            excluded_inconsistent_target=n_excluded_inconsistent,
            positive_patients=n_pos,
            negative_patients=n_neg,
            positive_rate=pos_rate,
            min_observation_visits=self.cfg.min_observation_visits,
            max_observation_visits=self.cfg.max_observation_visits,
            prediction_horizon_days=self.cfg.prediction_horizon_days,
            diabetes_onset_hba1c_threshold=self.cfg.diabetes_onset_hba1c_threshold,
            exclude_known_diabetes_at_baseline=self.cfg.exclude_known_diabetes_at_baseline,
            target_construction_criteria=(
                "Pre-labeled: target taken directly from input dataset, "
                "verified for within-patient consistency."
            ),
        )

        logger.info(
            "Pre-labeled cohort: %d eligible / %d input, pos_rate=%.3f",
            eligible, input_patients, pos_rate,
        )
        return CohortResult(
            observation_data=obs_df,
            patient_targets=target_series,
            future_outcome_audit=audit_df,
            audit=audit,
        )

    # ------------------------------------------------------------------
    # Mode B: future-outcome construction
    # ------------------------------------------------------------------

    def _build_future_outcome(self, df: pd.DataFrame) -> CohortResult:
        """Derive targets from future visits; strictly protect observation data.

        This is the **canonical mode** for the DiaLong-AutoML longitudinal
        research experiment.

        Target rule applied per patient::

            cutoff_date = last observation visit date
            horizon_end = cutoff_date + prediction_horizon_days

            target = 1  iff any future visit v where
                            v.visit_date > cutoff_date
                        AND v.visit_date <= horizon_end
                        AND (v.diabetes_diagnosis == 1
                             OR v.hba1c >= diabetes_onset_hba1c_threshold)

            target = 0  otherwise (no qualifying event within horizon)
            excluded    when no future visit falls within the horizon

        The ``horizon_end`` is computed with :class:`pandas.DateOffset` so
        the window is always measured in calendar days relative to each
        patient's individual cutoff date.

        Leakage protection applied before returning:

        * Future rows are never included in ``observation_data``.
        * A hard ``RuntimeError`` is raised if any observation row has
          ``visit_date > cutoff_date`` (programming-error guard).
        * :func:`_strip_leakage_columns` removes ``target``,
          ``is_future_visit``, and all ``_FUTURE_LEAKAGE_COLUMNS`` from
          ``observation_data``.
        """
        input_patients = df["patient_id"].nunique()
        n_excluded_prior_diabetes = 0
        n_excluded_visits = 0
        n_excluded_followup = 0
        n_excluded_inconsistent = 0

        obs_rows: list[pd.DataFrame] = []
        targets: dict[str, int] = {}
        audit_rows: list[dict[str, Any]] = []

        # Determine which rows are "future" -- if is_future_visit is present
        # in the dataframe, use that; otherwise derive from visit ordering.
        # Handle string-encoded booleans ("True"/"False") from CSV round-trips.
        has_future_flag = "is_future_visit" in df.columns

        for pid, grp in df.groupby("patient_id", sort=False):
            grp = grp.sort_values("visit_date").reset_index(drop=True)

            # --- Split into observation vs future rows ---
            if has_future_flag:
                future_flag = grp["is_future_visit"]
                # Normalise: handle bool, int, and string representations
                if future_flag.dtype == object:
                    future_bool = future_flag.astype(str).str.strip().str.lower().map(
                        {"true": True, "1": True, "false": False, "0": False}
                    ).fillna(False).astype(bool)
                else:
                    future_bool = future_flag.astype(bool)
                obs_grp_full = grp[~future_bool].copy()
                future_grp = grp[future_bool].copy()
            else:
                # Without a flag: use max_observation_visits to define cutoff
                n_obs = min(len(grp), self.cfg.max_observation_visits)
                obs_grp_full = grp.iloc[:n_obs].copy()
                future_grp = grp.iloc[n_obs:].copy()

            # --- Minimum observation visits check ---
            if len(obs_grp_full) < self.cfg.min_observation_visits:
                n_excluded_visits += 1
                continue

            # --- Cutoff date: last observation visit date ---
            cutoff_date: pd.Timestamp = obs_grp_full["visit_date"].max()

            # --- Require sufficient future follow-up ---
            if self.cfg.require_followup_after_cutoff:
                future_within_horizon = future_grp[
                    future_grp["visit_date"] > cutoff_date
                ]
                if len(future_within_horizon) == 0:
                    n_excluded_followup += 1
                    continue
            else:
                future_within_horizon = future_grp[
                    future_grp["visit_date"] > cutoff_date
                ]

            # --- Exclude patients with diabetes already at baseline ---
            if self.cfg.exclude_known_diabetes_at_baseline and "diabetes_diagnosis" in obs_grp_full.columns:
                if (obs_grp_full["diabetes_diagnosis"] == 1).any():
                    n_excluded_prior_diabetes += 1
                    continue

            # --- Limit to max_observation_visits (take LAST N for recency) ---
            if len(obs_grp_full) > self.cfg.max_observation_visits:
                obs_grp = obs_grp_full.tail(self.cfg.max_observation_visits).copy()
            else:
                obs_grp = obs_grp_full.copy()

            # --- CRITICAL: verify no future rows bleed into observation ---
            # Any observation row must have visit_date <= cutoff_date
            leaked = obs_grp[obs_grp["visit_date"] > cutoff_date]
            if len(leaked) > 0:
                # This is a programming error — fail loudly
                raise RuntimeError(
                    f"TEMPORAL LEAKAGE DETECTED for patient {pid!r}: "
                    f"{len(leaked)} observation rows have visit_date > cutoff_date "
                    f"({cutoff_date.date()}).  This is a bug in the cohort builder."
                )

            # --- Derive target from future visits ---
            # Filter future visits to those within the prediction horizon
            horizon_end = cutoff_date + pd.DateOffset(days=self.cfg.prediction_horizon_days)
            future_in_horizon = future_within_horizon[
                future_within_horizon["visit_date"] <= horizon_end
            ]

            if len(future_in_horizon) == 0:
                n_excluded_followup += 1
                continue

            patient_target = self._derive_future_target(future_in_horizon)

            # --- CRITICAL: strip leakage columns from observation data ---
            obs_grp_clean = _strip_leakage_columns(obs_grp)

            targets[str(pid)] = patient_target
            obs_rows.append(obs_grp_clean)

            # --- Audit row (future details kept here for diagnostics only) ---
            future_hba1c_max = float(future_in_horizon["hba1c"].max()) \
                if "hba1c" in future_in_horizon.columns else float("nan")
            future_diag_any = bool(
                (future_in_horizon["diabetes_diagnosis"] == 1).any()
            ) if "diabetes_diagnosis" in future_in_horizon.columns else False

            audit_rows.append({
                "patient_id": str(pid),
                "target": patient_target,
                "n_obs_visits": len(obs_grp_clean),
                "cutoff_date": str(cutoff_date.date()),
                "horizon_end_date": str(horizon_end.date()),
                "n_future_visits_in_horizon": len(future_in_horizon),
                "future_hba1c_max": round(future_hba1c_max, 2) if not pd.isna(future_hba1c_max) else None,
                "future_diabetes_diagnosis": future_diag_any,
                "mode": "future_outcome",
            })

        obs_df = pd.concat(obs_rows, ignore_index=True) if obs_rows else pd.DataFrame()
        target_series = pd.Series(targets, name="target")
        target_series.index.name = "patient_id"
        audit_df = pd.DataFrame(audit_rows)

        eligible = len(targets)
        n_pos = int((target_series == 1).sum())
        n_neg = int((target_series == 0).sum())
        pos_rate = round(n_pos / (n_pos + n_neg), 4) if (n_pos + n_neg) > 0 else 0.0

        # Per-patient visit counts and cutoff dates for audit
        obs_visit_counts = obs_df.groupby("patient_id").size().to_dict() if not obs_df.empty else {}
        cutoff_dates_dict: dict[str, str] = {}
        for row in audit_rows:
            cutoff_dates_dict[row["patient_id"]] = row["cutoff_date"]

        audit = CohortAudit(
            mode="future_outcome",
            input_patients=input_patients,
            eligible_patients=eligible,
            excluded_prior_diabetes=n_excluded_prior_diabetes,
            excluded_insufficient_visits=n_excluded_visits,
            excluded_insufficient_followup=n_excluded_followup,
            excluded_inconsistent_target=n_excluded_inconsistent,
            positive_patients=n_pos,
            negative_patients=n_neg,
            positive_rate=pos_rate,
            min_observation_visits=self.cfg.min_observation_visits,
            max_observation_visits=self.cfg.max_observation_visits,
            prediction_horizon_days=self.cfg.prediction_horizon_days,
            diabetes_onset_hba1c_threshold=self.cfg.diabetes_onset_hba1c_threshold,
            exclude_known_diabetes_at_baseline=self.cfg.exclude_known_diabetes_at_baseline,
            observation_visit_counts=obs_visit_counts,
            cutoff_dates=cutoff_dates_dict,
            target_construction_criteria=(
                f"Future outcome: target=1 when within {self.cfg.prediction_horizon_days} days "
                f"of cutoff, diabetes_diagnosis==1 OR hba1c >= {self.cfg.diabetes_onset_hba1c_threshold}. "
                f"target=0 when sufficient follow-up exists and neither criterion met."
            ),
        )

        logger.info(
            "Future-outcome cohort: %d eligible / %d input "
            "(excl: diabetes=%d, visits=%d, followup=%d), pos_rate=%.3f",
            eligible, input_patients,
            n_excluded_prior_diabetes, n_excluded_visits, n_excluded_followup,
            pos_rate,
        )
        return CohortResult(
            observation_data=obs_df,
            patient_targets=target_series,
            future_outcome_audit=audit_df,
            audit=audit,
        )

    # ------------------------------------------------------------------
    # Target derivation
    # ------------------------------------------------------------------

    def _derive_future_target(self, future_df: pd.DataFrame) -> int:
        """Derive binary target from future-horizon rows.

        Target = 1 when, within the horizon:
          - ``diabetes_diagnosis`` == 1  OR
          - ``hba1c`` >= ``diabetes_onset_hba1c_threshold``

        Target = 0 otherwise.
        """
        diag_col = "diabetes_diagnosis"
        hba1c_col = "hba1c"

        diag_positive = False
        hba1c_positive = False

        if diag_col in future_df.columns:
            diag_positive = bool((future_df[diag_col] == 1).any())

        if hba1c_col in future_df.columns:
            hba1c_vals = pd.to_numeric(future_df[hba1c_col], errors="coerce").dropna()
            hba1c_positive = bool((hba1c_vals >= self.cfg.diabetes_onset_hba1c_threshold).any())

        return 1 if (diag_positive or hba1c_positive) else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"CohortBuilder requires columns {missing}. "
            f"Found: {list(df.columns)}"
        )


def _strip_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns known to carry future-outcome information.

    This is the last line of defence against accidental leakage.
    Columns are identified by exact name match against
    :data:`_FUTURE_LEAKAGE_COLUMNS`.
    """
    to_drop = [c for c in df.columns if c in _FUTURE_LEAKAGE_COLUMNS]
    if to_drop:
        logger.debug("Stripping leakage columns from observation data: %s", to_drop)
        df = df.drop(columns=to_drop)
    return df


def get_safe_observation_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that are safe to use as model observation features.

    Strips any column whose name matches a leakage-candidate pattern.

    Parameters
    ----------
    df:
        Any DataFrame (typically ``CohortResult.observation_data``).

    Returns
    -------
    list[str]
        Column names safe for model input.
    """
    from dialong_automl.data.validator import LEAKAGE_PATTERNS

    safe: list[str] = []
    for col in df.columns:
        col_lower = col.lower()
        is_leakage = any(p.lower() in col_lower for p in LEAKAGE_PATTERNS)
        if not is_leakage and col not in _FUTURE_LEAKAGE_COLUMNS:
            safe.append(col)
    return safe
