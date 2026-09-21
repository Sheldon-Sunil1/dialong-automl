"""Tests for the cohort builder (Phase 3).

Covers:
- patients with prior diabetes are excluded
- progression label uses future records only
- insufficient follow-up excludes patient
- insufficient observation visits excludes patient
- observation rows never contain future records
- cutoff date is correct
- prediction horizon is applied correctly
- both modes (pre_labeled, future_outcome)
- audit counts are accurate
"""

from __future__ import annotations

import pandas as pd
import pytest

from dialong_automl.data.cohort import (
    CohortBuilder,
    CohortConfig,
    CohortMode,
    CohortResult,
    _strip_leakage_columns,
    _FUTURE_LEAKAGE_COLUMNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_longitudinal_df(
    patients: dict[str, list[dict]],
) -> pd.DataFrame:
    """Build a long-format DataFrame from a dict of patient → list-of-visit-dicts."""
    rows = []
    for pid, visits in patients.items():
        for v in visits:
            row = {"patient_id": pid}
            row.update(v)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    return df


def _visit(
    date: str,
    hba1c: float = 5.5,
    diabetes_diagnosis: int = 0,
    is_future: bool = False,
    target: int | None = None,
) -> dict:
    row: dict = {
        "visit_date": date,
        "hba1c": hba1c,
        "fasting_glucose": 90.0,
        "bmi": 25.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "ldl_cholesterol": 110.0,
        "hdl_cholesterol": 55.0,
        "triglycerides": 130.0,
        "heart_rate": 72.0,
        "age_at_visit": 45.0,
        "sex": "M",
        "smoking_status": "never",
        "physical_activity_level": "moderate",
        "family_history_diabetes": "no",
        "metformin_use": "no",
        "hypertension_diagnosis": "no",
        "diabetes_diagnosis": diabetes_diagnosis,
        "is_future_visit": is_future,
        "visit_number": 1,
    }
    if target is not None:
        row["target"] = target
    return row


# ---------------------------------------------------------------------------
# 1. Patients with prior diabetes are excluded
# ---------------------------------------------------------------------------

def test_prior_diabetes_patient_excluded() -> None:
    """Patient with diabetes_diagnosis=1 in an observation visit is excluded."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", diabetes_diagnosis=1, is_future=False),
            _visit("2020-04-01", diabetes_diagnosis=1, is_future=False),
            _visit("2020-07-01", diabetes_diagnosis=1, is_future=False),
            _visit("2021-01-01", is_future=True),
        ],
        "SYN_00002": [
            _visit("2020-01-01", is_future=False),
            _visit("2020-04-01", is_future=False),
            _visit("2020-07-01", is_future=False),
            _visit("2021-01-01", is_future=True),
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        max_observation_visits=4,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=True,
    )
    result = CohortBuilder(cfg).build(df)
    assert "SYN_00001" not in result.patient_targets.index, (
        "Patient with prior diabetes should be excluded"
    )
    assert "SYN_00002" in result.patient_targets.index
    assert result.audit.excluded_prior_diabetes == 1


def test_prior_diabetes_not_excluded_when_config_false() -> None:
    """Prior diabetes exclusion can be disabled via config."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", diabetes_diagnosis=1, is_future=False),
            _visit("2020-04-01", diabetes_diagnosis=1, is_future=False),
            _visit("2020-07-01", diabetes_diagnosis=1, is_future=False),
            _visit("2021-01-01", hba1c=7.0, diabetes_diagnosis=1, is_future=True),
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        max_observation_visits=4,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    assert "SYN_00001" in result.patient_targets.index


# ---------------------------------------------------------------------------
# 2. Progression label uses future records
# ---------------------------------------------------------------------------

def test_positive_target_from_future_hba1c() -> None:
    """Target = 1 when a future visit has HbA1c >= 6.5."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", hba1c=5.5, is_future=False),
            _visit("2020-04-01", hba1c=5.7, is_future=False),
            _visit("2020-07-01", hba1c=5.8, is_future=False),
            _visit("2021-01-01", hba1c=7.0, is_future=True),  # future: positive
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        max_observation_visits=4,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    assert result.patient_targets["SYN_00001"] == 1


def test_negative_target_when_future_hba1c_normal() -> None:
    """Target = 0 when all future HbA1c values are below threshold."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", hba1c=5.5, is_future=False),
            _visit("2020-04-01", hba1c=5.6, is_future=False),
            _visit("2020-07-01", hba1c=5.7, is_future=False),
            _visit("2021-01-01", hba1c=5.9, is_future=True),  # below 6.5
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        max_observation_visits=4,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    assert result.patient_targets["SYN_00001"] == 0


def test_positive_target_from_future_diabetes_diagnosis() -> None:
    """Target = 1 when a future visit has diabetes_diagnosis=1."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", is_future=False),
            _visit("2020-04-01", is_future=False),
            _visit("2020-07-01", is_future=False),
            _visit("2021-01-01", diabetes_diagnosis=1, is_future=True),
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    assert result.patient_targets["SYN_00001"] == 1


# ---------------------------------------------------------------------------
# 3. Insufficient follow-up
# ---------------------------------------------------------------------------

def test_no_future_visits_patient_excluded() -> None:
    """Patient with no future visits is excluded when require_followup=True."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", is_future=False),
            _visit("2020-04-01", is_future=False),
            _visit("2020-07-01", is_future=False),
            # No future visits
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        require_followup_after_cutoff=True,
    )
    result = CohortBuilder(cfg).build(df)
    assert "SYN_00001" not in result.patient_targets.index
    assert result.audit.excluded_insufficient_followup == 1


def test_future_visit_outside_horizon_excluded() -> None:
    """Patient whose only future visit is outside the prediction horizon is excluded."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", is_future=False),
            _visit("2020-04-01", is_future=False),
            _visit("2020-07-01", is_future=False),  # cutoff ≈ 2020-07-01
            _visit("2022-01-01", is_future=True),   # > 365 days after cutoff
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        require_followup_after_cutoff=True,
    )
    result = CohortBuilder(cfg).build(df)
    # Future visit is at 2022-01-01; cutoff is 2020-07-01; horizon end = 2021-07-01
    # So future visit is outside horizon → excluded
    assert "SYN_00001" not in result.patient_targets.index


# ---------------------------------------------------------------------------
# 4. Insufficient observation visits
# ---------------------------------------------------------------------------

def test_too_few_observation_visits_excluded() -> None:
    """Patient with fewer than min_observation_visits is excluded."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", is_future=False),
            _visit("2020-04-01", is_future=False),
            # Only 2 obs visits; min=3
            _visit("2021-01-01", is_future=True),
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
    )
    result = CohortBuilder(cfg).build(df)
    assert "SYN_00001" not in result.patient_targets.index
    assert result.audit.excluded_insufficient_visits >= 1


# ---------------------------------------------------------------------------
# 5. Observation rows do not contain future records
# ---------------------------------------------------------------------------

def test_observation_data_excludes_future_rows() -> None:
    """observation_data only contains rows where is_future_visit=False."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", hba1c=5.5, is_future=False),
            _visit("2020-04-01", hba1c=5.6, is_future=False),
            _visit("2020-07-01", hba1c=5.7, is_future=False),
            _visit("2021-01-01", hba1c=7.0, is_future=True),
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    # is_future_visit column must not be present (stripped)
    assert "is_future_visit" not in result.observation_data.columns
    # All observation rows must be from before the cutoff
    obs = result.observation_data[result.observation_data["patient_id"] == "SYN_00001"]
    assert len(obs) == 3
    # HbA1c 7.0 (the future value) must not appear in observation rows
    assert 7.0 not in obs["hba1c"].values


def test_future_leakage_columns_stripped() -> None:
    """target, is_future_visit, etc. are absent from observation_data."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", is_future=False),
            _visit("2020-04-01", is_future=False),
            _visit("2020-07-01", is_future=False),
            _visit("2021-01-01", hba1c=7.0, is_future=True),
        ],
    }
    df = _make_longitudinal_df(patients)
    # Add a target column to the input
    df["target"] = 0
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    for col in _FUTURE_LEAKAGE_COLUMNS:
        assert col not in result.observation_data.columns, (
            f"Leakage column {col!r} found in observation_data"
        )


# ---------------------------------------------------------------------------
# 6. Cutoff date is correct
# ---------------------------------------------------------------------------

def test_cutoff_date_is_last_observation_visit() -> None:
    """Cutoff date stored in audit equals the last observation visit date."""
    obs_dates = ["2020-01-01", "2020-04-01", "2020-07-01"]
    patients = {
        "SYN_00001": [
            _visit(d, is_future=False) for d in obs_dates
        ] + [
            _visit("2021-01-01", hba1c=7.0, is_future=True)
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    expected_cutoff = "2020-07-01"
    assert result.audit.cutoff_dates.get("SYN_00001") == expected_cutoff


# ---------------------------------------------------------------------------
# 7. Prediction horizon is applied correctly
# ---------------------------------------------------------------------------

def test_future_visit_within_horizon_counts() -> None:
    """A future visit at exactly horizon boundary is included."""
    patients = {
        "SYN_00001": [
            _visit("2020-01-01", is_future=False),
            _visit("2020-04-01", is_future=False),
            _visit("2020-07-01", is_future=False),  # cutoff
            # Exactly 365 days after cutoff
            _visit("2021-07-01", hba1c=7.0, is_future=True),
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    # The visit at exactly horizon end should be included → target=1
    assert "SYN_00001" in result.patient_targets.index


# ---------------------------------------------------------------------------
# 8. Pre-labeled mode
# ---------------------------------------------------------------------------

def test_pre_labeled_mode_reads_target_from_column() -> None:
    """Pre-labeled mode reads patient target directly from the input target column."""
    patients = {
        "SYN_00001": [
            {**_visit("2020-01-01"), "target": 1},
            {**_visit("2020-04-01"), "target": 1},
            {**_visit("2020-07-01"), "target": 1},
        ],
        "SYN_00002": [
            {**_visit("2020-01-01"), "target": 0},
            {**_visit("2020-04-01"), "target": 0},
            {**_visit("2020-07-01"), "target": 0},
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.PRE_LABELED,
        min_observation_visits=3,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    assert result.patient_targets["SYN_00001"] == 1
    assert result.patient_targets["SYN_00002"] == 0


def test_pre_labeled_inconsistent_target_raises() -> None:
    """Pre-labeled mode raises ValueError on inconsistent patient target."""
    patients = {
        "SYN_00001": [
            {**_visit("2020-01-01"), "target": 0},
            {**_visit("2020-04-01"), "target": 1},  # inconsistent
            {**_visit("2020-07-01"), "target": 0},
        ],
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.PRE_LABELED,
        min_observation_visits=3,
        target_inconsistency_as_error=True,
        exclude_known_diabetes_at_baseline=False,
    )
    with pytest.raises(ValueError, match="inconsistent target"):
        CohortBuilder(cfg).build(df)


# ---------------------------------------------------------------------------
# 9. Audit counts
# ---------------------------------------------------------------------------

def test_audit_positive_negative_counts() -> None:
    """Audit accurately counts positive and negative patients."""
    patients = {
        f"SYN_{i:05d}": [
            _visit("2020-01-01", is_future=False),
            _visit("2020-04-01", is_future=False),
            _visit("2020-07-01", is_future=False),
            # Even-numbered patients get high HbA1c future visit
            _visit("2021-01-01", hba1c=7.0 if i % 2 == 0 else 5.5, is_future=True),
        ]
        for i in range(1, 7)
    }
    df = _make_longitudinal_df(patients)
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    assert result.audit.positive_patients == 3  # patients 2, 4, 6
    assert result.audit.negative_patients == 3  # patients 1, 3, 5
    assert result.audit.positive_rate == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# 10. _strip_leakage_columns helper
# ---------------------------------------------------------------------------

def test_strip_leakage_columns_removes_all_known() -> None:
    """_strip_leakage_columns removes all columns in _FUTURE_LEAKAGE_COLUMNS."""
    df = pd.DataFrame({
        "patient_id": ["A"],
        "visit_date": [pd.Timestamp("2020-01-01")],
        "hba1c": [5.5],
        "target": [1],
        "is_future_visit": [False],
    })
    cleaned = _strip_leakage_columns(df)
    for col in _FUTURE_LEAKAGE_COLUMNS:
        assert col not in cleaned.columns
    # Safe columns remain
    assert "patient_id" in cleaned.columns
    assert "hba1c" in cleaned.columns
