"""Temporal leakage tests for DiaLong-AutoML (Phase 3).

These tests use deliberately extreme synthetic cases where future values
are set to extreme values (HbA1c = 12.0, fasting_glucose = 350, etc.),
making any accidental leakage immediately detectable.

Guarantees verified
-------------------
Guarantee 1 — Future observations are never returned as model features.
    test_future_hba1c_does_not_enter_observation_features
    test_observation_hba1c_remains_at_58
    test_extreme_future_values_not_in_numeric_obs_columns
    test_two_patients_leakage_check

Guarantee 2 — Target is derived only from post-cutoff observations
               within prediction_horizon_days.
    test_target_becomes_1_from_future_hba1c
    test_future_diabetes_diagnosis_not_in_observation_features
    test_future_outcome_audit_df_has_future_details

Guarantee 3 — target, is_future_visit, and all _FUTURE_LEAKAGE_COLUMNS
               are stripped from observation_data before it is returned.
    test_target_column_absent_from_observation_data
    test_is_future_visit_absent_from_observation_data
    test_all_future_leakage_columns_absent

Guarantee 4 — get_safe_observation_columns never includes leakage columns.
    test_get_safe_observation_columns_excludes_leakage

Guarantee 5 — _strip_leakage_columns is idempotent.
    test_strip_leakage_columns_is_deterministic

Design note: the prediction_horizon_days in these tests is set to 365 days.
The FUTURE_OUTCOME mode applies this window relative to each patient's
individual observation cutoff date (the last observation-window visit).
"""

from __future__ import annotations

import pandas as pd
import pytest

from dialong_automl.data.cohort import (
    CohortBuilder,
    CohortConfig,
    CohortMode,
    _FUTURE_LEAKAGE_COLUMNS,
    _strip_leakage_columns,
    get_safe_observation_columns,
)


# ---------------------------------------------------------------------------
# Fixture: extreme leakage scenario
# ---------------------------------------------------------------------------

def _leakage_scenario_df() -> pd.DataFrame:
    """
    Single patient with:
      Observation visits: HbA1c = 5.8 (all below diabetes threshold)
      Future visits:      HbA1c = 12.0, diabetes_diagnosis = 1

    If the cohort builder leaks future data, the observation rows would
    contain HbA1c = 12.0 or diabetes_diagnosis = 1 — both detectable.
    """
    rows = [
        # --- Observation visits (is_future_visit=False) ---
        {
            "patient_id": "SYN_00001",
            "visit_date": "2020-01-01",
            "visit_number": 1,
            "is_future_visit": False,
            "hba1c": 5.8,
            "fasting_glucose": 95.0,
            "bmi": 26.0,
            "systolic_bp": 122.0,
            "diastolic_bp": 80.0,
            "ldl_cholesterol": 115.0,
            "hdl_cholesterol": 53.0,
            "triglycerides": 135.0,
            "heart_rate": 70.0,
            "age_at_visit": 48.0,
            "sex": "F",
            "smoking_status": "never",
            "physical_activity_level": "moderate",
            "family_history_diabetes": "yes",
            "metformin_use": "no",
            "hypertension_diagnosis": "no",
            "diabetes_diagnosis": 0,
            "target": None,
        },
        {
            "patient_id": "SYN_00001",
            "visit_date": "2020-04-01",
            "visit_number": 2,
            "is_future_visit": False,
            "hba1c": 5.8,
            "fasting_glucose": 96.0,
            "bmi": 26.1,
            "systolic_bp": 123.0,
            "diastolic_bp": 80.0,
            "ldl_cholesterol": 116.0,
            "hdl_cholesterol": 52.0,
            "triglycerides": 136.0,
            "heart_rate": 71.0,
            "age_at_visit": 48.2,
            "sex": "F",
            "smoking_status": "never",
            "physical_activity_level": "moderate",
            "family_history_diabetes": "yes",
            "metformin_use": "no",
            "hypertension_diagnosis": "no",
            "diabetes_diagnosis": 0,
            "target": None,
        },
        {
            "patient_id": "SYN_00001",
            "visit_date": "2020-07-01",
            "visit_number": 3,
            "is_future_visit": False,
            "hba1c": 5.8,
            "fasting_glucose": 97.0,
            "bmi": 26.2,
            "systolic_bp": 124.0,
            "diastolic_bp": 81.0,
            "ldl_cholesterol": 117.0,
            "hdl_cholesterol": 51.0,
            "triglycerides": 137.0,
            "heart_rate": 72.0,
            "age_at_visit": 48.5,
            "sex": "F",
            "smoking_status": "never",
            "physical_activity_level": "moderate",
            "family_history_diabetes": "yes",
            "metformin_use": "no",
            "hypertension_diagnosis": "no",
            "diabetes_diagnosis": 0,
            "target": None,
        },
        # --- Future visits (is_future_visit=True) — DELIBERATELY EXTREME ---
        {
            "patient_id": "SYN_00001",
            "visit_date": "2021-01-01",
            "visit_number": 4,
            "is_future_visit": True,    # FUTURE — must NOT enter observation
            "hba1c": 12.0,              # EXTREME — leakage would be obvious
            "fasting_glucose": 350.0,   # EXTREME
            "bmi": 45.0,                # EXTREME
            "systolic_bp": 180.0,       # EXTREME
            "diastolic_bp": 110.0,      # EXTREME
            "ldl_cholesterol": 220.0,   # EXTREME
            "hdl_cholesterol": 20.0,    # EXTREME
            "triglycerides": 500.0,     # EXTREME
            "heart_rate": 120.0,        # EXTREME
            "age_at_visit": 49.0,
            "sex": "F",
            "smoking_status": "never",
            "physical_activity_level": "moderate",
            "family_history_diabetes": "yes",
            "metformin_use": "no",
            "hypertension_diagnosis": "yes",
            "diabetes_diagnosis": 1,    # FUTURE DIAGNOSIS — must not appear in obs
            "target": None,
        },
    ]
    df = pd.DataFrame(rows)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_future_hba1c_does_not_enter_observation_features() -> None:
    """Observation HbA1c must remain 5.8; the future 12.0 must NOT appear."""
    df = _leakage_scenario_df()
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)

    obs = result.observation_data[result.observation_data["patient_id"] == "SYN_00001"]
    assert len(obs) == 3, f"Expected 3 observation rows, got {len(obs)}"

    hba1c_values = obs["hba1c"].tolist()
    assert 12.0 not in hba1c_values, (
        f"TEMPORAL LEAKAGE: future HbA1c 12.0 found in observation features. "
        f"Observed HbA1c values: {hba1c_values}"
    )
    for val in hba1c_values:
        assert val == pytest.approx(5.8, abs=0.01), (
            f"Expected all observation HbA1c to be 5.8 but got {val}"
        )


def test_observation_hba1c_remains_at_58() -> None:
    """Observation feature HbA1c is exactly 5.8, regardless of future value 12.0."""
    df = _leakage_scenario_df()
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    obs = result.observation_data[result.observation_data["patient_id"] == "SYN_00001"]
    assert all(v == pytest.approx(5.8, abs=0.01) for v in obs["hba1c"]), (
        f"Observation HbA1c must be 5.8, got: {obs['hba1c'].tolist()}"
    )


def test_target_becomes_1_from_future_hba1c() -> None:
    """Target CAN become 1 because the future visit has HbA1c = 12.0."""
    df = _leakage_scenario_df()
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    assert result.patient_targets["SYN_00001"] == 1, (
        "Target should be 1 because future HbA1c = 12.0 >= 6.5 threshold"
    )


def test_future_diabetes_diagnosis_not_in_observation_features() -> None:
    """diabetes_diagnosis=1 from a future visit must NOT appear in observation rows."""
    df = _leakage_scenario_df()
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    obs = result.observation_data[result.observation_data["patient_id"] == "SYN_00001"]

    if "diabetes_diagnosis" in obs.columns:
        diag_values = obs["diabetes_diagnosis"].tolist()
        assert 1 not in diag_values, (
            f"TEMPORAL LEAKAGE: future diabetes_diagnosis=1 found in observation "
            f"features: {diag_values}"
        )


def test_target_column_absent_from_observation_data() -> None:
    """'target' column is stripped from observation_data — it is a leakage candidate."""
    df = _leakage_scenario_df()
    df["target"] = 0  # stamp a target column
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    assert "target" not in result.observation_data.columns, (
        "LEAKAGE: 'target' column found in observation_data"
    )


def test_is_future_visit_absent_from_observation_data() -> None:
    """'is_future_visit' flag must be stripped from observation_data."""
    df = _leakage_scenario_df()
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    assert "is_future_visit" not in result.observation_data.columns, (
        "LEAKAGE: 'is_future_visit' flag found in observation_data"
    )


def test_all_future_leakage_columns_absent() -> None:
    """All columns in _FUTURE_LEAKAGE_COLUMNS are absent from observation_data."""
    df = _leakage_scenario_df()
    # Add all leakage-candidate columns to the raw df
    for col in _FUTURE_LEAKAGE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    for col in _FUTURE_LEAKAGE_COLUMNS:
        assert col not in result.observation_data.columns, (
            f"LEAKAGE: column {col!r} found in observation_data"
        )


def test_extreme_future_values_not_in_numeric_obs_columns() -> None:
    """The extreme future numeric values (350, 45, 180, 500 etc.) are not in obs data."""
    df = _leakage_scenario_df()
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)
    obs = result.observation_data[result.observation_data["patient_id"] == "SYN_00001"]

    # Future extreme values that should never appear in observations
    extreme_values = {
        "hba1c": 12.0,
        "fasting_glucose": 350.0,
        "bmi": 45.0,
        "systolic_bp": 180.0,
        "diastolic_bp": 110.0,
        "ldl_cholesterol": 220.0,
        "hdl_cholesterol": 20.0,
        "triglycerides": 500.0,
        "heart_rate": 120.0,
    }
    for col, extreme_val in extreme_values.items():
        if col in obs.columns:
            col_vals = obs[col].tolist()
            assert extreme_val not in col_vals, (
                f"TEMPORAL LEAKAGE: future extreme value {extreme_val} "
                f"found in observation column {col!r}: {col_vals}"
            )


def test_get_safe_observation_columns_excludes_leakage() -> None:
    """get_safe_observation_columns never returns target or future-labeled cols."""
    df = _leakage_scenario_df()
    # Add a few leakage columns
    df["target"] = 0
    df["future_diabetes"] = 0
    df["diabetes_progression"] = 0
    safe = get_safe_observation_columns(df)
    for col in ["target", "future_diabetes", "diabetes_progression", "is_future_visit"]:
        assert col not in safe, (
            f"Column {col!r} should not be in safe observation columns"
        )
    # Non-leakage columns should be present
    assert "hba1c" in safe
    assert "patient_id" in safe


def test_strip_leakage_columns_is_deterministic() -> None:
    """_strip_leakage_columns is idempotent."""
    df = pd.DataFrame({
        "patient_id": ["A"],
        "hba1c": [5.5],
        "target": [0],
        "is_future_visit": [False],
    })
    once = _strip_leakage_columns(df.copy())
    twice = _strip_leakage_columns(once.copy())
    assert list(once.columns) == list(twice.columns)


def test_two_patients_leakage_check() -> None:
    """With two patients — one positive, one negative — neither leaks future data."""
    rows = []
    for pid, future_hba1c, expected_target in [
        ("SYN_00001", 12.0, 1),  # high future HbA1c → target=1
        ("SYN_00002", 5.5, 0),   # normal future HbA1c → target=0
    ]:
        for i, (d, h, future) in enumerate([
            ("2020-01-01", 5.5, False),
            ("2020-04-01", 5.6, False),
            ("2020-07-01", 5.7, False),
            ("2021-01-01", future_hba1c, True),
        ]):
            rows.append({
                "patient_id": pid,
                "visit_date": d,
                "visit_number": i + 1,
                "is_future_visit": future,
                "hba1c": h,
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
                "diabetes_diagnosis": 0,
            })
    df = pd.DataFrame(rows)
    df["visit_date"] = pd.to_datetime(df["visit_date"])

    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df)

    # Verify targets
    assert result.patient_targets["SYN_00001"] == 1
    assert result.patient_targets["SYN_00002"] == 0

    # Verify no leakage
    for pid in ["SYN_00001", "SYN_00002"]:
        obs = result.observation_data[result.observation_data["patient_id"] == pid]
        assert 12.0 not in obs["hba1c"].values, (
            f"Leakage: future HbA1c 12.0 found in patient {pid} observation data"
        )
        assert len(obs) == 3


def test_future_outcome_audit_df_has_future_details() -> None:
    """future_outcome_audit DataFrame records future hba1c and diagnosis info."""
    df_scenario = _leakage_scenario_df()
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        prediction_horizon_days=365,
        exclude_known_diabetes_at_baseline=False,
    )
    result = CohortBuilder(cfg).build(df_scenario)
    audit_df = result.future_outcome_audit
    # Audit should contain the patient
    assert "SYN_00001" in audit_df["patient_id"].values
    row = audit_df[audit_df["patient_id"] == "SYN_00001"].iloc[0]
    # Audit records the future HbA1c max
    assert row["future_hba1c_max"] == pytest.approx(12.0, abs=0.01)
    # Audit records that future diagnosis was positive
    assert row["future_diabetes_diagnosis"] is True or row["future_diabetes_diagnosis"] == 1
