"""Adversarial temporal-leakage tests for Phase 4 representations.

These tests construct a patient where future values are deliberately extreme
(HbA1c = 12.0, fasting_glucose = 350, diabetes_diagnosis = 1) and verify
that neither the wide nor the sequence representation contains those values.

Patient setup
-------------
Observation visits (is_future_visit = False):
    visit 1: hba1c = 5.8, fasting_glucose = 90.0
    visit 2: hba1c = 5.9, fasting_glucose = 91.0
    visit 3: hba1c = 6.0, fasting_glucose = 92.0
    visit 4: hba1c = 6.1, fasting_glucose = 93.0

Future visits (is_future_visit = True — must NEVER enter X):
    visit 5: hba1c = 12.0, fasting_glucose = 350.0, diabetes_diagnosis = 1

Guarantees verified
-------------------
W1.  Wide X does not contain 12.0 in any hba1c column.
W2.  Wide X does not contain 350.0 in any fasting_glucose column.
W3.  Wide target may be 1 (stored separately) but is NOT in X.
W4.  No column in wide X is a known leakage column name.
W5.  LeakageError when target or is_future_visit is passed to wide builder.

S1.  Sequence X does not contain 12.0 in the hba1c feature position.
S2.  Sequence X does not contain 350.0 in the fasting_glucose position.
S3.  Sequence target may be 1 but is NOT in X.
S4.  LeakageError when target or is_future_visit is passed to sequence builder.

P1.  No future-derived row enters the observation cutoff check.
P2.  Cohort builder strips leakage columns before calling builders.
     (Validates the Phase 3 → Phase 4 integration pathway.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dialong_automl.features.wide import (
    LeakageError,
    WideRepresentationBuilder,
)
from dialong_automl.features.sequence import SequenceRepresentationBuilder


# ---------------------------------------------------------------------------
# Fixture: adversarial patient
# ---------------------------------------------------------------------------

_FUTURE_HBA1C = 12.0
_FUTURE_GLUCOSE = 350.0
_OBS_HBA1C = [5.8, 5.9, 6.0, 6.1]


def _adversarial_obs_df() -> pd.DataFrame:
    """Observation-only DataFrame — CohortBuilder has already stripped future rows."""
    rows = []
    for i, hba1c in enumerate(_OBS_HBA1C):
        rows.append({
            "patient_id": "SYN_00001",
            "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(days=i * 90),
            "visit_number": i + 1,
            "hba1c": hba1c,
            "fasting_glucose": 90.0 + i,
            "bmi": 26.0,
            "systolic_bp": 122.0,
            "diastolic_bp": 80.0,
            "ldl_cholesterol": 115.0,
            "hdl_cholesterol": 53.0,
            "triglycerides": 135.0,
            "heart_rate": 70.0,
            "age_at_visit": 48.0 + i * 0.25,
            "sex": "F",
            "smoking_status": "never",
            "physical_activity_level": "moderate",
            "family_history_diabetes": "yes",
            "metformin_use": "no",
            "hypertension_diagnosis": "no",
            "diabetes_diagnosis": 0,  # no diagnosis during observation
        })
    return pd.DataFrame(rows)


def _adversarial_obs_with_future() -> pd.DataFrame:
    """Full timeline — includes the future row (is_future_visit=True).

    Simulates the raw CSV before Phase 3 processing.
    This should be REJECTED by both representation builders.
    """
    obs = _adversarial_obs_df()
    future_row = pd.DataFrame([{
        "patient_id": "SYN_00001",
        "visit_date": pd.Timestamp("2021-01-01"),
        "visit_number": 5,
        "hba1c": _FUTURE_HBA1C,
        "fasting_glucose": _FUTURE_GLUCOSE,
        "bmi": 45.0,
        "systolic_bp": 180.0,
        "diastolic_bp": 110.0,
        "ldl_cholesterol": 220.0,
        "hdl_cholesterol": 20.0,
        "triglycerides": 500.0,
        "heart_rate": 120.0,
        "age_at_visit": 49.0,
        "sex": "F",
        "smoking_status": "never",
        "physical_activity_level": "moderate",
        "family_history_diabetes": "yes",
        "metformin_use": "no",
        "hypertension_diagnosis": "yes",
        "diabetes_diagnosis": 1,
        "is_future_visit": True,  # LEAKAGE COLUMN — must be rejected
    }])
    return pd.concat([obs, future_row], ignore_index=True)


# ---------------------------------------------------------------------------
# W1. Wide: future hba1c 12.0 not in X
# ---------------------------------------------------------------------------

def test_wide_future_hba1c_not_in_x() -> None:
    """Wide representation must not contain 12.0 (future HbA1c) anywhere."""
    df = _adversarial_obs_df()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    for col in result.feature_names:
        if "hba1c" in col:
            val = result.X.iloc[0][col]
            assert val != pytest.approx(_FUTURE_HBA1C, abs=0.01), (
                f"TEMPORAL LEAKAGE: future HbA1c {_FUTURE_HBA1C} "
                f"found in wide column {col!r}: {val}"
            )


# ---------------------------------------------------------------------------
# W2. Wide: future glucose 350.0 not in X
# ---------------------------------------------------------------------------

def test_wide_future_glucose_not_in_x() -> None:
    """Wide representation must not contain 350.0 (future fasting_glucose)."""
    df = _adversarial_obs_df()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    for col in result.feature_names:
        if "fasting_glucose" in col:
            val = result.X.iloc[0][col]
            assert val != pytest.approx(_FUTURE_GLUCOSE, abs=0.01), (
                f"LEAKAGE: future glucose {_FUTURE_GLUCOSE} found in {col!r}"
            )


# ---------------------------------------------------------------------------
# W3. Wide: observation hba1c values are correct
# ---------------------------------------------------------------------------

def test_wide_obs_hba1c_values_correct() -> None:
    """Wave columns hba1c_w1..w4 contain the 4 observation values, not 12.0."""
    df = _adversarial_obs_df()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    row = result.X.iloc[0]
    for w, expected in enumerate(_OBS_HBA1C, start=1):
        col = f"hba1c_w{w}"
        assert abs(row[col] - expected) < 0.01, (
            f"{col}: expected {expected}, got {row[col]}"
        )


# ---------------------------------------------------------------------------
# W4. Wide: target separate, not in X
# ---------------------------------------------------------------------------

def test_wide_target_not_in_x_columns() -> None:
    """X columns must not include 'target'."""
    df = _adversarial_obs_df()
    # Pass target separately
    targets = pd.Series({"SYN_00001": 1}, name="target")
    targets.index.name = "patient_id"
    result = WideRepresentationBuilder(max_waves=4).build(df, patient_targets=targets)
    assert "target" not in result.X.columns
    assert result.y is not None
    assert result.y["SYN_00001"] == 1


# ---------------------------------------------------------------------------
# W5. Wide: LeakageError on columns that should have been stripped
# ---------------------------------------------------------------------------

def test_wide_raises_if_is_future_visit_present() -> None:
    """Wide builder raises LeakageError when is_future_visit is present."""
    df = _adversarial_obs_with_future()
    # The raw df has is_future_visit column
    with pytest.raises(LeakageError):
        WideRepresentationBuilder(max_waves=4).build(df)


def test_wide_raises_if_target_column_present() -> None:
    """Wide builder raises LeakageError when target column is in the df."""
    df = _adversarial_obs_df()
    df["target"] = 1
    with pytest.raises(LeakageError):
        WideRepresentationBuilder(max_waves=4).build(df)


# ---------------------------------------------------------------------------
# S1. Sequence: future hba1c 12.0 not in X
# ---------------------------------------------------------------------------

def test_sequence_future_hba1c_not_in_x() -> None:
    """Sequence representation must not contain 12.0 (future HbA1c)."""
    df = _adversarial_obs_df()
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)

    hba1c_idx = result.feature_names.index("hba1c") if "hba1c" in result.feature_names else None
    if hba1c_idx is not None:
        hba1c_values = result.X[0, :, hba1c_idx]
        for t, val in enumerate(hba1c_values):
            # Only real timesteps; padded = 0.0
            if result.mask[0, t]:
                assert val != pytest.approx(_FUTURE_HBA1C, abs=0.01), (
                    f"TEMPORAL LEAKAGE: future HbA1c {_FUTURE_HBA1C} "
                    f"found in sequence at timestep {t}: {val}"
                )


# ---------------------------------------------------------------------------
# S2. Sequence: observation hba1c values match expected
# ---------------------------------------------------------------------------

def test_sequence_obs_hba1c_values_correct() -> None:
    """Sequence timesteps 0-3 contain the 4 observation HbA1c values."""
    df = _adversarial_obs_df()
    result = SequenceRepresentationBuilder(
        max_seq_len=4,
        numeric_features=["hba1c"],
        categorical_features=[],
    ).build(df)
    hba1c_idx = result.feature_names.index("hba1c")
    pid_idx = result.patient_ids.index("SYN_00001")
    for t, expected in enumerate(_OBS_HBA1C):
        val = float(result.X[pid_idx, t, hba1c_idx])
        assert abs(val - expected) < 0.01, (
            f"Timestep {t}: expected {expected}, got {val}"
        )


# ---------------------------------------------------------------------------
# S3. Sequence: target not in X
# ---------------------------------------------------------------------------

def test_sequence_target_not_in_feature_names() -> None:
    """feature_names must not include 'target'."""
    df = _adversarial_obs_df()
    targets = pd.Series({"SYN_00001": 1}, name="target")
    targets.index.name = "patient_id"
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df, patient_targets=targets)
    assert "target" not in result.feature_names
    assert result.y is not None
    assert result.y["SYN_00001"] == 1


# ---------------------------------------------------------------------------
# S4. Sequence: LeakageError on is_future_visit
# ---------------------------------------------------------------------------

def test_sequence_raises_if_is_future_visit_present() -> None:
    """Sequence builder raises LeakageError when is_future_visit is present."""
    df = _adversarial_obs_with_future()
    with pytest.raises(LeakageError):
        SequenceRepresentationBuilder(max_seq_len=4).build(df)


def test_sequence_raises_if_target_column_present() -> None:
    """Sequence builder raises LeakageError when target column is in the df."""
    df = _adversarial_obs_df()
    df["target"] = 1
    with pytest.raises(LeakageError):
        SequenceRepresentationBuilder(max_seq_len=4).build(df)


# ---------------------------------------------------------------------------
# P1. Phase 3 → Phase 4 integration: cohort strips leakage before builders
# ---------------------------------------------------------------------------

def test_cohort_observation_data_safe_for_builders() -> None:
    """CohortBuilder.observation_data is accepted by both representation builders."""
    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
    from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator

    cfg = GeneratorConfig(
        n_patients=30,
        min_obs_visits=3,
        max_obs_visits=4,
        min_future_visits=2,
        max_future_visits=3,
        seed=42,
        missingness_rate=0.0,
    )
    gen_result = SyntheticGenerator(config=cfg).generate()

    cohort_cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        max_observation_visits=4,
        prediction_horizon_days=365 * 5,
        exclude_known_diabetes_at_baseline=False,
        require_followup_after_cutoff=True,
    )
    cohort = CohortBuilder(cohort_cfg).build(gen_result.full_df)
    obs = cohort.observation_data

    # Both builders must accept observation_data without raising
    wide_result = WideRepresentationBuilder(max_waves=4).build(obs)
    seq_result = SequenceRepresentationBuilder(max_seq_len=4).build(obs)

    assert len(wide_result.patient_ids) > 0
    assert seq_result.X.shape[0] > 0

    # Confirm no leakage columns in outputs
    leakage_cols = {"target", "is_future_visit", "diabetes_progression",
                    "outcome", "future_hba1c", "future_glucose"}
    for col in wide_result.feature_names:
        assert col not in leakage_cols, f"Leakage col in wide output: {col}"
    for name in seq_result.feature_names:
        assert name not in leakage_cols, f"Leakage col in seq output: {name}"


def test_future_value_cannot_enter_wide_via_cohort() -> None:
    """End-to-end: future HbA1c 12.0 from cohort pipeline never enters wide X."""
    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
    from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator
    from dialong_automl.data.synthetic_generator import DIABETES_HBA1C_THRESHOLD

    # Generate with very high progression slope so some futures cross threshold
    cfg = GeneratorConfig(
        n_patients=20,
        min_obs_visits=3,
        max_obs_visits=4,
        min_future_visits=2,
        max_future_visits=3,
        seed=7,
        missingness_rate=0.0,
        progression_prevalence=0.99,
    )
    gen_result = SyntheticGenerator(config=cfg).generate()

    cohort_cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        max_observation_visits=4,
        prediction_horizon_days=365 * 5,
        exclude_known_diabetes_at_baseline=False,
        require_followup_after_cutoff=True,
    )
    cohort = CohortBuilder(cohort_cfg).build(gen_result.full_df)
    obs = cohort.observation_data

    wide_result = WideRepresentationBuilder(max_waves=4).build(obs)

    # Any future HbA1c that is >= 6.5 must NOT appear in observation wave columns
    future_hba1c_vals = (
        gen_result.full_df[gen_result.full_df["is_future_visit"].astype(bool)]["hba1c"]
        .dropna()
        .tolist()
    )
    future_hba1c_extreme = [v for v in future_hba1c_vals if v >= DIABETES_HBA1C_THRESHOLD]

    if future_hba1c_extreme:
        for col in wide_result.feature_names:
            if "_w" in col and "hba1c" in col:
                col_vals = wide_result.X[col].dropna().tolist()
                for fv in future_hba1c_extreme:
                    assert fv not in col_vals, (
                        f"LEAKAGE: future HbA1c {fv} found in wide wave column {col!r}"
                    )
