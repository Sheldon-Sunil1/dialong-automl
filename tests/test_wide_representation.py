"""Tests for the wide longitudinal representation (Phase 4).

Covers:
1.  One row per patient.
2.  Chronological wave ordering (wave 1 = oldest visit).
3.  Expected wave-specific feature names (hba1c_w1 … hba1c_w4).
4.  Expected temporal feature calculations (mean, std, delta, slope, …).
5.  Feature mapping exists for every generated column.
6.  Patients with fewer visits than max_waves get NaN in later waves.
7.  Categorical features use the most-recent observed value.
8.  n_obs_visits column counts correctly.
9.  target never enters X.
10. LeakageError when leakage columns are present in input.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from dialong_automl.features.wide import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeatureInfo,
    LeakageError,
    WideRepresentationBuilder,
    WideRepresentationResult,
    TEMPORAL_SUFFIXES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_df(
    patients: dict[str, list[dict]],
) -> pd.DataFrame:
    """Build a minimal observation DataFrame from a dict of pid -> visits."""
    rows = []
    for pid, visits in patients.items():
        for i, v in enumerate(visits):
            row = {
                "patient_id": pid,
                "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(days=i * 90),
                "visit_number": i + 1,
                "diabetes_diagnosis": 0,
            }
            row.update(v)
            rows.append(row)
    return pd.DataFrame(rows)


def _minimal_obs(n_patients: int = 4, n_visits: int = 4) -> pd.DataFrame:
    """Return a minimal but complete obs DataFrame."""
    rows = []
    for i in range(1, n_patients + 1):
        pid = f"SYN_{i:05d}"
        for v in range(n_visits):
            rows.append({
                "patient_id": pid,
                "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(days=v * 90),
                "visit_number": v + 1,
                "hba1c": 5.0 + i * 0.1 + v * 0.05,
                "fasting_glucose": 90.0 + v,
                "bmi": 25.0,
                "systolic_bp": 120.0,
                "diastolic_bp": 80.0,
                "ldl_cholesterol": 110.0,
                "hdl_cholesterol": 55.0,
                "triglycerides": 130.0,
                "heart_rate": 72.0,
                "age_at_visit": 45.0 + v * 0.25,
                "sex": "M",
                "smoking_status": "never",
                "physical_activity_level": "moderate",
                "family_history_diabetes": "no",
                "metformin_use": "no",
                "hypertension_diagnosis": "no",
                "diabetes_diagnosis": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. One row per patient
# ---------------------------------------------------------------------------

def test_one_row_per_patient() -> None:
    """WideRepresentationBuilder produces exactly one row per patient."""
    df = _minimal_obs(n_patients=5)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    assert len(result.X) == 5
    assert len(result.patient_ids) == 5


def test_patient_ids_are_strings() -> None:
    """patient_ids are string values."""
    df = _minimal_obs(n_patients=3)
    result = WideRepresentationBuilder().build(df)
    assert all(isinstance(p, str) for p in result.patient_ids)


# ---------------------------------------------------------------------------
# 2. Chronological wave ordering
# ---------------------------------------------------------------------------

def test_wave_1_is_oldest_visit() -> None:
    """Wave 1 must be the oldest (first chronologically) visit."""
    # Patient has 3 visits with known hba1c values in date order
    patients = {
        "P01": [
            {"visit_date": "2020-01-01", "hba1c": 5.1},
            {"visit_date": "2020-04-01", "hba1c": 5.5},
            {"visit_date": "2020-07-01", "hba1c": 5.9},
        ]
    }
    df = _obs_df(patients)
    # Override dates to be explicitly ordered
    df = df.copy()
    df["visit_date"] = pd.to_datetime(df["visit_date"])

    result = WideRepresentationBuilder(max_waves=4).build(df)
    row = result.X.iloc[0]
    assert abs(row["hba1c_w1"] - 5.1) < 0.01, f"Expected 5.1 at wave 1, got {row['hba1c_w1']}"
    assert abs(row["hba1c_w2"] - 5.5) < 0.01
    assert abs(row["hba1c_w3"] - 5.9) < 0.01
    assert math.isnan(row["hba1c_w4"])  # only 3 visits, wave 4 = NaN


def test_wave_ordering_reversed_dates_sorted() -> None:
    """If input dates are out of order, waves are still chronological."""
    rows = [
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-07-01"), "hba1c": 6.0, "visit_number": 3, "diabetes_diagnosis": 0},
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-01-01"), "hba1c": 5.0, "visit_number": 1, "diabetes_diagnosis": 0},
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-04-01"), "hba1c": 5.5, "visit_number": 2, "diabetes_diagnosis": 0},
    ]
    df = pd.DataFrame(rows)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    row = result.X.iloc[0]
    assert abs(row["hba1c_w1"] - 5.0) < 0.01  # oldest
    assert abs(row["hba1c_w2"] - 5.5) < 0.01
    assert abs(row["hba1c_w3"] - 6.0) < 0.01  # most recent


# ---------------------------------------------------------------------------
# 3. Expected wave-specific feature names
# ---------------------------------------------------------------------------

def test_wave_column_names_present() -> None:
    """Wave columns hba1c_w1..w4 are present when max_waves=4."""
    df = _minimal_obs()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    for w in range(1, 5):
        assert f"hba1c_w{w}" in result.feature_names


def test_wave_columns_for_all_numeric_features() -> None:
    """Every numeric feature gets wave columns."""
    df = _minimal_obs()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    for var in NUMERIC_FEATURES:
        if var in df.columns:
            for w in range(1, 5):
                assert f"{var}_w{w}" in result.feature_names, (
                    f"Missing wave column: {var}_w{w}"
                )


def test_temporal_suffix_columns_present() -> None:
    """Every numeric feature has all temporal suffix columns."""
    df = _minimal_obs()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    for var in NUMERIC_FEATURES:
        if var in df.columns:
            for suf in TEMPORAL_SUFFIXES:
                col = f"{var}{suf}"
                assert col in result.feature_names, f"Missing: {col}"


# ---------------------------------------------------------------------------
# 4. Expected temporal feature calculations
# ---------------------------------------------------------------------------

def test_hba1c_mean_correct() -> None:
    """hba1c_mean equals the arithmetic mean of the patient's hba1c values."""
    patients = {"P1": [{"hba1c": 5.0}, {"hba1c": 6.0}, {"hba1c": 7.0}]}
    df = _obs_df(patients)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    assert abs(result.X.iloc[0]["hba1c_mean"] - 6.0) < 1e-6


def test_hba1c_min_max_correct() -> None:
    """hba1c_min and hba1c_max match numpy min/max."""
    patients = {"P1": [{"hba1c": 5.1}, {"hba1c": 4.8}, {"hba1c": 5.9}]}
    df = _obs_df(patients)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    assert abs(result.X.iloc[0]["hba1c_min"] - 4.8) < 1e-6
    assert abs(result.X.iloc[0]["hba1c_max"] - 5.9) < 1e-6


def test_hba1c_latest_is_most_recent() -> None:
    """hba1c_latest equals the value from the most recent visit."""
    patients = {"P1": [{"hba1c": 5.0}, {"hba1c": 5.5}, {"hba1c": 6.1}]}
    df = _obs_df(patients)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    assert abs(result.X.iloc[0]["hba1c_latest"] - 6.1) < 1e-6


def test_hba1c_delta_last_minus_first() -> None:
    """hba1c_delta = last - first."""
    patients = {"P1": [{"hba1c": 5.2}, {"hba1c": 5.8}, {"hba1c": 6.3}]}
    df = _obs_df(patients)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    expected = 6.3 - 5.2
    assert abs(result.X.iloc[0]["hba1c_delta"] - expected) < 1e-5


def test_hba1c_std_two_values() -> None:
    """hba1c_std is the sample std (ddof=1) of observations."""
    patients = {"P1": [{"hba1c": 4.0}, {"hba1c": 6.0}]}
    df = _obs_df(patients)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    expected = np.std([4.0, 6.0], ddof=1)
    assert abs(result.X.iloc[0]["hba1c_std"] - expected) < 1e-5


def test_temporal_nan_for_single_visit_std_delta_slope() -> None:
    """std, delta, slope are NaN when patient has only 1 observation visit."""
    patients = {"P1": [{"hba1c": 5.5}]}
    df = _obs_df(patients)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    row = result.X.iloc[0]
    assert math.isnan(row["hba1c_std"])
    assert math.isnan(row["hba1c_delta"])
    assert math.isnan(row["hba1c_slope"])


def test_hba1c_slope_linear_trajectory() -> None:
    """hba1c_slope matches the known OLS slope for a linear trajectory."""
    # Values: 5.0, 5.5, 6.0 at visit indices 0, 1, 2
    # OLS slope = (3*sum(i*y) - sum(i)*sum(y)) / (3*sum(i^2) - sum(i)^2)
    # = (3*(0+5.5+12) - 3*16.5) / (3*5 - 9) = (52.5-49.5)/6 = 0.5
    patients = {"P1": [{"hba1c": 5.0}, {"hba1c": 5.5}, {"hba1c": 6.0}]}
    df = _obs_df(patients)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    assert abs(result.X.iloc[0]["hba1c_slope"] - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# 5. Feature mapping
# ---------------------------------------------------------------------------

def test_feature_mapping_covers_all_columns() -> None:
    """Every column in feature_names has a corresponding FeatureInfo entry."""
    df = _minimal_obs()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    for col in result.feature_names:
        assert col in result.feature_mapping, f"No mapping for column: {col}"


def test_feature_mapping_wave_info() -> None:
    """Wave columns have visit_wave set and feature_type == 'numeric_wave'."""
    df = _minimal_obs()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    info = result.feature_mapping["hba1c_w2"]
    assert info.visit_wave == 2
    assert info.source_variable == "hba1c"
    assert info.feature_type == "numeric_wave"
    assert info.transformation == "raw_observation"


def test_feature_mapping_slope_info() -> None:
    """Slope columns have transformation == 'linear_slope'."""
    df = _minimal_obs()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    info = result.feature_mapping["hba1c_slope"]
    assert info.transformation == "linear_slope"
    assert info.feature_type == "numeric_temporal"
    assert info.visit_wave is None


def test_feature_mapping_categorical_info() -> None:
    """Categorical columns have feature_type == 'categorical'."""
    df = _minimal_obs()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    info = result.feature_mapping["sex_latest"]
    assert info.feature_type == "categorical"
    assert info.transformation == "categorical_latest"


def test_feature_mapping_to_dict() -> None:
    """FeatureInfo.to_dict() returns expected keys."""
    df = _minimal_obs()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    d = result.feature_mapping["hba1c_w1"].to_dict()
    assert "column_name" in d
    assert "source_variable" in d
    assert "visit_wave" in d
    assert "transformation" in d
    assert "description" in d
    assert "feature_type" in d


# ---------------------------------------------------------------------------
# 6. Patients with fewer visits get NaN in later waves
# ---------------------------------------------------------------------------

def test_short_patient_gets_nan_in_later_waves() -> None:
    """Patient with 2 visits gets NaN for waves 3 and 4 (max_waves=4)."""
    patients = {
        "LONG": [{"hba1c": 5.0}, {"hba1c": 5.1}, {"hba1c": 5.2}, {"hba1c": 5.3}],
        "SHORT": [{"hba1c": 6.0}, {"hba1c": 6.1}],
    }
    df = _obs_df(patients)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    short_row = result.X.loc[result.X.index[result.patient_ids.index("SHORT")]]
    assert not math.isnan(short_row["hba1c_w1"])
    assert not math.isnan(short_row["hba1c_w2"])
    assert math.isnan(short_row["hba1c_w3"])
    assert math.isnan(short_row["hba1c_w4"])


# ---------------------------------------------------------------------------
# 7. Categorical most-recent value
# ---------------------------------------------------------------------------

def test_categorical_latest_uses_most_recent() -> None:
    """sex_latest reflects the value from the most recent visit."""
    rows = [
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-01-01"),
         "sex": "M", "visit_number": 1, "diabetes_diagnosis": 0},
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-04-01"),
         "sex": "M", "visit_number": 2, "diabetes_diagnosis": 0},
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-07-01"),
         "sex": "F", "visit_number": 3, "diabetes_diagnosis": 0},  # changed
    ]
    df = pd.DataFrame(rows)
    result = WideRepresentationBuilder(max_waves=4, categorical_features=["sex"]).build(df)
    assert result.X.iloc[0]["sex_latest"] == "F"


# ---------------------------------------------------------------------------
# 8. n_obs_visits
# ---------------------------------------------------------------------------

def test_n_obs_visits_counts_correctly() -> None:
    """n_obs_visits column equals the actual number of visits per patient."""
    patients = {
        "P1": [{"hba1c": 5.0}] * 4,
        "P2": [{"hba1c": 5.0}] * 2,
    }
    df = _obs_df(patients)
    result = WideRepresentationBuilder(max_waves=4, include_n_visits=True).build(df)
    n_col = result.X["n_obs_visits"].tolist()
    p1_idx = result.patient_ids.index("P1")
    p2_idx = result.patient_ids.index("P2")
    assert n_col[p1_idx] == 4
    assert n_col[p2_idx] == 2


# ---------------------------------------------------------------------------
# 9. Target never in X
# ---------------------------------------------------------------------------

def test_target_not_in_X_columns() -> None:
    """X must not contain a 'target' column even if targets are passed."""
    df = _minimal_obs()
    targets = pd.Series(
        {f"SYN_{i:05d}": i % 2 for i in range(1, 5)},
        name="target",
    )
    targets.index.name = "patient_id"
    result = WideRepresentationBuilder().build(df, patient_targets=targets)
    assert "target" not in result.X.columns
    assert result.y is not None  # stored separately


def test_no_leakage_column_names_in_X() -> None:
    """X must not contain any known leakage column names."""
    leakage_names = {
        "target", "is_future_visit", "future_hba1c", "future_glucose",
        "diabetes_progression", "outcome", "diagnosis_after", "post_outcome",
    }
    df = _minimal_obs()
    result = WideRepresentationBuilder().build(df)
    for col in result.feature_names:
        assert col not in leakage_names, f"Leakage column found in X: {col}"


# ---------------------------------------------------------------------------
# 10. LeakageError
# ---------------------------------------------------------------------------

def test_leakage_error_on_target_column() -> None:
    """LeakageError when 'target' column is present in the input."""
    df = _minimal_obs()
    df["target"] = 0
    with pytest.raises(LeakageError):
        WideRepresentationBuilder().build(df)


def test_leakage_error_on_is_future_visit() -> None:
    """LeakageError when 'is_future_visit' column is present."""
    df = _minimal_obs()
    df["is_future_visit"] = False
    with pytest.raises(LeakageError):
        WideRepresentationBuilder().build(df)


def test_leakage_error_on_outcome_column() -> None:
    """LeakageError for any _FUTURE_LEAKAGE_COLUMNS member."""
    df = _minimal_obs()
    df["outcome"] = 0
    with pytest.raises(LeakageError):
        WideRepresentationBuilder().build(df)


# ---------------------------------------------------------------------------
# 11. Metadata
# ---------------------------------------------------------------------------

def test_metadata_n_features_matches() -> None:
    """metadata['n_features'] matches the actual number of feature columns."""
    df = _minimal_obs()
    result = WideRepresentationBuilder(max_waves=4).build(df)
    assert result.metadata["n_features"] == len(result.feature_names)


def test_result_x_shape() -> None:
    """X has exactly (n_patients, n_features) shape."""
    df = _minimal_obs(n_patients=6)
    result = WideRepresentationBuilder(max_waves=4).build(df)
    assert result.X.shape[0] == 6
    assert result.X.shape[1] == len(result.feature_names)
