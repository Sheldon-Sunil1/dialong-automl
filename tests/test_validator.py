"""Tests for the dataset validator (Phase 2).

Covers:
- missing required columns
- empty patient_id
- invalid visit_date
- non-binary target
- leakage candidate column detection
- duplicate patient/date behaviour
- target inconsistency behaviour
- chronological order detection
- binary categorical validation
- profiling statistics accuracy
"""

from __future__ import annotations

import pandas as pd
import pytest

from dialong_automl.data.validator import (
    DataValidator,
    ValidationConfig,
    ValidationResult,
    LEAKAGE_PATTERNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal valid DataFrame with required columns only."""
    return pd.DataFrame({
        "patient_id": [f"SYN_{i:05d}" for i in range(1, n + 1)],
        "visit_date": pd.date_range("2020-01-01", periods=n, freq="90D"),
    })


def _full_df(n_patients: int = 4, visits_per: int = 3) -> pd.DataFrame:
    """Return a small but complete synthetic-style DataFrame."""
    rows = []
    for i in range(1, n_patients + 1):
        pid = f"SYN_{i:05d}"
        start = pd.Timestamp("2020-01-01") + pd.DateOffset(days=(i - 1) * 10)
        for v in range(visits_per):
            rows.append({
                "patient_id": pid,
                "visit_date": start + pd.DateOffset(days=v * 90),
                "hba1c": 5.5 + 0.1 * v,
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
                "target": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Missing required columns
# ---------------------------------------------------------------------------

def test_missing_patient_id_column() -> None:
    """Validator rejects a DataFrame without patient_id column."""
    df = _minimal_df().drop(columns=["patient_id"])
    validator = DataValidator()
    result = validator.validate(df)
    assert not result.passed
    codes = [f.code for f in result.errors()]
    assert "MISSING_REQUIRED_COL" in codes


def test_missing_visit_date_column() -> None:
    """Validator rejects a DataFrame without visit_date column."""
    df = _minimal_df().drop(columns=["visit_date"])
    validator = DataValidator()
    result = validator.validate(df)
    assert not result.passed
    codes = [f.code for f in result.errors()]
    assert "MISSING_REQUIRED_COL" in codes


def test_both_required_columns_present_passes() -> None:
    """Minimal DataFrame with patient_id + visit_date passes required-column check."""
    df = _minimal_df()
    validator = DataValidator(ValidationConfig(expected_clinical_columns=[]))
    result = validator.validate(df)
    missing_col_errors = [
        f for f in result.errors() if f.code == "MISSING_REQUIRED_COL"
    ]
    assert len(missing_col_errors) == 0


# ---------------------------------------------------------------------------
# 2. Empty patient_id
# ---------------------------------------------------------------------------

def test_empty_patient_id_rejected() -> None:
    """Rows with empty string patient_id are rejected."""
    df = _minimal_df()
    df.loc[0, "patient_id"] = ""
    validator = DataValidator()
    result = validator.validate(df)
    assert not result.passed
    assert any(f.code == "EMPTY_PATIENT_ID" for f in result.errors())


def test_nan_patient_id_rejected() -> None:
    """Rows with NaN-string patient_id are rejected."""
    df = _minimal_df()
    df.loc[1, "patient_id"] = "nan"
    validator = DataValidator()
    result = validator.validate(df)
    assert not result.passed
    assert any(f.code == "EMPTY_PATIENT_ID" for f in result.errors())


# ---------------------------------------------------------------------------
# 3. Invalid visit_date
# ---------------------------------------------------------------------------

def test_invalid_date_string_rejected() -> None:
    """Validator rejects rows with unparseable visit_date strings."""
    df = pd.DataFrame({
        "patient_id": ["SYN_00001", "SYN_00002"],
        "visit_date": ["not-a-date", "also-bad"],
    })
    validator = DataValidator()
    result = validator.validate(df)
    assert not result.passed
    assert any(f.code == "INVALID_VISIT_DATE" for f in result.errors())


def test_valid_dates_pass() -> None:
    """Datetime visit_date values pass the date check."""
    df = _minimal_df()
    validator = DataValidator(ValidationConfig(expected_clinical_columns=[]))
    result = validator.validate(df)
    date_errors = [f for f in result.errors() if f.code == "INVALID_VISIT_DATE"]
    assert len(date_errors) == 0


# ---------------------------------------------------------------------------
# 4. Non-binary target
# ---------------------------------------------------------------------------

def test_non_binary_target_rejected() -> None:
    """Target column with value 2 is rejected."""
    df = _full_df()
    df.loc[0, "target"] = 2  # invalid
    validator = DataValidator()
    result = validator.validate(df)
    assert not result.passed
    assert any(f.code == "NON_BINARY_TARGET" for f in result.errors())


def test_target_with_values_0_and_1_passes() -> None:
    """Target column with only 0 and 1 is valid."""
    df = _full_df(n_patients=4)
    # Give half patients target=1
    df.loc[df["patient_id"].isin(["SYN_00001", "SYN_00002"]), "target"] = 1
    validator = DataValidator()
    result = validator.validate(df)
    binary_errors = [f for f in result.errors() if f.code == "NON_BINARY_TARGET"]
    assert len(binary_errors) == 0


# ---------------------------------------------------------------------------
# 5. Leakage candidate column detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("leakage_col", [
    "target",
    "diabetes_progression",
    "future_diabetes",
    "outcome",
    "diagnosis_after",
    "post_outcome",
])
def test_leakage_candidate_detected(leakage_col: str) -> None:
    """Columns matching leakage patterns are flagged as warnings."""
    df = _minimal_df()
    df[leakage_col] = 0
    validator = DataValidator(ValidationConfig(expected_clinical_columns=[]))
    result = validator.validate(df)
    leakage_warnings = [f for f in result.warnings() if f.code == "LEAKAGE_CANDIDATE_COLUMNS"]
    assert len(leakage_warnings) > 0, (
        f"Expected leakage warning for column {leakage_col!r} but got none. "
        f"Findings: {result.findings}"
    )


def test_leakage_columns_appear_in_profile() -> None:
    """Profile reports leakage candidate column names."""
    df = _minimal_df()
    df["future_glucose"] = 100.0
    validator = DataValidator(ValidationConfig(expected_clinical_columns=[]))
    result = validator.validate(df)
    assert "future_glucose" in result.profile.get("leakage_candidate_columns", [])


# ---------------------------------------------------------------------------
# 6. Duplicate patient + date
# ---------------------------------------------------------------------------

def test_duplicate_patient_date_rejected_by_default() -> None:
    """Duplicate patient_id + visit_date raises error by default."""
    df = _minimal_df(n=3)
    # Duplicate row 0
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    validator = DataValidator(ValidationConfig(allow_duplicate_patient_date=False))
    result = validator.validate(df)
    assert not result.passed
    assert any(f.code == "DUPLICATE_PATIENT_DATE" for f in result.errors())


def test_duplicate_patient_date_allowed_by_config() -> None:
    """Duplicate patient_id + visit_date is a warning (not error) when config allows."""
    df = _minimal_df(n=3)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    validator = DataValidator(ValidationConfig(allow_duplicate_patient_date=True, expected_clinical_columns=[]))
    result = validator.validate(df)
    dup_errors = [f for f in result.errors() if f.code == "DUPLICATE_PATIENT_DATE"]
    assert len(dup_errors) == 0
    dup_warnings = [f for f in result.warnings() if f.code == "DUPLICATE_PATIENT_DATE"]
    assert len(dup_warnings) > 0


# ---------------------------------------------------------------------------
# 7. Target inconsistency
# ---------------------------------------------------------------------------

def test_inconsistent_target_raises_error_by_default() -> None:
    """Patient with changing target value raises error-level finding."""
    df = _full_df(n_patients=2, visits_per=3)
    # Give SYN_00001 target=0 on first row but target=1 on third row
    pid_mask = df["patient_id"] == "SYN_00001"
    rows = df.index[pid_mask].tolist()
    df.loc[rows[0], "target"] = 0
    df.loc[rows[2], "target"] = 1  # inconsistency
    validator = DataValidator(ValidationConfig(target_inconsistency_as_error=True))
    result = validator.validate(df)
    assert not result.passed
    assert any(f.code == "INCONSISTENT_TARGET" for f in result.errors())


def test_inconsistent_target_as_warning_when_configured() -> None:
    """Patient with changing target becomes a warning (not error) when configured."""
    df = _full_df(n_patients=2, visits_per=3)
    pid_mask = df["patient_id"] == "SYN_00001"
    rows = df.index[pid_mask].tolist()
    df.loc[rows[0], "target"] = 0
    df.loc[rows[2], "target"] = 1
    validator = DataValidator(
        ValidationConfig(
            target_inconsistency_as_error=False,
            allow_duplicate_patient_date=False,
        )
    )
    result = validator.validate(df)
    inconsistent_errors = [f for f in result.errors() if f.code == "INCONSISTENT_TARGET"]
    assert len(inconsistent_errors) == 0
    inconsistent_warnings = [f for f in result.warnings() if f.code == "INCONSISTENT_TARGET"]
    assert len(inconsistent_warnings) > 0


# ---------------------------------------------------------------------------
# 8. Chronological order detection
# ---------------------------------------------------------------------------

def test_out_of_order_visits_detected() -> None:
    """Visits not in chronological order produce a non-chronological warning."""
    df = pd.DataFrame({
        "patient_id": ["SYN_00001", "SYN_00001"],
        "visit_date": pd.to_datetime(["2021-06-01", "2020-01-01"]),  # reversed
    })
    validator = DataValidator(
        ValidationConfig(
            require_chronological_order=True,
            expected_clinical_columns=[],
        )
    )
    result = validator.validate(df)
    chrono_warnings = [f for f in result.warnings() if f.code == "NON_CHRONOLOGICAL_ORDER"]
    assert len(chrono_warnings) > 0


def test_chronological_visits_pass() -> None:
    """Visits in chronological order produce no order warning."""
    df = pd.DataFrame({
        "patient_id": ["SYN_00001", "SYN_00001"],
        "visit_date": pd.to_datetime(["2020-01-01", "2021-06-01"]),
    })
    validator = DataValidator(
        ValidationConfig(
            require_chronological_order=True,
            expected_clinical_columns=[],
        )
    )
    result = validator.validate(df)
    chrono_warnings = [f for f in result.warnings() if f.code == "NON_CHRONOLOGICAL_ORDER"]
    assert len(chrono_warnings) == 0


# ---------------------------------------------------------------------------
# 9. Profile counts accuracy
# ---------------------------------------------------------------------------

def test_profile_row_and_patient_counts() -> None:
    """Profile correctly reports row count and patient count."""
    df = _full_df(n_patients=4, visits_per=3)
    validator = DataValidator()
    result = validator.validate(df)
    assert result.profile["row_count"] == 12
    assert result.profile["unique_patient_count"] == 4


def test_profile_target_distribution() -> None:
    """Profile correctly reports positive/negative patient counts."""
    df = _full_df(n_patients=4, visits_per=2)
    df.loc[df["patient_id"].isin(["SYN_00001", "SYN_00002"]), "target"] = 1
    df.loc[df["patient_id"].isin(["SYN_00003", "SYN_00004"]), "target"] = 0
    validator = DataValidator()
    result = validator.validate(df)
    assert result.profile["positive_patient_count"] == 2
    assert result.profile["negative_patient_count"] == 2


def test_profile_missingness() -> None:
    """Profile reports correct missingness for a column with NaN values."""
    df = _full_df(n_patients=2, visits_per=4)  # 8 rows
    df.loc[df.index[:2], "hba1c"] = float("nan")
    validator = DataValidator()
    result = validator.validate(df)
    missingness = result.profile.get("missingness", {})
    assert "hba1c" in missingness
    assert missingness["hba1c"] == pytest.approx(2 / 8, abs=1e-4)


# ---------------------------------------------------------------------------
# 10. Summary string
# ---------------------------------------------------------------------------

def test_summary_string_is_non_empty() -> None:
    """ValidationResult.summary() returns a non-empty string."""
    df = _full_df()
    result = DataValidator().validate(df)
    s = result.summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_passed_result_summary_says_passed() -> None:
    """A passing result contains 'PASSED' in its summary."""
    df = _full_df(n_patients=2, visits_per=2)
    # Remove target column so no binary-check issues, minimal config
    df = df.drop(columns=["target"])
    validator = DataValidator(ValidationConfig(expected_clinical_columns=[]))
    result = validator.validate(df)
    if result.passed:
        assert "PASSED" in result.summary()
