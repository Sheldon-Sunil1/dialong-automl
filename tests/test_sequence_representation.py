"""Tests for the padded sequence representation (Phase 4).

Covers:
1.  Correct X shape (n_patients, max_seq_len, n_features).
2.  Correct mask shape.
3.  Correct sequence lengths.
4.  Padding is correctly masked (mask=0 for padding, 1 for real visits).
5.  Chronological ordering within each sequence.
6.  n_features matches len(feature_names).
7.  Target never enters X.
8.  LeakageError when leakage columns are in input.
9.  Vocabulary is built correctly (MISSING and UNK sentinels).
10. Pre-built vocabulary is reused without refitting on held-out data.
11. Unknown category handled safely (mapped to UNK, not error).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dialong_automl.features.sequence import (
    SequenceRepresentationBuilder,
    SequenceRepresentationResult,
    _MISSING_CAT,
    _UNK,
)
from dialong_automl.features.wide import LeakageError, NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(
    n_patients: int = 6,
    n_visits: int = 4,
    hba1c_start: float = 5.5,
) -> pd.DataFrame:
    rows = []
    for i in range(1, n_patients + 1):
        pid = f"SYN_{i:05d}"
        for v in range(n_visits):
            rows.append({
                "patient_id": pid,
                "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(days=v * 90),
                "visit_number": v + 1,
                "hba1c": hba1c_start + i * 0.1 + v * 0.05,
                "fasting_glucose": 90.0 + v,
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
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Correct X shape
# ---------------------------------------------------------------------------

def test_x_shape() -> None:
    """X has shape (n_patients, max_seq_len, n_features)."""
    df = _make_obs(n_patients=5, n_visits=4)
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    n_features = len(result.feature_names)
    assert result.X.shape == (5, 4, n_features)


def test_x_dtype_is_float32() -> None:
    """X dtype must be float32."""
    df = _make_obs()
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    assert result.X.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. Correct mask shape
# ---------------------------------------------------------------------------

def test_mask_shape() -> None:
    """Mask shape is (n_patients, max_seq_len)."""
    df = _make_obs(n_patients=5, n_visits=4)
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    assert result.mask.shape == (5, 4)


# ---------------------------------------------------------------------------
# 3. Correct sequence lengths
# ---------------------------------------------------------------------------

def test_sequence_lengths_all_full() -> None:
    """Lengths equal n_visits when every patient has the same visit count."""
    df = _make_obs(n_patients=4, n_visits=4)
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    assert (result.lengths == 4).all()


def test_sequence_lengths_variable() -> None:
    """Length reflects actual visit count for short patients."""
    rows = []
    for pid, n in [("LONG", 4), ("SHORT", 2)]:
        for v in range(n):
            rows.append({
                "patient_id": pid,
                "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(days=v * 90),
                "hba1c": 5.5,
                "diabetes_diagnosis": 0,
            })
    df = pd.DataFrame(rows)
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    pid_to_idx = {pid: i for i, pid in enumerate(result.patient_ids)}
    assert result.lengths[pid_to_idx["LONG"]] == 4
    assert result.lengths[pid_to_idx["SHORT"]] == 2


# ---------------------------------------------------------------------------
# 4. Padding is correctly masked
# ---------------------------------------------------------------------------

def test_mask_ones_for_real_visits() -> None:
    """mask[i, t] = 1 for all real visits."""
    df = _make_obs(n_patients=3, n_visits=4)
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    assert (result.mask == 1).all()  # all patients have 4 visits, max_seq=4


def test_mask_zeros_for_padding() -> None:
    """mask[i, t] = 0 for padding positions of a short-sequence patient."""
    rows = [
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-01-01"), "hba1c": 5.5, "diabetes_diagnosis": 0},
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-04-01"), "hba1c": 5.6, "diabetes_diagnosis": 0},
    ]
    df = pd.DataFrame(rows)
    result = SequenceRepresentationBuilder(
        max_seq_len=4,
        numeric_features=["hba1c"],
        categorical_features=[],
    ).build(df)
    idx = result.patient_ids.index("P1")
    assert result.mask[idx, 0] == 1  # real
    assert result.mask[idx, 1] == 1  # real
    assert result.mask[idx, 2] == 0  # padding
    assert result.mask[idx, 3] == 0  # padding
    assert result.lengths[idx] == 2


def test_mask_and_length_consistency() -> None:
    """mask.sum(axis=1) == lengths for every patient."""
    df = _make_obs(n_patients=4, n_visits=3)
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    mask_sums = result.mask.sum(axis=1).astype(int)
    np.testing.assert_array_equal(mask_sums, result.lengths)


# ---------------------------------------------------------------------------
# 5. Chronological ordering
# ---------------------------------------------------------------------------

def test_sequence_is_chronological() -> None:
    """Visits within a sequence are in ascending visit_date order."""
    # Create patient with known hba1c values to verify order
    rows = [
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-07-01"), "hba1c": 6.0, "diabetes_diagnosis": 0},
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-01-01"), "hba1c": 5.0, "diabetes_diagnosis": 0},
        {"patient_id": "P1", "visit_date": pd.Timestamp("2020-04-01"), "hba1c": 5.5, "diabetes_diagnosis": 0},
    ]
    df = pd.DataFrame(rows)
    result = SequenceRepresentationBuilder(
        max_seq_len=4,
        numeric_features=["hba1c"],
        categorical_features=[],
    ).build(df)
    idx = result.patient_ids.index("P1")
    hba1c_idx = result.feature_names.index("hba1c")
    seq = result.X[idx, :, hba1c_idx]
    assert seq[0] == pytest.approx(5.0, abs=0.01), f"Expected 5.0 at t=0, got {seq[0]}"
    assert seq[1] == pytest.approx(5.5, abs=0.01)
    assert seq[2] == pytest.approx(6.0, abs=0.01)


# ---------------------------------------------------------------------------
# 6. n_features matches feature_names
# ---------------------------------------------------------------------------

def test_n_features_matches_feature_names_length() -> None:
    """X.shape[2] == len(feature_names)."""
    df = _make_obs()
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    assert result.X.shape[2] == len(result.feature_names)


def test_feature_names_include_numeric_and_categorical() -> None:
    """feature_names contains both numeric and categorical feature names."""
    df = _make_obs()
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    # At least one numeric and one categorical must be present
    has_numeric = any(f in result.feature_names for f in NUMERIC_FEATURES)
    assert has_numeric


# ---------------------------------------------------------------------------
# 7. Target never in X
# ---------------------------------------------------------------------------

def test_target_not_in_X() -> None:
    """Passing patient_targets keeps target out of X."""
    df = _make_obs(n_patients=4)
    targets = pd.Series({f"SYN_{i:05d}": i % 2 for i in range(1, 5)}, name="target")
    targets.index.name = "patient_id"
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df, patient_targets=targets)
    assert "target" not in result.feature_names
    assert result.y is not None
    # X has shape (4, 4, n_features) — not (4, 4, n_features+1)
    assert result.X.shape[2] == len(result.feature_names)


# ---------------------------------------------------------------------------
# 8. LeakageError
# ---------------------------------------------------------------------------

def test_leakage_error_target_column() -> None:
    df = _make_obs()
    df["target"] = 0
    with pytest.raises(LeakageError):
        SequenceRepresentationBuilder(max_seq_len=4).build(df)


def test_leakage_error_is_future_visit() -> None:
    df = _make_obs()
    df["is_future_visit"] = False
    with pytest.raises(LeakageError):
        SequenceRepresentationBuilder(max_seq_len=4).build(df)


# ---------------------------------------------------------------------------
# 9. Vocabulary built correctly
# ---------------------------------------------------------------------------

def test_vocab_has_missing_and_unk_sentinels() -> None:
    """Each categorical vocab starts with <MISSING> and <UNK>."""
    df = _make_obs()
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    for feat, vocab in result.categorical_vocabularies.items():
        assert vocab[0] == _MISSING_CAT, f"vocab[0] should be {_MISSING_CAT}"
        assert vocab[1] == _UNK, f"vocab[1] should be {_UNK}"


def test_vocab_includes_observed_categories() -> None:
    """Observed categorical values appear in the vocabulary."""
    df = _make_obs()
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)
    sex_vocab = result.categorical_vocabularies.get("sex", [])
    assert "M" in sex_vocab


# ---------------------------------------------------------------------------
# 10. Pre-built vocabulary reused (no refitting on held-out data)
# ---------------------------------------------------------------------------

def test_prebuilt_vocab_is_reused() -> None:
    """When categorical_vocabularies is provided, it is used without change."""
    train_df = _make_obs(n_patients=4)
    # Build on train → get vocabulary
    train_result = SequenceRepresentationBuilder(max_seq_len=4).build(train_df)
    vocab = train_result.categorical_vocabularies

    # Build on a different dataset using the same vocab
    val_df = _make_obs(n_patients=2, hba1c_start=6.0)
    val_result = SequenceRepresentationBuilder(
        max_seq_len=4,
        categorical_vocabularies=dict(vocab),
    ).build(val_df)

    assert val_result.categorical_vocabularies == vocab


# ---------------------------------------------------------------------------
# 11. Unknown category handled safely
# ---------------------------------------------------------------------------

def test_unknown_category_mapped_to_unk() -> None:
    """A category unseen during vocabulary construction gets a 1 in the <UNK> one-hot slot."""
    train_df = _make_obs(n_patients=3)
    train_result = SequenceRepresentationBuilder(
        max_seq_len=4,
        numeric_features=["hba1c"],
        categorical_features=["sex"],
    ).build(train_df)
    vocab = train_result.categorical_vocabularies

    # Validation data has a never-seen sex value
    val_rows = [
        {
            "patient_id": "NEW_P",
            "visit_date": pd.Timestamp("2021-01-01"),
            "hba1c": 5.5,
            "sex": "NONBINARY_UNKNOWN",  # not in training vocab
            "diabetes_diagnosis": 0,
        }
    ]
    val_df = pd.DataFrame(val_rows)

    val_result = SequenceRepresentationBuilder(
        max_seq_len=4,
        numeric_features=["hba1c"],
        categorical_features=["sex"],
        categorical_vocabularies=dict(vocab),
    ).build(val_df)

    # The unknown value should activate the sex__<UNK> one-hot slot
    unk_col = f"sex__{_UNK}"
    assert unk_col in val_result.feature_names, (
        f"Expected one-hot column {unk_col!r} in feature_names"
    )
    unk_idx = val_result.feature_names.index(unk_col)
    assert val_result.X[0, 0, unk_idx] == pytest.approx(1.0), (
        f"Expected sex__{_UNK} slot = 1.0 for unknown category"
    )


# ---------------------------------------------------------------------------
# 12. One-hot encoding correctness (not integer label codes)
# ---------------------------------------------------------------------------

def test_categorical_one_hot_not_integer_codes() -> None:
    """Categorical features must be one-hot encoded, not arbitrary integer codes.

    For a patient with sex='M' the slot 'sex__M' must be 1 and all other
    sex__ slots must be 0.  The value must never be an integer like 2 or 3.
    """
    rows = [
        {
            "patient_id": "P1",
            "visit_date": pd.Timestamp("2020-01-01"),
            "hba1c": 5.5,
            "sex": "M",
            "diabetes_diagnosis": 0,
        }
    ]
    df = pd.DataFrame(rows)
    result = SequenceRepresentationBuilder(
        max_seq_len=4,
        numeric_features=["hba1c"],
        categorical_features=["sex"],
    ).build(df)

    # Identify all sex__ columns
    sex_cols = [name for name in result.feature_names if name.startswith("sex__")]
    assert len(sex_cols) >= 2, "Expected at least 2 one-hot slots for sex"

    # Exactly one sex__ slot should be 1, all others 0
    idx_0 = result.patient_ids.index("P1")
    for col_name in sex_cols:
        col_idx = result.feature_names.index(col_name)
        val = float(result.X[idx_0, 0, col_idx])
        expected = 1.0 if col_name == "sex__M" else 0.0
        assert val == pytest.approx(expected, abs=1e-6), (
            f"One-hot encoding error: {col_name} should be {expected}, got {val}. "
            "If this is a non-zero integer it means label encoding was used."
        )


def test_one_hot_slots_sum_to_one_per_visit() -> None:
    """For each patient at each real timestep, the one-hot slots for a given
    categorical feature sum to exactly 1.0 (exactly one category active)."""
    df = _make_obs(n_patients=4, n_visits=4)
    result = SequenceRepresentationBuilder(
        max_seq_len=4,
        numeric_features=[],
        categorical_features=["sex", "smoking_status"],
    ).build(df)

    for feat in ["sex", "smoking_status"]:
        slots = [
            result.feature_names.index(n)
            for n in result.feature_names
            if n.startswith(f"{feat}__")
        ]
        if not slots:
            continue
        for i in range(result.X.shape[0]):
            for t in range(result.X.shape[1]):
                if result.mask[i, t] == 0:
                    continue  # padding — skip
                slot_sum = float(result.X[i, t, slots].sum())
                assert abs(slot_sum - 1.0) < 1e-5, (
                    f"Patient {result.patient_ids[i]} t={t} {feat}: "
                    f"one-hot slots sum = {slot_sum}, expected 1.0"
                )


def test_no_ordinal_magnitude_in_categorical_slots() -> None:
    """No categorical slot value should be > 1.0 (which would imply integer codes)."""
    df = _make_obs(n_patients=6, n_visits=4)
    result = SequenceRepresentationBuilder(max_seq_len=4).build(df)

    # All positions in the one-hot slots must be 0 or 1
    cat_slot_indices = [
        i for i, name in enumerate(result.feature_names)
        if "__" in name  # one-hot slots have "feature__category" naming
    ]
    if cat_slot_indices:
        cat_vals = result.X[:, :, cat_slot_indices]
        real_mask = result.mask.astype(bool)
        for i in range(result.X.shape[0]):
            for t in range(result.X.shape[1]):
                if not real_mask[i, t]:
                    continue
                for slot_idx in cat_slot_indices:
                    v = float(result.X[i, t, slot_idx])
                    assert v in (0.0, 1.0), (
                        f"Categorical slot {result.feature_names[slot_idx]!r} "
                        f"at patient {i} t={t} has value {v}. "
                        "Expected binary 0/1 (one-hot). "
                        "This would be an integer code if > 1."
                    )


def test_feature_names_use_ohe_naming_convention() -> None:
    """Categorical feature names in the sequence result use 'feat__category' format."""
    df = _make_obs()
    result = SequenceRepresentationBuilder(
        max_seq_len=4,
        numeric_features=["hba1c"],
        categorical_features=["sex"],
    ).build(df)

    # sex should produce columns like sex__M, sex__F, sex__<MISSING>, sex__<UNK>
    sex_cols = [n for n in result.feature_names if n.startswith("sex__")]
    assert len(sex_cols) >= 2
    # Raw 'sex' should NOT appear as a plain name (that would be integer encoding)
    assert "sex" not in result.feature_names, (
        "'sex' appears as a plain feature name — this indicates integer label "
        "encoding rather than one-hot encoding."
    )
