"""Tests for patient-level splitting (Phase 3).

Covers:
- no patient overlap between splits
- correct approximate proportions
- reproducibility (same seed → same split)
- different seed → different split
- stratification (both classes present in each split)
- failure when splitting into impossible distributions
- chronological split ordering
- SplitConfig validation
"""

from __future__ import annotations

import pandas as pd
import pytest

from dialong_automl.data.splitter import PatientSplitter, SplitConfig, SplitResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs_df(patient_ids: list[str], visits_per: int = 3) -> pd.DataFrame:
    """Build an observation DataFrame for the given patient IDs."""
    rows = []
    for idx, pid in enumerate(patient_ids):
        start = pd.Timestamp("2020-01-01") + pd.DateOffset(days=idx * 5)
        for v in range(visits_per):
            rows.append({
                "patient_id": pid,
                "visit_date": start + pd.DateOffset(days=v * 90),
                "hba1c": 5.5,
                "age_at_visit": 45.0,
            })
    return pd.DataFrame(rows)


def _make_targets(patient_ids: list[str], positive_fraction: float = 0.3) -> pd.Series:
    """Assign binary targets: positive_fraction of patients get target=1."""
    n_pos = max(1, int(len(patient_ids) * positive_fraction))
    targets = {pid: (1 if i < n_pos else 0) for i, pid in enumerate(patient_ids)}
    return pd.Series(targets, name="target")


def _n_patients(n: int = 100, positive_fraction: float = 0.3):
    """Return (obs_df, targets) for n patients."""
    pids = [f"SYN_{i:05d}" for i in range(1, n + 1)]
    return _make_obs_df(pids), _make_targets(pids, positive_fraction)


# ---------------------------------------------------------------------------
# 1. No patient overlap
# ---------------------------------------------------------------------------

def test_no_patient_overlap_default() -> None:
    """No patient appears in more than one split."""
    obs_df, targets = _n_patients(100)
    result = PatientSplitter().split(obs_df, targets)
    result.assert_no_overlap()  # raises AssertionError on overlap


def test_no_patient_overlap_explicit() -> None:
    """Explicit set intersection confirms zero overlap."""
    obs_df, targets = _n_patients(60)
    result = PatientSplitter(SplitConfig(seed=42)).split(obs_df, targets)

    train_set = set(result.train_ids)
    val_set = set(result.validation_ids)
    test_set = set(result.test_ids)

    assert len(train_set & val_set) == 0, "train ∩ validation must be empty"
    assert len(train_set & test_set) == 0, "train ∩ test must be empty"
    assert len(val_set & test_set) == 0, "validation ∩ test must be empty"


def test_all_patients_accounted_for() -> None:
    """Every patient in obs_df appears in exactly one split."""
    obs_df, targets = _n_patients(80)
    result = PatientSplitter().split(obs_df, targets)

    all_in_splits = set(result.train_ids) | set(result.validation_ids) | set(result.test_ids)
    all_in_df = set(obs_df["patient_id"].unique())
    # Intersection of cohort patients should be fully covered
    cohort_patients = all_in_df & set(targets.index)
    assert cohort_patients == all_in_splits


# ---------------------------------------------------------------------------
# 2. Correct approximate proportions
# ---------------------------------------------------------------------------

def test_approximate_train_proportion() -> None:
    """Training set is approximately 70% of patients (±5%)."""
    obs_df, targets = _n_patients(200)
    result = PatientSplitter(SplitConfig(train_size=0.70, seed=42)).split(obs_df, targets)
    actual_frac = len(result.train_ids) / 200
    assert 0.65 <= actual_frac <= 0.75, f"Train fraction {actual_frac:.3f} out of range"


def test_approximate_val_proportion() -> None:
    """Validation set is approximately 15% (±5%)."""
    obs_df, targets = _n_patients(200)
    result = PatientSplitter(SplitConfig(validation_size=0.15, seed=42)).split(obs_df, targets)
    actual_frac = len(result.validation_ids) / 200
    assert 0.10 <= actual_frac <= 0.20, f"Val fraction {actual_frac:.3f} out of range"


def test_approximate_test_proportion() -> None:
    """Test set is approximately 15% (±5%)."""
    obs_df, targets = _n_patients(200)
    result = PatientSplitter(SplitConfig(test_size=0.15, seed=42)).split(obs_df, targets)
    actual_frac = len(result.test_ids) / 200
    assert 0.10 <= actual_frac <= 0.20, f"Test fraction {actual_frac:.3f} out of range"


# ---------------------------------------------------------------------------
# 3. Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_split() -> None:
    """Same seed always produces identical splits."""
    obs_df, targets = _n_patients(100)
    cfg = SplitConfig(seed=42)
    result_a = PatientSplitter(cfg).split(obs_df, targets)
    result_b = PatientSplitter(cfg).split(obs_df, targets)
    assert result_a.train_ids == result_b.train_ids
    assert result_a.validation_ids == result_b.validation_ids
    assert result_a.test_ids == result_b.test_ids


def test_different_seed_different_split() -> None:
    """Different seeds produce different splits (with high probability for n≥50)."""
    obs_df, targets = _n_patients(100)
    result_42 = PatientSplitter(SplitConfig(seed=42)).split(obs_df, targets)
    result_99 = PatientSplitter(SplitConfig(seed=99)).split(obs_df, targets)
    # With 100 patients, it is essentially impossible for the splits to be identical
    assert result_42.train_ids != result_99.train_ids, (
        "Different seeds should produce different train sets"
    )


# ---------------------------------------------------------------------------
# 4. Stratification — both classes present
# ---------------------------------------------------------------------------

def test_both_classes_in_train_split() -> None:
    """Training split contains both positive and negative patients."""
    obs_df, targets = _n_patients(100, positive_fraction=0.3)
    result = PatientSplitter(SplitConfig(stratify=True, seed=42)).split(obs_df, targets)
    n_pos = int((result.train_targets == 1).sum())
    n_neg = int((result.train_targets == 0).sum())
    assert n_pos > 0, "Training set must contain positive patients"
    assert n_neg > 0, "Training set must contain negative patients"


def test_both_classes_in_validation_split() -> None:
    """Validation split contains both positive and negative patients."""
    obs_df, targets = _n_patients(100, positive_fraction=0.3)
    result = PatientSplitter(SplitConfig(stratify=True, seed=42)).split(obs_df, targets)
    n_pos = int((result.validation_targets == 1).sum())
    n_neg = int((result.validation_targets == 0).sum())
    assert n_pos > 0, "Validation set must contain positive patients"
    assert n_neg > 0, "Validation set must contain negative patients"


def test_both_classes_in_test_split() -> None:
    """Test split contains both positive and negative patients."""
    obs_df, targets = _n_patients(100, positive_fraction=0.3)
    result = PatientSplitter(SplitConfig(stratify=True, seed=42)).split(obs_df, targets)
    n_pos = int((result.test_targets == 1).sum())
    n_neg = int((result.test_targets == 0).sum())
    assert n_pos > 0, "Test set must contain positive patients"
    assert n_neg > 0, "Test set must contain negative patients"


# ---------------------------------------------------------------------------
# 5. Failure on impossible distributions
# ---------------------------------------------------------------------------

def test_raises_when_too_few_patients() -> None:
    """Splitter raises ValueError when there are fewer than 3 patients."""
    pids = ["SYN_00001", "SYN_00002"]
    obs_df = _make_obs_df(pids)
    targets = pd.Series({"SYN_00001": 0, "SYN_00002": 1}, name="target")
    with pytest.raises(ValueError, match="at least 3"):
        PatientSplitter().split(obs_df, targets)


def test_raises_when_stratify_impossible() -> None:
    """Raises ValueError when stratify=True but a split cannot contain both classes."""
    # 3 patients: 2 positive, 1 negative — with 70/15/15 split and stratify,
    # the test or validation set may only get one class
    pids = [f"SYN_{i:05d}" for i in range(1, 4)]
    obs_df = _make_obs_df(pids)
    targets = pd.Series(
        {"SYN_00001": 1, "SYN_00002": 1, "SYN_00003": 0},
        name="target",
    )
    with pytest.raises(ValueError):
        PatientSplitter(SplitConfig(stratify=True, seed=42)).split(obs_df, targets)


def test_split_config_sizes_must_sum_to_one() -> None:
    """SplitConfig raises ValueError when sizes do not sum to 1.0."""
    with pytest.raises(ValueError, match="must equal 1.0"):
        SplitConfig(train_size=0.6, validation_size=0.2, test_size=0.1)


def test_split_config_rejects_zero_size() -> None:
    """SplitConfig raises ValueError when a split size is 0."""
    with pytest.raises(ValueError):
        SplitConfig(train_size=0.85, validation_size=0.15, test_size=0.0)


# ---------------------------------------------------------------------------
# 6. DataFrame slicing correctness
# ---------------------------------------------------------------------------

def test_train_df_contains_only_train_patients() -> None:
    """train_df rows all belong to train_ids."""
    obs_df, targets = _n_patients(60)
    result = PatientSplitter().split(obs_df, targets)
    train_patient_set = set(result.train_ids)
    assert set(result.train_df["patient_id"].unique()) == train_patient_set


def test_validation_df_contains_only_validation_patients() -> None:
    """validation_df rows all belong to validation_ids."""
    obs_df, targets = _n_patients(60)
    result = PatientSplitter().split(obs_df, targets)
    val_patient_set = set(result.validation_ids)
    assert set(result.validation_df["patient_id"].unique()) == val_patient_set


def test_test_df_contains_only_test_patients() -> None:
    """test_df rows all belong to test_ids."""
    obs_df, targets = _n_patients(60)
    result = PatientSplitter().split(obs_df, targets)
    test_patient_set = set(result.test_ids)
    assert set(result.test_df["patient_id"].unique()) == test_patient_set


# ---------------------------------------------------------------------------
# 7. Chronological split
# ---------------------------------------------------------------------------

def test_chronological_split_no_overlap() -> None:
    """Chronological split still produces non-overlapping sets."""
    obs_df, targets = _n_patients(90)
    cfg = SplitConfig(chronological=True, stratify=False, seed=42)
    result = PatientSplitter(cfg).split(obs_df, targets)
    result.assert_no_overlap()


def test_chronological_split_uses_first_visit_order() -> None:
    """Chronological split puts earliest patients in training, latest in test."""
    # Create patients with clearly separated first visit dates
    pids = [f"SYN_{i:05d}" for i in range(1, 31)]
    rows = []
    for idx, pid in enumerate(pids):
        # Each patient's first visit is 30 days apart
        first = pd.Timestamp("2018-01-01") + pd.DateOffset(days=idx * 30)
        rows.append({
            "patient_id": pid,
            "visit_date": first,
            "hba1c": 5.5,
            "age_at_visit": 45.0,
        })
        rows.append({
            "patient_id": pid,
            "visit_date": first + pd.DateOffset(days=90),
            "hba1c": 5.6,
            "age_at_visit": 45.25,
        })
        rows.append({
            "patient_id": pid,
            "visit_date": first + pd.DateOffset(days=180),
            "hba1c": 5.7,
            "age_at_visit": 45.5,
        })
    obs_df = pd.DataFrame(rows)
    # Give all patients a valid target
    targets = pd.Series({pid: (0 if i < 20 else 1) for i, pid in enumerate(pids)}, name="target")

    cfg = SplitConfig(chronological=True, stratify=False, seed=42)
    result = PatientSplitter(cfg).split(obs_df, targets)

    # The first patient (earliest first visit) should be in training
    assert "SYN_00001" in result.train_ids, (
        "Earliest patient should be in training set with chronological split"
    )
    # The last patient (latest first visit) should be in test
    assert "SYN_00030" in result.test_ids, (
        "Latest patient should be in test set with chronological split"
    )


# ---------------------------------------------------------------------------
# 8. Split stats
# ---------------------------------------------------------------------------

def test_split_stats_total_patients() -> None:
    """split_stats total_patients matches the total in obs_df ∩ targets."""
    obs_df, targets = _n_patients(100)
    result = PatientSplitter().split(obs_df, targets)
    assert result.split_stats["total_patients"] == 100


def test_summary_string_non_empty() -> None:
    """SplitResult.summary() returns a non-empty string."""
    obs_df, targets = _n_patients(60)
    result = PatientSplitter().split(obs_df, targets)
    s = result.summary()
    assert isinstance(s, str)
    assert "train" in s.lower()
    assert "validation" in s.lower()
    assert "test" in s.lower()
