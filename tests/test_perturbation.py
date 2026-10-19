"""Phase 11 — perturbation interface tests.

Safety guarantees verified: determinism, no source mutation, label/layout
preservation, patient-set (split) preservation, and no future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dialong_automl.evaluation.perturbation import (
    jitter_visit_timing,
    remove_measurements,
    remove_visits,
    thin_followup,
    truncate_to_recent_visits,
)
from dialong_automl.features.wide import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _obs(n_patients: int = 6, visits_per: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n_patients):
        pid = f"SYN_{i:03d}"
        for v in range(visits_per):
            row = {
                "patient_id": pid,
                "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(days=90 * v),
            }
            for c in NUMERIC_FEATURES:
                row[c] = 5.0 + i + v
            for c in CATEGORICAL_FEATURES:
                row[c] = "yes"
            rows.append(row)
    return pd.DataFrame(rows)


def _layout(df: pd.DataFrame) -> list[str]:
    return list(df.columns)


# --- remove_measurements ---------------------------------------------------

def test_remove_measurements_does_not_mutate_source():
    df = _obs()
    before = df.copy(deep=True)
    remove_measurements(df, fraction=0.5, seed=1)
    pd.testing.assert_frame_equal(df, before)  # source untouched


def test_remove_measurements_preserves_layout_and_patients():
    df = _obs()
    res = remove_measurements(df, fraction=0.5, seed=1)
    out = res.observation_data
    assert _layout(out) == _layout(df)  # columns unchanged (NaN in place)
    assert set(out["patient_id"]) == set(df["patient_id"])  # all patients kept
    assert len(out) == len(df)  # no rows dropped
    assert res.condition["kind"] == "remove_measurements"
    assert res.condition["strength"] == 0.5


def test_remove_measurements_deterministic():
    df = _obs()
    a = remove_measurements(df, fraction=0.3, seed=7).observation_data
    b = remove_measurements(df, fraction=0.3, seed=7).observation_data
    pd.testing.assert_frame_equal(a, b)
    c = remove_measurements(df, fraction=0.3, seed=8).observation_data
    assert not a.equals(c)  # different seed → different blanking


def test_remove_measurements_zero_fraction_is_noop():
    df = _obs()
    out = remove_measurements(df, fraction=0.0, seed=1).observation_data
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), df.reset_index(drop=True)
    )


def test_remove_measurements_blanks_as_nan():
    df = _obs()
    out = remove_measurements(df, fraction=1.0, seed=1).observation_data
    # fraction=1.0 → all clinical cells blanked.
    for c in NUMERIC_FEATURES:
        assert out[c].isna().all()


# --- remove_visits ---------------------------------------------------------

def test_remove_visits_keeps_all_patients():
    df = _obs(n_patients=5, visits_per=4)
    res = remove_visits(df, fraction=0.5, seed=3, min_visits_kept=1)
    out = res.observation_data
    assert set(out["patient_id"]) == set(df["patient_id"])  # no patient removed
    # Each patient keeps >= 1 visit.
    assert (out.groupby("patient_id").size() >= 1).all()
    assert res.condition["params"]["patients_after"] == df["patient_id"].nunique()


def test_remove_visits_does_not_mutate_source():
    df = _obs()
    before = df.copy(deep=True)
    remove_visits(df, fraction=0.5, seed=2)
    pd.testing.assert_frame_equal(df, before)


def test_remove_visits_deterministic():
    df = _obs()
    a = remove_visits(df, fraction=0.5, seed=5).observation_data
    b = remove_visits(df, fraction=0.5, seed=5).observation_data
    pd.testing.assert_frame_equal(a, b)


def test_remove_visits_preserves_layout():
    df = _obs()
    out = remove_visits(df, fraction=0.5, seed=1).observation_data
    assert _layout(out) == _layout(df)


# --- thin_followup ---------------------------------------------------------

def test_thin_followup_keeps_last_n_and_all_patients():
    df = _obs(n_patients=4, visits_per=5)
    res = thin_followup(df, keep_last_n=2)
    out = res.observation_data
    assert set(out["patient_id"]) == set(df["patient_id"])
    assert (out.groupby("patient_id").size() == 2).all()
    # Kept visits are the most recent ones (no future info introduced; only
    # existing later visits retained).
    for pid, grp in df.groupby("patient_id"):
        kept = out[out["patient_id"] == pid]["visit_date"].tolist()
        expected = grp.sort_values("visit_date")["visit_date"].tolist()[-2:]
        assert kept == expected


def test_thin_followup_does_not_mutate_source():
    df = _obs()
    before = df.copy(deep=True)
    thin_followup(df, keep_last_n=2)
    pd.testing.assert_frame_equal(df, before)


def test_thin_followup_fewer_visits_unchanged():
    df = _obs(n_patients=2, visits_per=2)
    out = thin_followup(df, keep_last_n=5).observation_data
    assert len(out) == len(df)  # keep_last_n > visits → all kept


# --- input validation ------------------------------------------------------

def test_invalid_args():
    df = _obs()
    with pytest.raises(ValueError):
        remove_measurements(df, fraction=1.5)
    with pytest.raises(ValueError):
        remove_visits(df, fraction=-0.1)
    with pytest.raises(ValueError):
        thin_followup(df, keep_last_n=0)
    with pytest.raises(ValueError):
        remove_measurements(pd.DataFrame({"x": [1]}), fraction=0.5)  # missing contract cols


# --- label preservation (labels live separately; perturbation never sees them) ---

def test_perturbation_operates_on_features_only():
    """A target column, if present, is treated as a normal column and is NOT a
    clinical feature — perturbation must not alter patient identity. We assert
    that the patient→(implicit label) mapping is unaffected by verifying the
    patient set and ordering of a separate label Series stays valid."""
    df = _obs(n_patients=4, visits_per=3)
    labels = pd.Series({pid: i % 2 for i, pid in enumerate(df["patient_id"].unique())})
    out = remove_visits(df, fraction=0.5, seed=1).observation_data
    # Every patient in the perturbed data still has a label (mapping intact).
    assert set(out["patient_id"]).issubset(set(labels.index))
    assert set(out["patient_id"]) == set(labels.index)


# ===========================================================================
# Phase 11 review fixes — per-patient determinism / cross-cohort invariance
# ===========================================================================

def _variable_visit_cohort() -> pd.DataFrame:
    """Cohort where patients have DIFFERENT visit counts (to expose the old
    shared-RNG coupling in remove_visits)."""
    rows = []
    counts = {"SYN_000": 3, "SYN_001": 6, "SYN_002": 4}
    for pid, n in counts.items():
        for v in range(n):
            row = {
                "patient_id": pid,
                "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(days=30 * v),
            }
            for c in NUMERIC_FEATURES:
                row[c] = 5.0 + v
            for c in CATEGORICAL_FEATURES:
                row[c] = "yes"
            rows.append(row)
    return pd.DataFrame(rows)


def _kept_dates(df: pd.DataFrame, pid: str) -> list:
    return sorted(df[df["patient_id"] == pid]["visit_date"].tolist())


# --- remove_visits: cross-cohort invariance (regression for review finding 1) ---

def test_remove_visits_patient_pattern_independent_of_other_patients():
    """The visit-drop pattern for a given patient must NOT depend on how many
    visits unrelated patients have. This is the bug flagged in the Phase 11
    review (a single shared RNG made patient N depend on preceding patients)."""
    cohort_a = _variable_visit_cohort()

    # cohort_b: same SYN_001 rows, but SYN_000 has a different visit count and
    # an extra unrelated patient is inserted before it.
    rows = []
    for v in range(2):  # SYN_000 now has 2 visits instead of 3
        row = {"patient_id": "SYN_000",
               "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(days=30 * v)}
        for c in NUMERIC_FEATURES:
            row[c] = 5.0 + v
        for c in CATEGORICAL_FEATURES:
            row[c] = "yes"
        rows.append(row)
    for v in range(5):  # an extra unrelated patient
        row = {"patient_id": "SYN_099",
               "visit_date": pd.Timestamp("2021-01-01") + pd.DateOffset(days=30 * v)}
        for c in NUMERIC_FEATURES:
            row[c] = 1.0 + v
        for c in CATEGORICAL_FEATURES:
            row[c] = "no"
        rows.append(row)
    # SYN_001 identical to cohort_a.
    rows.extend(cohort_a[cohort_a["patient_id"] == "SYN_001"].to_dict("records"))
    cohort_b = pd.DataFrame(rows)

    out_a = remove_visits(cohort_a, fraction=0.5, seed=123, min_visits_kept=1).observation_data
    out_b = remove_visits(cohort_b, fraction=0.5, seed=123, min_visits_kept=1).observation_data

    # SYN_001 is identical in both cohorts → must get the SAME kept visits.
    assert _kept_dates(out_a, "SYN_001") == _kept_dates(out_b, "SYN_001")


def test_remove_visits_single_patient_subset_matches_full_cohort():
    """Perturbing a patient alone must match perturbing them within a cohort."""
    full = _variable_visit_cohort()
    just_one = full[full["patient_id"] == "SYN_001"].reset_index(drop=True)

    out_full = remove_visits(full, fraction=0.5, seed=7).observation_data
    out_one = remove_visits(just_one, fraction=0.5, seed=7).observation_data

    assert _kept_dates(out_full, "SYN_001") == _kept_dates(out_one, "SYN_001")


def test_remove_visits_respects_min_visits_kept():
    df = _variable_visit_cohort()
    out = remove_visits(df, fraction=1.0, seed=1, min_visits_kept=2).observation_data
    assert (out.groupby("patient_id").size() >= 2).all()
    assert set(out["patient_id"]) == set(df["patient_id"])


# --- remove_measurements: invariance to ordering + unrelated patients ---

def _blanked_mask_by_identity(res_df: pd.DataFrame) -> dict:
    """Map (patient_id, visit_date, feature) -> is-NaN, for comparison across
    different row/column orderings."""
    mask = {}
    for _, row in res_df.iterrows():
        for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            if c in res_df.columns:
                mask[(row["patient_id"], row["visit_date"], c)] = pd.isna(row[c])
    return mask


def test_remove_measurements_invariant_to_row_order():
    df = _obs(n_patients=5, visits_per=4)
    shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    a = remove_measurements(df, fraction=0.4, seed=11).observation_data
    b = remove_measurements(shuffled, fraction=0.4, seed=11).observation_data
    assert _blanked_mask_by_identity(a) == _blanked_mask_by_identity(b)


def test_remove_measurements_invariant_to_column_order():
    df = _obs(n_patients=4, visits_per=3)
    reordered = df[list(reversed(df.columns))]
    a = remove_measurements(df, fraction=0.4, seed=11).observation_data
    b = remove_measurements(reordered, fraction=0.4, seed=11).observation_data
    assert _blanked_mask_by_identity(a) == _blanked_mask_by_identity(b)


def test_remove_measurements_unrelated_patient_does_not_change_mask():
    df = _obs(n_patients=3, visits_per=3)
    extra = _obs(n_patients=6, visits_per=3)  # superset with same first 3 patients
    a = remove_measurements(df, fraction=0.5, seed=5).observation_data
    b = remove_measurements(extra, fraction=0.5, seed=5).observation_data
    mask_a = _blanked_mask_by_identity(a)
    mask_b = {k: v for k, v in _blanked_mask_by_identity(b).items()
              if k[0] in set(df["patient_id"])}
    assert mask_a == mask_b


def test_remove_measurements_same_seed_identical():
    df = _obs()
    a = remove_measurements(df, fraction=0.3, seed=7).observation_data
    b = remove_measurements(df, fraction=0.3, seed=7).observation_data
    pd.testing.assert_frame_equal(a, b)


# --- truncate_to_recent_visits (renamed) + thin_followup alias ---

def test_truncate_to_recent_visits_keeps_last_n():
    df = _obs(n_patients=4, visits_per=5)
    res = truncate_to_recent_visits(df, keep_last_n=2)
    out = res.observation_data
    assert (out.groupby("patient_id").size() == 2).all()
    assert res.condition["kind"] == "truncate_to_recent_visits"
    assert "seed" not in res.condition["params"]  # no randomness


def test_thin_followup_alias_matches_truncate():
    df = _obs(n_patients=4, visits_per=5)
    a = thin_followup(df, keep_last_n=2).observation_data
    b = truncate_to_recent_visits(df, keep_last_n=2).observation_data
    pd.testing.assert_frame_equal(a, b)


def test_thin_followup_seed_is_ignored():
    df = _obs(n_patients=3, visits_per=4)
    a = thin_followup(df, keep_last_n=2, seed=1).observation_data
    b = thin_followup(df, keep_last_n=2, seed=99).observation_data
    pd.testing.assert_frame_equal(a, b)  # seed has no effect


# --- jitter_visit_timing (new timing operator) ---

def test_jitter_visit_timing_preserves_patients_and_visit_counts():
    df = _obs(n_patients=4, visits_per=4)
    res = jitter_visit_timing(df, max_shift_days=10, seed=3)
    out = res.observation_data
    assert set(out["patient_id"]) == set(df["patient_id"])
    assert (out.groupby("patient_id").size()
            == df.groupby("patient_id").size()).all()
    assert res.condition["kind"] == "jitter_visit_timing"


def test_jitter_visit_timing_preserves_order_and_window():
    df = _obs(n_patients=3, visits_per=5)
    out = jitter_visit_timing(df, max_shift_days=15, seed=4).observation_data
    for pid, grp in df.groupby("patient_id"):
        orig = pd.to_datetime(grp.sort_values("visit_date")["visit_date"]).tolist()
        new = pd.to_datetime(
            out[out["patient_id"] == pid].sort_values("visit_date")["visit_date"]
        ).tolist()
        # Non-decreasing order preserved.
        assert all(new[i] <= new[i + 1] for i in range(len(new) - 1))
        # Stays within the original observed window (never crosses it).
        assert min(new) >= min(orig)
        assert max(new) <= max(orig)


def test_jitter_visit_timing_deterministic_and_per_patient():
    df = _obs(n_patients=4, visits_per=4)
    a = jitter_visit_timing(df, max_shift_days=10, seed=3).observation_data
    b = jitter_visit_timing(df, max_shift_days=10, seed=3).observation_data
    pd.testing.assert_frame_equal(a, b)
    # Zero shift is a no-op on dates.
    out0 = jitter_visit_timing(df, max_shift_days=0, seed=3).observation_data
    pd.testing.assert_frame_equal(
        out0.sort_values(["patient_id", "visit_date"]).reset_index(drop=True),
        df.sort_values(["patient_id", "visit_date"]).reset_index(drop=True),
    )


def test_jitter_visit_timing_does_not_mutate_source():
    df = _obs()
    before = df.copy(deep=True)
    jitter_visit_timing(df, max_shift_days=10, seed=1)
    pd.testing.assert_frame_equal(df, before)


def test_jitter_visit_timing_invalid_arg():
    df = _obs()
    with pytest.raises(ValueError):
        jitter_visit_timing(df, max_shift_days=-1)
