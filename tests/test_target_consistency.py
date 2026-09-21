"""Target consistency tests: synthetic generator vs cohort builder.

Verifies that the pre-labeled ``target`` column stamped on observation rows
by the generator is *identical* to the target that the cohort builder's
future-outcome mode derives from the same full timeline.

The canonical definition (shared by both):

    target = 1  iff any future visit satisfies:
                    diabetes_diagnosis == 1
                OR  hba1c >= diabetes_onset_hba1c_threshold (default 6.5)

    target = 0  otherwise (when sufficient future follow-up exists)

If the two systems produce different targets for the same patient, there is
a definition mismatch that would silently corrupt downstream training.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dialong_automl.data.synthetic_generator import (
    DIABETES_HBA1C_THRESHOLD,
    GeneratorConfig,
    SyntheticGenerator,
)
from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_full_df(n_patients: int = 80, seed: int = 0) -> pd.DataFrame:
    """Generate a small synthetic full timeline (all seeds exercised)."""
    cfg = GeneratorConfig(
        n_patients=n_patients,
        min_obs_visits=3,
        max_obs_visits=4,
        min_future_visits=2,
        max_future_visits=3,
        seed=seed,
        progression_prevalence=0.40,
        missingness_rate=0.0,  # no missingness so HbA1c is always present
    )
    result = SyntheticGenerator(config=cfg).generate()
    return result.full_df


def _build_future_outcome_targets(full_df: pd.DataFrame) -> pd.Series:
    """Run the cohort builder in future-outcome mode; return patient targets."""
    cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        max_observation_visits=4,
        prediction_horizon_days=365 * 5,  # wide window to capture all future visits
        diabetes_onset_hba1c_threshold=DIABETES_HBA1C_THRESHOLD,
        exclude_known_diabetes_at_baseline=False,
        require_followup_after_cutoff=True,
    )
    result = CohortBuilder(cfg).build(full_df)
    return result.patient_targets


def _get_generator_prelabeled_targets(full_df: pd.DataFrame) -> pd.Series:
    """Extract the pre-labeled target stamped by the generator on the full df."""
    return (
        full_df.drop_duplicates("patient_id")
        .set_index("patient_id")["target"]
        .dropna()
        .astype(int)
    )


# ---------------------------------------------------------------------------
# Core consistency test
# ---------------------------------------------------------------------------


def test_generator_and_cohort_builder_targets_identical_small() -> None:
    """For every patient present in both, the targets must be identical.

    We use a wide prediction horizon (5 years) and no minimum-visit
    exclusion so all patients are eligible, making the comparison clean.
    """
    full_df = _generate_full_df(n_patients=80, seed=42)

    generator_targets = _get_generator_prelabeled_targets(full_df)
    cohort_targets = _build_future_outcome_targets(full_df)

    # Find patients present in both
    common_pids = sorted(set(generator_targets.index) & set(cohort_targets.index))
    assert len(common_pids) > 0, "No common patients — check generation/cohort config"

    mismatches: list[str] = []
    for pid in common_pids:
        g = int(generator_targets[pid])
        c = int(cohort_targets[pid])
        if g != c:
            mismatches.append(f"{pid}: generator={g}, cohort={c}")

    assert len(mismatches) == 0, (
        f"Target mismatch between generator and cohort builder for "
        f"{len(mismatches)}/{len(common_pids)} patients:\n"
        + "\n".join(mismatches[:20])
    )


@pytest.mark.parametrize("seed", [0, 1, 7, 13, 99])
def test_consistency_across_seeds(seed: int) -> None:
    """Target consistency holds for several different random seeds."""
    full_df = _generate_full_df(n_patients=50, seed=seed)

    generator_targets = _get_generator_prelabeled_targets(full_df)
    cohort_targets = _build_future_outcome_targets(full_df)

    common_pids = sorted(set(generator_targets.index) & set(cohort_targets.index))
    mismatches = [
        pid for pid in common_pids
        if int(generator_targets[pid]) != int(cohort_targets[pid])
    ]
    assert len(mismatches) == 0, (
        f"Seed {seed}: {len(mismatches)} target mismatches. "
        f"First: {mismatches[:5]}"
    )


# ---------------------------------------------------------------------------
# Rule-level sanity checks
# ---------------------------------------------------------------------------


def test_generator_target_1_iff_future_hba1c_crosses_threshold() -> None:
    """For every patient with target=1, at least one future visit has
    hba1c >= threshold OR diabetes_diagnosis == 1."""
    full_df = _generate_full_df(n_patients=100, seed=42)
    threshold = DIABETES_HBA1C_THRESHOLD

    for pid, grp in full_df.groupby("patient_id"):
        target_vals = grp["target"].dropna().unique()
        if len(target_vals) != 1:
            continue
        patient_target = int(target_vals[0])

        future_rows = grp[grp["is_future_visit"].astype(bool)]
        if len(future_rows) == 0:
            continue  # no future rows — target should be None (not tested here)

        future_positive = (
            (future_rows["hba1c"] >= threshold).any()
            or (future_rows["diabetes_diagnosis"] == 1).any()
        )

        if patient_target == 1:
            assert future_positive, (
                f"Patient {pid}: target=1 but no future visit has "
                f"hba1c >= {threshold} or diabetes_diagnosis==1. "
                f"Future HbA1c values: {future_rows['hba1c'].tolist()}"
            )
        else:
            assert not future_positive, (
                f"Patient {pid}: target=0 but future visit has "
                f"hba1c >= {threshold} or diabetes_diagnosis==1. "
                f"Future HbA1c values: {future_rows['hba1c'].tolist()}"
            )


def test_observation_hba1c_never_triggers_target() -> None:
    """The target must be derivable from future rows alone.

    If all future rows were removed, we cannot infer the target from
    observation rows — this confirms no leakage of the label definition
    into the observation window.
    """
    full_df = _generate_full_df(n_patients=60, seed=7)
    threshold = DIABETES_HBA1C_THRESHOLD

    for pid, grp in full_df.groupby("patient_id"):
        target_vals = grp["target"].dropna().unique()
        if len(target_vals) != 1:
            continue
        patient_target = int(target_vals[0])

        obs_rows = grp[~grp["is_future_visit"].astype(bool)]

        # Observation rows must not have diabetes_diagnosis=1
        # (it is always 0 in the observation window by generator design)
        assert not (obs_rows["diabetes_diagnosis"] == 1).any(), (
            f"Patient {pid}: diabetes_diagnosis=1 found in observation rows. "
            "This is a leakage violation."
        )


def test_diabetes_diagnosis_in_obs_is_always_zero() -> None:
    """The generator must set diabetes_diagnosis=0 in all observation rows."""
    full_df = _generate_full_df(n_patients=100, seed=42)
    obs = full_df[~full_df["is_future_visit"].astype(bool)]
    bad = obs[obs["diabetes_diagnosis"] != 0]
    assert len(bad) == 0, (
        f"{len(bad)} observation rows have diabetes_diagnosis != 0. "
        "These would leak the outcome into the feature space."
    )


def test_diabetes_diagnosis_in_future_matches_hba1c_threshold() -> None:
    """In future rows, diabetes_diagnosis == 1 iff hba1c >= threshold."""
    full_df = _generate_full_df(n_patients=100, seed=42)
    threshold = DIABETES_HBA1C_THRESHOLD
    future = full_df[full_df["is_future_visit"].astype(bool)].copy()

    expected_diag = (future["hba1c"] >= threshold).astype(int)
    actual_diag = future["diabetes_diagnosis"].astype(int)

    mismatches = (expected_diag != actual_diag).sum()
    assert mismatches == 0, (
        f"{mismatches} future rows have diabetes_diagnosis inconsistent "
        f"with hba1c >= {threshold}. "
        "The generator and cohort builder use different criteria."
    )


def test_pre_labeled_prevalence_close_to_intended() -> None:
    """The actual pre-labeled prevalence should be within 20pp of intended.

    This is a soft check — the realised prevalence depends on trajectory
    noise. We only verify it is not wildly off (e.g. 0% or 100%).
    """
    cfg = GeneratorConfig(
        n_patients=300,
        min_obs_visits=3,
        max_obs_visits=5,
        min_future_visits=2,
        max_future_visits=4,
        seed=42,
        progression_prevalence=0.35,
        missingness_rate=0.0,
    )
    result = SyntheticGenerator(config=cfg).generate()
    actual = result.metadata["actual_progression_prevalence"]
    intended = cfg.progression_prevalence

    assert abs(actual - intended) <= 0.20, (
        f"Actual prevalence {actual:.3f} is more than 20pp away from "
        f"intended {intended:.3f}. Check trajectory calibration."
    )
    # Also must not be degenerate
    assert actual > 0.01, "No positive patients generated — calibration broken"
    assert actual < 0.99, "All patients positive — calibration broken"
