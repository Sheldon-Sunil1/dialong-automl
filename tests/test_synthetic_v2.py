"""Phase 13D: synthetic scenarios (legacy_v1 pass-through and v2_standard).

Covers reproducibility, the legacy regression hash, leakage canaries
(observation rows independent of the future process; truth never in the
model-facing table), missingness mechanisms, scheduling, ground truth / oracle,
prevalence control, and the cohort's unascertained-outcome rule.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
from dialong_automl.data.loader import load_longitudinal_csv
from dialong_automl.data.synthetic_generator import FULL_COLUMNS
from dialong_automl.data.synthetic_v2 import (
    SCENARIOS,
    V2Params,
    dataframe_sha256,
    generate_scenario,
)

# SHA-256 of legacy_v1 (n=1000, seed=42). Equals the historical
# data/synthetic/diabetes_progression_full.csv used by research, _v2 and _v3.
LEGACY_SEED42_SHA = "277894ab9f1011b5ab59baca8b54039137d360594fdf83655986254f8bd8deef"
# Frozen v2_standard (n=50, seed=7): changes only if V2Params or the code change.
V2_N50_SEED7_SHA = "7259f41e395c7b17ad0da29908709f2744563d64a9ec0817ebc76351e9bcfc87"


def _cohort(require_measurement: bool = True) -> CohortBuilder:
    return CohortBuilder(CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME, min_observation_visits=3, max_observation_visits=4,
        prediction_horizon_days=365, exclude_known_diabetes_at_baseline=True,
        require_followup_after_cutoff=True, require_outcome_measurement=require_measurement,
    ))


@pytest.fixture(scope="module")
def v2_600():
    return generate_scenario("v2_standard", 600, 11, compute_oracle=True, oracle_draws=16)


# ---------------------------------------------------------------------------
# legacy_v1
# ---------------------------------------------------------------------------

def test_legacy_v1_reproduces_the_historical_dataset_hash():
    res = generate_scenario("legacy_v1", 1000, 42)
    assert dataframe_sha256(res.full_df) == LEGACY_SEED42_SHA
    assert res.truth_patients is None and res.metadata["ground_truth_available"] is False


def test_unknown_scenario_and_bad_arguments_rejected():
    assert SCENARIOS == ("legacy_v1", "v2_standard")
    with pytest.raises(ValueError):
        generate_scenario("v3_magic", 10, 1)
    with pytest.raises(ValueError):
        generate_scenario("v2_standard", 0, 1)
    with pytest.raises(ValueError):
        generate_scenario("legacy_v1", 10, 1, params=V2Params())


# ---------------------------------------------------------------------------
# v2_standard: reproducibility and schema
# ---------------------------------------------------------------------------

def test_v2_is_frozen_and_deterministic():
    a = generate_scenario("v2_standard", 50, 7, compute_oracle=False)
    b = generate_scenario("v2_standard", 50, 7, compute_oracle=False)
    assert dataframe_sha256(a.full_df) == dataframe_sha256(b.full_df) == V2_N50_SEED7_SHA
    c = generate_scenario("v2_standard", 50, 8, compute_oracle=False)
    assert dataframe_sha256(c.full_df) != V2_N50_SEED7_SHA
    assert a.metadata["params_hash"] == V2Params().params_hash()


def test_v2_params_are_immutable_and_validated():
    p = V2Params()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.bio_sd = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        V2Params(missingness_mode="sometimes")
    with pytest.raises(ValueError):
        V2Params(p_hba1c=1.5)
    with pytest.raises(ValueError):
        V2Params(n_obs_min=1)


def test_v2_table_has_exactly_the_legacy_columns(v2_600):
    df = v2_600.full_df
    assert list(df.columns) == FULL_COLUMNS
    assert df["patient_id"].is_monotonic_increasing
    assert set(df["sex"]) <= {"M", "F"}
    assert set(df["smoking_status"]) <= {"never", "former", "current"}
    assert set(df["physical_activity_level"]) <= {"sedentary", "moderate", "active"}
    for col in ("family_history_diabetes", "metformin_use", "hypertension_diagnosis"):
        assert set(df[col]) <= {"yes", "no"}


# ---------------------------------------------------------------------------
# Leakage canaries
# ---------------------------------------------------------------------------

def test_observation_rows_do_not_depend_on_the_future_process():
    base = generate_scenario("v2_standard", 120, 5, compute_oracle=False)
    other = generate_scenario(
        "v2_standard", 120, 5, compute_oracle=False,
        params=V2Params(followup_days=300, dropout_prob=0.5, max_future_visits=2),
    )
    obs_a = base.full_df[~base.full_df["is_future_visit"]].drop(columns=["target"])
    obs_b = other.full_df[~other.full_df["is_future_visit"]].drop(columns=["target"])
    pd.testing.assert_frame_equal(obs_a.reset_index(drop=True), obs_b.reset_index(drop=True))
    assert not base.full_df[base.full_df["is_future_visit"]].equals(
        other.full_df[other.full_df["is_future_visit"]])


def test_truth_never_reaches_the_model_facing_data(v2_600, tmp_path):
    truth_cols = set(v2_600.truth_patients.columns) | set(v2_600.truth_visits.columns)
    truth_only = truth_cols - {"patient_id", "visit_date", "is_future_visit", "metformin_use"}
    assert not truth_only & set(v2_600.full_df.columns)
    paths = v2_600.save(tmp_path / "scenario")
    loaded = load_longitudinal_csv(paths["full_csv"])
    assert not truth_only & set(loaded.columns)
    assert paths["truth_patients"].parent.name == "truth"
    cohort = _cohort().build(loaded)
    assert not truth_only & set(cohort.observation_data.columns)


def test_save_refuses_to_overwrite(v2_600, tmp_path):
    v2_600.save(tmp_path / "s")
    with pytest.raises(FileExistsError):
        v2_600.save(tmp_path / "s")


def test_observation_rows_contain_no_future_visit_dates(v2_600):
    df = v2_600.full_df
    last_obs = df[~df.is_future_visit].groupby("patient_id")["visit_date"].max()
    first_fut = df[df.is_future_visit].groupby("patient_id")["visit_date"].min()
    common = first_fut.index
    assert (first_fut.loc[common] > last_obs.loc[common]).all()


# ---------------------------------------------------------------------------
# Missingness, scheduling, heterogeneity
# ---------------------------------------------------------------------------

def test_missingness_applies_to_future_visits_too(v2_600):
    df = v2_600.full_df
    assert df.loc[~df.is_future_visit, "hba1c"].isna().mean() > 0.1
    assert df.loc[df.is_future_visit, "hba1c"].isna().mean() > 0.05


def test_mcar_rate_matches_parameter():
    res = generate_scenario("v2_standard", 400, 3, compute_oracle=False,
                            params=V2Params(missingness_mode="mcar", p_hba1c=0.7))
    assert res.full_df["hba1c"].isna().mean() == pytest.approx(0.3, abs=0.03)


def test_mnar_measures_high_true_values_more_often():
    res = generate_scenario("v2_standard", 400, 3, compute_oracle=False,
                            params=V2Params(missingness_mode="mnar"))
    tv = res.truth_visits
    hi = tv[tv.true_hba1c >= 6.0]["hba1c_measured"].mean()
    lo = tv[tv.true_hba1c < 5.5]["hba1c_measured"].mean()
    assert hi > lo + 0.2


def test_visit_gaps_are_irregular_and_shorter_after_high_readings(v2_600):
    df = v2_600.full_df.sort_values(["patient_id", "visit_date"])
    gap = df.groupby("patient_id")["visit_date"].diff().dt.days
    prev = df.groupby("patient_id")["hba1c"].shift().groupby(df["patient_id"]).ffill()
    ok = gap.notna()
    assert gap[ok].std() > 40
    assert gap[ok & (prev >= 6.0)].median() < gap[ok & (prev < 6.0)].median()


def test_variable_length_histories_reach_the_cohort(v2_600):
    cohort = _cohort().build(v2_600.full_df)
    counts = cohort.observation_data.groupby("patient_id").size()
    assert set(counts.unique()) == {3, 4}


def test_subgroups_overlap_in_rate(v2_600):
    tp = v2_600.truth_patients
    assert tp["subgroup"].nunique() == 4
    stable = tp[tp.subgroup == "stable"]["hba1c_rate_per_year"]
    slow = tp[tp.subgroup == "slow"]["hba1c_rate_per_year"]
    assert stable.max() > slow.min()


# ---------------------------------------------------------------------------
# Outcome, cohort and oracle
# ---------------------------------------------------------------------------

def test_cohort_labels_match_the_generator_rule(v2_600):
    cohort = _cohort().build(v2_600.full_df)
    tp = v2_600.truth_patients.set_index("patient_id").loc[cohort.patient_targets.index]
    assert (tp["label_generator_rule"].astype(int) == cohort.patient_targets).all()
    assert tp["outcome_ascertained"].all()
    assert not tp["diagnosed_by_index"].any()


def test_every_exclusion_rule_is_exercised(v2_600):
    a = _cohort().build(v2_600.full_df).audit
    assert a.excluded_prior_diabetes > 0
    assert a.excluded_insufficient_visits > 0
    assert a.excluded_insufficient_followup > 0
    assert a.excluded_unmeasured_outcome > 0
    assert a.to_dict()["require_outcome_measurement"] is True


def test_oracle_is_a_valid_probability_and_beats_last_hba1c(v2_600):
    from sklearn.metrics import roc_auc_score

    cohort = _cohort().build(v2_600.full_df)
    y = cohort.patient_targets
    tp = v2_600.truth_patients.set_index("patient_id").loc[y.index]
    assert tp["oracle_prob"].between(0, 1).all()
    last = (cohort.observation_data.groupby("patient_id")["hba1c"]
            .apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan).loc[y.index])
    auc_last = roc_auc_score(y, last.fillna(last.median()))
    auc_oracle = roc_auc_score(y, tp["oracle_prob"])
    assert auc_oracle > auc_last


@pytest.mark.parametrize("seed", [21, 22])
def test_eligible_prevalence_is_controlled(seed):
    res = generate_scenario("v2_standard", 1200, seed, compute_oracle=False)
    y = _cohort().build(res.full_df).patient_targets
    assert 0.06 <= y.mean() <= 0.18
    assert y.sum() >= 40


def test_unascertained_outcome_rule_on_a_hand_built_patient():
    rows = []
    for pid, future_a1c, dx in (("P1", np.nan, 0), ("P2", 5.8, 0), ("P3", np.nan, 1)):
        for d, a1c, fut in (("2020-01-01", 5.5, False), ("2020-04-01", 5.6, False),
                            ("2020-07-01", 5.6, False), ("2020-10-01", future_a1c, True)):
            rows.append({"patient_id": pid, "visit_date": pd.Timestamp(d), "hba1c": a1c,
                         "is_future_visit": fut, "diabetes_diagnosis": dx if fut else 0})
    df = pd.DataFrame(rows)
    legacy = _cohort(require_measurement=False).build(df)
    # Old rule: an unmeasured outcome counts as no onset (label 0).
    assert legacy.patient_targets.to_dict() == {"P1": 0, "P2": 0, "P3": 1}
    strict = _cohort(require_measurement=True).build(df)
    assert strict.patient_targets.to_dict() == {"P2": 0, "P3": 1}            # P1 excluded
    assert strict.audit.excluded_unmeasured_outcome == 1
