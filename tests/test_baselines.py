"""Phase 13C step 2: fixed reference baselines B0-B3 and the BaselineRunner.

Unit tests use hand-computed values; leakage tests check that nothing a
baseline learns, nor any threshold, depends on data it must not see.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from dialong_automl.baselines import (
    AggregateLogisticBaseline,
    BaselineInputs,
    BaselineRunner,
    LastHbA1cBaseline,
    PrevalenceBaseline,
    TrendExtrapolationBaseline,
    b3_columns,
    default_baselines,
    hba1c_slope_per_day,
    select_raw_score_threshold,
)
from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
from dialong_automl.data.splitter import PatientSplitter, SplitConfig
from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator
from dialong_automl.features.wide import WideRepresentationBuilder

BASELINE_NAMES = [
    "b0_prevalence", "b1_last_hba1c", "b2_trend_extrapolation",
    "b3_logreg_all", "b3_logreg_hba1c_only", "b3_logreg_no_hba1c",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _inputs(obs: pd.DataFrame, targets: pd.Series | None = None) -> BaselineInputs:
    wide = WideRepresentationBuilder(max_waves=4).build(obs, targets)
    y = None if targets is None else targets.loc[wide.patient_ids].to_numpy()
    return BaselineInputs(wide_X=wide.X, patient_ids=wide.patient_ids, obs_df=obs, y=y)


def _hand_obs() -> pd.DataFrame:
    """A: 3 readings 100 days apart rising 0.2 (slope 0.002/day); B: 1 reading; C: no HbA1c."""
    rows = [
        ("A", "2020-01-01", 5.0), ("A", "2020-04-10", 5.2), ("A", "2020-07-19", 5.4),
        ("B", "2020-01-01", 6.0), ("B", "2020-03-01", np.nan),
        ("C", "2020-01-01", np.nan), ("C", "2020-02-01", np.nan),
    ]
    df = pd.DataFrame(rows, columns=["patient_id", "visit_date", "hba1c"])
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    return df


@pytest.fixture(scope="module")
def partitions():
    full = SyntheticGenerator(config=GeneratorConfig(n_patients=300, seed=3)).generate().full_df
    cohort = CohortBuilder(CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME, exclude_known_diabetes_at_baseline=False,
    )).build(full)
    split = PatientSplitter(SplitConfig(seed=3, stratify=True)).split(
        cohort.observation_data, cohort.patient_targets
    )
    t = cohort.patient_targets
    tv_df = pd.concat([split.train_df, split.validation_df], ignore_index=True)
    tv_t = pd.concat([split.train_targets, split.validation_targets])
    return {
        "train": _inputs(split.train_df, t.loc[split.train_ids]),
        "validation": _inputs(split.validation_df, t.loc[split.validation_ids]),
        "train_val": _inputs(tv_df, tv_t),
        "test": _inputs(split.test_df, t.loc[split.test_ids]),
    }


def _runner(strategy: str = "fixed") -> BaselineRunner:
    return BaselineRunner(
        default_baselines(horizon_days=365, C=1.0, max_iter=2000, seed=42),
        threshold_strategy=strategy, fixed_threshold=0.5, raw_score_fallback_threshold=6.5,
    )


def _run(p, strategy="fixed", test=None) -> BaselineRunner:
    r = _runner(strategy)
    r.fit_and_validate(p["train"], p["validation"])
    r.refit(p["train_val"])
    r.evaluate_test(test or p["test"])
    return r


# ---------------------------------------------------------------------------
# B0 / B1 / B2
# ---------------------------------------------------------------------------

def test_b0_predicts_fit_prevalence_for_everyone():
    data = BaselineInputs(pd.DataFrame(index=range(5)), list("abcde"), pd.DataFrame(),
                          y=np.array([1, 0, 0, 0, 1]))
    b = PrevalenceBaseline().fit(data)
    np.testing.assert_array_equal(b.predict_scores(data), np.full(5, 0.4))
    assert b.params() == {"prevalence": 0.4}


def test_b1_uses_last_observed_hba1c_and_fit_median_for_missing():
    obs = _hand_obs()
    data = _inputs(obs, pd.Series({"A": 0, "B": 1, "C": 0}))
    b = LastHbA1cBaseline().fit(data)
    # hba1c_latest: A=5.4, B=6.0 (last non-missing), C=NaN -> median of {5.4, 6.0}
    assert b.fill_value_ == pytest.approx(5.7)
    np.testing.assert_allclose(b.predict_scores(data), [5.4, 6.0, 5.7])


def test_b1_fill_value_comes_only_from_fit_data():
    obs = _hand_obs()
    fit_data = _inputs(obs[obs.patient_id != "C"], pd.Series({"A": 0, "B": 1}))
    b = LastHbA1cBaseline().fit(fit_data)
    other = _hand_obs()
    other.loc[other.patient_id == "A", "hba1c"] = 9.9   # scoring data changes ...
    scores = b.predict_scores(_inputs(other))
    assert b.fill_value_ == pytest.approx(5.7)           # ... the fitted value does not
    np.testing.assert_allclose(scores, [9.9, 6.0, 5.7])


def test_slope_per_day_hand_computed_and_ordered_by_patient_ids():
    obs = _hand_obs()
    slopes = hba1c_slope_per_day(obs, ["C", "B", "A", "Z"])
    np.testing.assert_allclose(slopes, [0.0, 0.0, 0.002, 0.0], atol=1e-12)


def test_slope_is_zero_for_same_day_readings():
    obs = pd.DataFrame({"patient_id": ["A", "A"], "visit_date": pd.to_datetime(["2020-01-01"] * 2),
                        "hba1c": [5.0, 6.0]})
    assert hba1c_slope_per_day(obs, ["A"])[0] == 0.0


def test_b2_extrapolates_in_consistent_units():
    obs = _hand_obs()
    data = _inputs(obs, pd.Series({"A": 1, "B": 0, "C": 0}))
    b = TrendExtrapolationBaseline(horizon_days=365).fit(data)
    # A: 5.4 + 0.002 * 365 = 6.13; B: 6.0 + 0; C: fill 5.7 + 0
    np.testing.assert_allclose(b.predict_scores(data), [6.13, 6.0, 5.7], atol=1e-9)
    assert b.params()["horizon_days"] == 365


def test_b2_rejects_non_positive_horizon():
    with pytest.raises(ValueError):
        TrendExtrapolationBaseline(horizon_days=0)


def test_unfitted_baseline_refuses_to_score():
    with pytest.raises(RuntimeError):
        LastHbA1cBaseline().predict_scores(_inputs(_hand_obs()))


# ---------------------------------------------------------------------------
# B3
# ---------------------------------------------------------------------------

def test_b3_column_sets_are_pre_specified():
    num_all, cat_all = b3_columns("all")
    num_h, cat_h = b3_columns("hba1c_only")
    num_n, cat_n = b3_columns("no_hba1c")
    assert num_h == ["hba1c_latest", "hba1c_mean", "hba1c_min", "hba1c_max", "hba1c_slope"]
    assert cat_h == []
    assert set(num_h) <= set(num_all)
    assert not any(c.startswith("hba1c") for c in num_n)
    assert set(num_n) == set(num_all) - set(num_h)
    assert cat_all == cat_n and all(c.endswith("_latest") for c in cat_all)
    every = num_all + cat_all
    assert not any(re.search(r"_w\d+$", c) for c in every)          # no per-wave columns
    assert not any(c.endswith(("_std", "_delta")) for c in every)
    assert "n_obs_visits" not in every
    assert "age_at_visit_latest" in num_all and "age_at_visit_slope" not in num_all
    with pytest.raises(ValueError):
        b3_columns("everything")


def test_b3_columns_exist_in_the_wide_representation(partitions):
    cols = set(partitions["train"].wide_X.columns)
    for v in ("all", "hba1c_only", "no_hba1c"):
        num, cat = b3_columns(v)
        assert set(num + cat) <= cols


def test_b3_preprocessing_is_fitted_on_fit_data_only(partitions):
    p = partitions
    b = AggregateLogisticBaseline("all").fit(p["train"])
    medians = dict(b._prep._num_medians)
    shifted = p["validation"].wide_X.copy()
    shifted["hba1c_latest"] = shifted["hba1c_latest"] + 100.0
    b.predict_scores(dataclasses.replace(p["validation"], wide_X=shifted))
    assert b._prep._num_medians == medians
    ref = AggregateLogisticBaseline("all").fit(p["train"])
    np.testing.assert_array_equal(b.predict_scores(p["validation"]),
                                  ref.predict_scores(p["validation"]))   # deterministic


def test_b3_needs_both_classes(partitions):
    p = partitions["train"]
    one_class = dataclasses.replace(p, y=np.zeros(len(p.patient_ids), dtype=np.int32))
    with pytest.raises(ValueError, match="both classes"):
        AggregateLogisticBaseline("hba1c_only").fit(one_class)


def test_b3_variants_produce_probabilities(partitions):
    for v in ("all", "hba1c_only", "no_hba1c"):
        s = AggregateLogisticBaseline(v).fit(partitions["train"]).predict_scores(partitions["test"])
        assert s.shape == (len(partitions["test"].patient_ids),)
        assert np.all((s >= 0) & (s <= 1))


# ---------------------------------------------------------------------------
# Raw-score threshold
# ---------------------------------------------------------------------------

def test_raw_score_threshold_hand_computed():
    y = np.array([0, 0, 1, 1, 0])
    s = np.array([5.0, 5.5, 6.0, 6.2, 6.1])
    info = select_raw_score_threshold(y, s, fallback_threshold=6.5)
    # candidates: 5.0, 5.25, 5.75, 6.05, 6.15 ; best F1 at 5.75 (P=2/3, R=1) -> 0.8
    assert info["threshold"] == pytest.approx(5.75)
    assert info["fallback_reason"] is None and info["test_set_used"] is False
    assert info["score_scale"] == "HbA1c %"


def test_raw_score_threshold_falls_back_without_positives():
    info = select_raw_score_threshold(np.zeros(4), np.array([5.0, 5.1, 5.2, 5.3]), 6.5)
    assert info["threshold"] == 6.5 and info["fallback_reason"]


# ---------------------------------------------------------------------------
# Runner: policy, ordering, leakage
# ---------------------------------------------------------------------------

def test_runner_reports_all_baselines_with_three_metric_sets(partitions):
    res = _run(partitions).results()
    assert list(res) == BASELINE_NAMES
    n_val, n_test = len(partitions["validation"].patient_ids), len(partitions["test"].patient_ids)
    for name, r in res.items():
        assert r["validation_metrics"]["val_n_samples"] == n_val
        assert r["retrain_resubstitution_metrics"]["resub_n_samples"] == n_val
        assert r["test_metrics"]["test_n_samples"] == n_test
        assert r["validation_metrics"]["val_threshold"] == r["threshold"]
        assert r["test_metrics"]["test_threshold"] == r["threshold"]          # frozen
        assert r["threshold_selection"]["test_set_used"] is False
        assert "params_train_fit" in r and "params_final_fit" in r


def test_b0_auroc_is_exactly_half_and_auprc_is_prevalence(partitions):
    r = _run(partitions).results()["b0_prevalence"]
    y_test = partitions["test"].y
    assert r["validation_metrics"]["val_auroc"] == 0.5
    assert r["test_metrics"]["test_auroc"] == 0.5
    assert r["test_metrics"]["test_auprc"] == pytest.approx(y_test.mean())


def test_raw_score_baselines_have_no_brier(partitions):
    res = _run(partitions).results()
    for name in ("b1_last_hba1c", "b2_trend_extrapolation"):
        r = res[name]
        assert r["is_probabilistic"] is False
        assert r["validation_metrics"]["val_brier"] is None
        assert r["retrain_resubstitution_metrics"]["resub_brier"] is None
        assert r["test_metrics"]["test_brier"] is None
        assert r["threshold_selection"]["strategy"] == "validation_f1_raw_score"
    assert res["b0_prevalence"]["test_metrics"]["test_brier"] is not None


def test_probability_baselines_follow_the_configured_strategy(partitions):
    fixed = _run(partitions, "fixed").results()
    f1 = _run(partitions, "validation_f1").results()
    for name in ("b0_prevalence", "b3_logreg_all"):
        assert fixed[name]["threshold"] == 0.5
        assert fixed[name]["threshold_selection"]["strategy"] == "fixed"
        assert f1[name]["threshold_selection"]["strategy"] == "validation_f1"
        assert "validation" in f1[name]["threshold_selection"]["selected_on"]


def test_refit_uses_train_plus_validation(partitions):
    r = _run(partitions).results()["b0_prevalence"]
    p = partitions
    assert r["params_train_fit"]["prevalence"] == pytest.approx(p["train"].y.mean())
    assert r["params_final_fit"]["prevalence"] == pytest.approx(p["train_val"].y.mean())


def test_test_labels_and_features_cannot_change_anything_but_test_metrics(partitions):
    p = partitions
    base = _run(p).results()
    t = p["test"]
    corrupted_X = t.wide_X.copy()
    corrupted_X["hba1c_latest"] = corrupted_X["hba1c_latest"] + 3.0
    flipped = dataclasses.replace(t, y=1 - t.y, wide_X=corrupted_X)
    other = _run(p, test=flipped).results()
    for name in BASELINE_NAMES:
        for key in ("threshold", "threshold_selection", "params_train_fit", "params_final_fit",
                    "validation_metrics", "retrain_resubstitution_metrics"):
            assert base[name][key] == other[name][key], (name, key)
    assert base["b1_last_hba1c"]["test_metrics"] != other["b1_last_hba1c"]["test_metrics"]


def test_runner_enforces_the_stage_order(partitions):
    p = partitions
    r = _runner()
    with pytest.raises(RuntimeError):
        r.refit(p["train_val"])
    with pytest.raises(RuntimeError):
        r.evaluate_test(p["test"])
    r.fit_and_validate(p["train"], p["validation"])
    with pytest.raises(RuntimeError):
        r.evaluate_test(p["test"])                      # not before refit
    with pytest.raises(RuntimeError):
        r.fit_and_validate(p["train"], p["validation"])  # only once
    r.refit(p["train_val"])
    r.evaluate_test(p["test"])
    with pytest.raises(RuntimeError):
        r.evaluate_test(p["test"])                      # test scored exactly once
    assert r.stage == "tested"


def test_runner_rejects_duplicate_names():
    with pytest.raises(ValueError):
        BaselineRunner([PrevalenceBaseline(), PrevalenceBaseline()], "fixed", 0.5, 6.5)


def test_baselines_are_not_registered_models():
    from dialong_automl.models import REGISTRY

    registered = set(REGISTRY.list_all())
    assert not registered & set(BASELINE_NAMES)
    assert not any(n.startswith(("b0_", "b1_", "b2_", "b3_")) for n in registered)


def test_inputs_validate_alignment():
    with pytest.raises(ValueError):
        BaselineInputs(pd.DataFrame(index=range(3)), ["a", "b"], pd.DataFrame())
    with pytest.raises(ValueError):
        BaselineInputs(pd.DataFrame(index=range(2)), ["a", "b"], pd.DataFrame(), y=np.array([1]))
