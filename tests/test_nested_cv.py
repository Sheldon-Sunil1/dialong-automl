"""Phase 13E: unified evaluation runner (repeated nested patient-level CV).

Small, fast settings. Checks coverage (each patient once per repeat), measured
single test access, test-fold isolation, label-invariance of every selection,
the inner-CV objective, a label-shuffle null, statistics helpers, provenance
and never-overwrite outputs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dialong_automl.config.schema import AppConfig, NestedCVConfig
from dialong_automl.evaluation.result_schema import RunResult
from dialong_automl.experiments.nested_cv import (
    CALIBRATED_PREDICTOR,
    MODEL_PREDICTOR,
    ORACLE_PREDICTOR,
    HeldOutFold,
    NestedCVRunner,
    _subset,
)
from dialong_automl.experiments.statistics import (
    bootstrap_p_value,
    bootstrap_summary,
    holm_adjust,
)

BASELINES = ["b0_prevalence", "b1_last_hba1c", "b2_trend_extrapolation",
             "b3_logreg_all", "b3_logreg_hba1c_only", "b3_logreg_no_hba1c"]


def _cfg(tmp_path, scenario="v2_standard", n=500, repeats=2, trials=2) -> AppConfig:
    cfg = AppConfig()
    cfg.paths.outputs_dir = str(tmp_path / "outputs")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.cohort.require_outcome_measurement = scenario == "v2_standard"
    cfg.training.threshold_strategy = "validation_f1"
    cfg.model.enabled_models = ["logistic_regression", "lightgbm"]
    cfg.nested_cv = NestedCVConfig(scenario=scenario, n_patients=n, dataset_seed=5, oracle_draws=6,
                                   repeats=repeats, outer_folds=3, inner_folds=2, n_trials=trials,
                                   bootstrap_resamples=60, bootstrap_seed=1)
    return cfg


@pytest.fixture(scope="module")
def v2_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ncv")
    runner = NestedCVRunner(_cfg(tmp))
    gen = runner.generate()
    cohort = runner.build_cohort(gen.full_df)
    res = runner.run_on_cohort(cohort.observation_data, cohort.patient_targets,
                               generation_metadata=gen.metadata, truth=gen.truth_patients,
                               cohort_audit=cohort.audit.to_dict(), run_name="fixture")
    return runner, cohort, gen, res


# ---------------------------------------------------------------------------
# Coverage, test access, isolation
# ---------------------------------------------------------------------------

def test_every_patient_scored_exactly_once_per_repeat_by_every_predictor(v2_run):
    _, cohort, _, res = v2_run
    ids = set(cohort.patient_targets.index)
    pred = res.predictions
    assert set(pred["predictor"]) == {MODEL_PREDICTOR, CALIBRATED_PREDICTOR, ORACLE_PREDICTOR,
                                      *BASELINES}
    for (rep, name), grp in pred.groupby(["repeat", "predictor"]):
        assert set(grp["patient_id"]) == ids and not grp["patient_id"].duplicated().any()
        labels = cohort.patient_targets.loc[grp["patient_id"]].to_numpy()
        np.testing.assert_array_equal(grp["y"].to_numpy(), labels)


def test_each_outer_test_fold_is_read_exactly_once(v2_run):
    *_, res = v2_run
    assert len(res.folds) == 2 * 3
    assert all(f["test_access_count"] == 1 for f in res.folds)
    assert set(res.run_result.metadata["test_access_counts"].values()) == {1}


def test_outer_folds_partition_the_cohort_within_each_repeat(v2_run):
    runner, cohort, _, _ = v2_run
    ids = sorted(cohort.patient_targets.index)
    for r in range(2):
        seen = []
        for tr, te in runner.outer_splits(ids, cohort.patient_targets, r):
            assert not set(tr) & set(te)
            seen.extend(te)
        assert sorted(seen) == ids


def test_run_fold_refuses_test_patients_in_its_training_inputs(v2_run):
    runner, cohort, _, _ = v2_run
    obs, y = cohort.observation_data, cohort.patient_targets
    tr, te = next(runner.outer_splits(sorted(y.index), y, 0))
    with pytest.raises(RuntimeError, match="outer-training"):
        runner.run_fold(0, 0, obs, y, tr, HeldOutFold(obs, y, te))   # full cohort passed


def test_flipping_test_labels_changes_no_selection(v2_run):
    runner, cohort, _, _ = v2_run
    obs, y = cohort.observation_data, cohort.patient_targets.astype(int)
    tr, te = next(runner.outer_splits(sorted(y.index), y, 0))
    tr_obs, tr_y = _subset(obs, y, tr)
    flipped = y.copy()
    flipped.loc[te] = 1 - flipped.loc[te]
    a, rows_a = runner.run_fold(0, 0, tr_obs, tr_y, tr, HeldOutFold(obs, y, te))
    b, rows_b = runner.run_fold(0, 0, tr_obs, tr_y, tr, HeldOutFold(obs, flipped, te))
    for key in ("selected_model", "hyperparameters", "inner_cv_auroc_best", "threshold",
                "threshold_selection", "baseline_thresholds", "n_train_positive"):
        assert a[key] == b[key], key
    sa = pd.DataFrame(rows_a).sort_values(["predictor", "patient_id"])
    sb = pd.DataFrame(rows_b).sort_values(["predictor", "patient_id"])
    np.testing.assert_allclose(sa["score"].to_numpy(), sb["score"].to_numpy())   # same predictions
    assert (sa["y"].to_numpy() != sb["y"].to_numpy()).all()                      # only labels moved


def test_inner_cv_objective_is_mean_of_inner_fold_aurocs(v2_run):
    runner, cohort, _, _ = v2_run
    from dialong_automl.experiments.nested_cv import CVAutoMLOptimizer

    obs, y = cohort.observation_data, cohort.patient_targets.astype(int)
    tr, _ = next(runner.outer_splits(sorted(y.index), y, 0))
    tr_obs, tr_y = _subset(obs, y, tr)
    pairs = [(f, e) for f, e, _ in runner._inner_folds(tr_obs, tr_y, tr, 3)]
    opt = CVAutoMLOptimizer(n_trials=2, seed=1, enabled_models=["logistic_regression"])
    study = opt.optimize_cv(pairs, sorted(tr))
    for t in study.trials:
        assert t.value == pytest.approx(np.mean(t.user_attrs["inner_fold_aurocs"]))
        assert len(t.user_attrs["inner_fold_aurocs"]) == 2
    best = opt.best_trial_result()
    assert len(best.val_proba) == len(tr) and not np.isnan(best.val_proba).any()
    np.testing.assert_array_equal(np.sort(opt.oof_labels), np.sort(tr_y.to_numpy()))


# ---------------------------------------------------------------------------
# Results, oracle, provenance, outputs
# ---------------------------------------------------------------------------

def test_summary_contains_cis_and_sensible_reference_values(v2_run):
    *_, res = v2_run
    rows = {r["predictor"]: r for r in res.summary}
    for name, r in rows.items():
        assert r["auroc_ci_low"] <= r["auroc"] <= r["auroc_ci_high"], name
        assert r["auroc_fold_mean_ci_low"] <= r["auroc_fold_mean"] <= r["auroc_fold_mean_ci_high"]
    # A monotone per-fold calibrator cannot change within-fold ranking.
    assert rows[CALIBRATED_PREDICTOR]["auroc_fold_mean"] == pytest.approx(
        rows[MODEL_PREDICTOR]["auroc_fold_mean"], abs=1e-12)
    assert rows["b0_prevalence"]["auroc"] == pytest.approx(0.5, abs=0.06)
    assert rows["b1_last_hba1c"]["brier"] is None
    assert rows[ORACLE_PREDICTOR]["auroc"] > rows["b0_prevalence"]["auroc"]
    comps = [c for c in res.comparisons if c["metric"] == "auroc"]
    assert {c["comparator"] for c in comps} == {*BASELINES, ORACLE_PREDICTOR, CALIBRATED_PREDICTOR}
    cal = next(c for c in comps if c["comparator"] == CALIBRATED_PREDICTOR)
    assert cal["role"] == "calibrated_variant" and cal["p_value_holm"] is None
    assert sum(c["primary"] for c in comps) == 1
    assert next(c for c in comps if c["primary"])["comparator"] == "b3_logreg_all"
    oracle = next(c for c in comps if c["comparator"] == ORACLE_PREDICTOR)
    assert oracle["role"] == "ceiling" and oracle["p_value_holm"] is None   # not in Holm family
    holm_family = [c for c in comps if c["role"] == "baseline" and not c["primary"]]
    assert all(c["p_value_holm"] is not None for c in holm_family) and len(holm_family) == 5


def test_run_result_has_full_provenance_and_files(v2_run):
    *_, gen, res = v2_run
    d = res.run_dir
    for f in ("run_result.json", "predictions.csv", "summary.csv", "comparisons.csv",
              "folds.json", "resolved_config.json", "generation_metadata.json"):
        assert (d / f).exists(), f
    rr = RunResult.load(d / "run_result.json")
    m = rr.metadata
    for key in ("git_commit", "git_dirty", "python", "packages", "dataset_sha256", "split_hash",
                "generator_params_hash", "test_access_counts", "protocol", "disclaimer"):
        assert key in m, key
    assert m["synthetic_data"] is True
    assert m["dataset_sha256"] == gen.metadata["dataset_sha256"]
    assert m["optuna"]["library"] == "Optuna (third-party)"
    assert rr.metrics[f"{MODEL_PREDICTOR}.auroc"] == pytest.approx(
        next(r["auroc"] for r in res.summary if r["predictor"] == MODEL_PREDICTOR))
    folds = json.loads((d / "folds.json").read_text())
    assert len(folds) == 6


def test_runs_never_overwrite(v2_run):
    runner, cohort, gen, _ = v2_run
    with pytest.raises(FileExistsError):
        runner.run_on_cohort(cohort.observation_data, cohort.patient_targets,
                             generation_metadata=gen.metadata, run_name="fixture")


def test_runner_is_deterministic(tmp_path):
    def summary():
        r = NestedCVRunner(_cfg(tmp_path, n=350, repeats=1, trials=1))
        g = r.generate()
        c = r.build_cohort(g.full_df)
        return r.run_on_cohort(c.observation_data, c.patient_targets, truth=g.truth_patients,
                               write=False).summary
    assert summary() == summary()


def test_label_shuffle_null_gives_chance_auroc(tmp_path):
    runner = NestedCVRunner(_cfg(tmp_path, n=500, repeats=1, trials=1))
    gen = runner.generate()
    cohort = runner.build_cohort(gen.full_df)
    y = cohort.patient_targets.copy()
    y[:] = np.random.default_rng(0).permutation(y.to_numpy())
    res = runner.run_on_cohort(cohort.observation_data, y, write=False)
    row = next(r for r in res.summary if r["predictor"] == MODEL_PREDICTOR)
    assert row["auroc_ci_low"] < 0.5 < row["auroc_ci_high"]


def test_legacy_scenario_runs_without_oracle(tmp_path):
    cfg = _cfg(tmp_path, scenario="legacy_v1", n=400, repeats=1, trials=1)
    res = NestedCVRunner(cfg).run(write=False)
    assert ORACLE_PREDICTOR not in set(res.predictions["predictor"])
    assert res.run_result.metadata["scenario"] == "legacy_v1"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def test_holm_adjustment_hand_example():
    # p sorted: 0.01*3=0.03, 0.02*2=0.04, 0.04*1 -> max(0.04, 0.04)=0.04
    assert holm_adjust([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_bootstrap_p_value_bounds():
    assert bootstrap_p_value(np.zeros(99)) == 1.0
    assert bootstrap_p_value(np.full(99, 0.3)) == pytest.approx(2 / 100)


def test_identical_predictors_have_zero_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random((2, 200))
    t = np.full((2, 200), 0.5)
    summary, comps = bootstrap_summary(
        y, {"a": s, "b": s.copy()}, {"a": t, "b": t}, {"a": True, "b": True},
        n_boot=50, seed=0, reference="a", primary="b",
    )
    auc = next(c for c in comps if c["metric"] == "auroc")
    assert auc["difference"] == 0 and auc["ci_low"] == 0 and auc["ci_high"] == 0
    assert auc["p_value"] == 1.0 and auc["primary"] is True


def test_fold_mean_auroc_removes_between_fold_scale_shift():
    # Perfect ranking inside each fold, but fold 1's scores are shifted by a
    # constant: pooled AUROC drops, the fold-mean AUROC does not.
    rng = np.random.default_rng(1)
    n = 400
    y = np.tile([0, 1], n // 2)
    folds = np.repeat([0, 1], n // 2)[None, :]
    base = y + 0.1 * rng.random(n)
    shifted = (base + np.where(folds[0] == 1, 5.0, 0.0))[None, :]
    t = np.full((1, n), 0.5)
    summary, comps = bootstrap_summary(
        y, {"shifted": shifted, "plain": base[None, :]}, {"shifted": t, "plain": t},
        {"shifted": False, "plain": False}, n_boot=30, seed=0, reference="shifted",
        primary="plain", fold_ids=folds,
    )
    rows = {r["predictor"]: r for r in summary}
    assert rows["shifted"]["auroc"] < 0.8
    assert rows["shifted"]["auroc_fold_mean"] == pytest.approx(1.0)
    assert rows["plain"]["auroc_fold_mean"] == pytest.approx(1.0)
    fm = next(c for c in comps if c["metric"] == "auroc_fold_mean")
    assert fm["difference"] == pytest.approx(0.0)
    # Never the primary comparison, never in the Holm family.
    assert fm["primary"] is False and fm["p_value_holm"] is None
    assert next(c for c in comps if c["metric"] == "auroc")["primary"] is True


def test_fold_mean_metric_absent_without_fold_ids():
    y = np.array([0, 1] * 20)
    s = np.linspace(0, 1, 40)[None, :]
    summary, comps = bootstrap_summary(y, {"a": s}, {"a": s}, {"a": True}, n_boot=5,
                                       seed=0, reference="a")
    assert summary[0]["auroc_fold_mean"] is None
    with pytest.raises(ValueError):
        bootstrap_summary(y, {"a": s}, {"a": s}, {"a": True}, n_boot=5, seed=0,
                          reference="a", fold_ids=np.zeros((2, 40)))


def test_nested_cv_config_validation():
    with pytest.raises(ValueError):
        NestedCVConfig(scenario="v9")
    with pytest.raises(ValueError):
        NestedCVConfig(outer_folds=1)
    with pytest.raises(ValueError):
        NestedCVConfig(bootstrap_resamples=-1)
