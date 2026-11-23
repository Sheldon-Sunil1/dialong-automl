"""Phase 14B: post-hoc calibration (R1) and its integration.

Unit tests use independent references (scikit-learn) or hand-computed cases.
Integration tests check that the calibrator is fitted only on permitted data
(inner-CV OOF / validation split) and never on the evaluated patients.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dialong_automl.evaluation.metrics import auroc
from dialong_automl.reliability import (
    IsotonicCalibrator,
    PlattCalibrator,
    calibration_slope_intercept,
    calibration_summary,
    calibrator_from_dict,
    ece_equal_mass,
    make_calibrator,
)

sk_lm = pytest.importorskip("sklearn.linear_model")
sk_iso = pytest.importorskip("sklearn.isotonic")


def _sim(n=20000, shift=0.0, scale=1.0, seed=0):
    """True risk p_true; the 'model' reports sigmoid(scale*logit(p_true) + shift)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(-2.2, 1.2, n)
    p_true = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p_true).astype(int)
    p_model = 1 / (1 + np.exp(-(scale * z + shift)))
    return y, p_model, p_true


# ---------------------------------------------------------------------------
# Platt
# ---------------------------------------------------------------------------

def test_platt_matches_unpenalised_sklearn_logistic_regression():
    y, p, _ = _sim(n=3000, shift=0.8, scale=0.7, seed=1)
    cal = PlattCalibrator().fit(p, y)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    ref = sk_lm.LogisticRegression(C=1e12, max_iter=10000, tol=1e-12).fit(z, y)
    assert cal.slope == pytest.approx(ref.coef_[0][0], abs=1e-4)
    assert cal.intercept == pytest.approx(ref.intercept_[0], abs=1e-4)


def test_platt_recovers_identity_for_already_calibrated_scores():
    y, p, _ = _sim(seed=2)
    cal = PlattCalibrator().fit(p, y)
    assert cal.slope == pytest.approx(1.0, abs=0.06)
    assert cal.intercept == pytest.approx(0.0, abs=0.12)


def test_platt_corrects_systematic_over_prediction():
    """The 14A finding: mean predicted 17 % vs 10 % observed. Calibration must fix it."""
    y_fit, p_fit, _ = _sim(seed=3, shift=0.9)
    y_new, p_new, _ = _sim(seed=4, shift=0.9)          # independent evaluation data
    cal = PlattCalibrator().fit(p_fit, y_fit)
    raw, fixed = calibration_summary(y_new, p_new), calibration_summary(y_new, cal.transform(p_new))
    assert raw["mean_predicted"] > raw["observed_rate"] * 1.4
    assert fixed["mean_predicted"] == pytest.approx(fixed["observed_rate"], abs=0.01)
    assert fixed["ece"] < raw["ece"] / 3
    assert abs(fixed["calibration_in_the_large"]) < 0.1 < abs(raw["calibration_in_the_large"])


def test_platt_is_strictly_monotone_and_preserves_auroc_exactly():
    y, p, _ = _sim(n=2000, shift=1.0, scale=0.6, seed=5)
    cal = PlattCalibrator().fit(p, y)
    q = cal.transform(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= 0)
    assert auroc(y, q) == auroc(y, p)


def test_platt_keeps_distinct_extreme_probabilities_distinct():
    # Confident models emit values like 1e-12; a coarse logit clip tied them.
    p = np.array([1e-13, 1e-11, 1e-9, 1e-7, 0.2, 0.5, 0.8, 1 - 1e-9])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    q = PlattCalibrator().fit(p, y).transform(p)
    assert np.all(np.diff(q) > 0)
    assert np.all(np.isfinite(q))


def test_platt_edge_cases():
    with pytest.raises(ValueError, match="both classes"):
        PlattCalibrator().fit(np.array([0.1, 0.2]), np.array([0, 0]))
    const = PlattCalibrator().fit(np.full(10, 0.3), np.array([1] * 2 + [0] * 8))
    # Constant fit input: intercept-only shift, so the fit value maps to the
    # prevalence but later inputs keep their order (no collapse to one value).
    np.testing.assert_allclose(const.transform(np.array([0.3])), [0.2], atol=1e-6)
    lo, hi = const.transform(np.array([0.1, 0.9]))
    assert lo < 0.2 < hi and const.slope == 1.0
    assert "prevalence" in const.fallback_reason
    inverted = PlattCalibrator().fit(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1]))
    assert inverted.slope == 1.0 and inverted.intercept == 0.0 and inverted.fallback_reason
    with pytest.raises(RuntimeError):
        PlattCalibrator().transform(np.array([0.5]))
    with pytest.raises(ValueError):
        PlattCalibrator().fit(np.array([1.2]), np.array([1]))


# ---------------------------------------------------------------------------
# Isotonic / identity / serialisation
# ---------------------------------------------------------------------------

def test_isotonic_matches_sklearn_on_the_fit_data_and_is_monotone():
    y, p, _ = _sim(n=1500, shift=0.7, seed=6)
    cal = IsotonicCalibrator().fit(p, y)
    ref = sk_iso.IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
    np.testing.assert_allclose(cal.transform(p), ref.predict(p), atol=1e-12)
    grid = np.linspace(0, 1, 501)
    assert np.all(np.diff(cal.transform(grid)) >= 0)


def test_isotonic_hand_example():
    # p sorted 0.1..0.4 with y 0,1,0,1 -> PAV pools (0.2,0.3) -> [0, .5, .5, 1]
    cal = IsotonicCalibrator().fit(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 0, 1]))
    np.testing.assert_allclose(cal.transform(np.array([0.1, 0.25, 0.3, 0.4, 0.05, 0.99])),
                               [0, 0.5, 0.5, 1, 0, 1])


@pytest.mark.parametrize("method", ["platt", "isotonic", "none"])
def test_round_trip_through_json(method):
    y, p, _ = _sim(n=800, shift=0.5, seed=7)
    cal = make_calibrator(method).fit(p, y)
    d = json.loads(json.dumps(cal.to_dict()))
    back = calibrator_from_dict(d)
    np.testing.assert_allclose(back.transform(p), cal.transform(p), atol=1e-12)
    assert d["method"] == method and d["n_fit"] == 800 and d["n_positive_fit"] == int(y.sum())


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        make_calibrator("beta")
    from dialong_automl.config.schema import ReliabilityConfig

    with pytest.raises(ValueError):
        ReliabilityConfig(calibration_method="temperature")
    assert ReliabilityConfig().enabled is True and ReliabilityConfig().calibration_method == "platt"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_calibration_slope_and_intercept_on_calibrated_data():
    y, p, _ = _sim(seed=8)
    slope, citl = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.06) and citl == pytest.approx(0.0, abs=0.06)
    y2, p2, _ = _sim(seed=8, scale=2.0)                # over-confident -> slope < 1
    assert calibration_slope_intercept(y2, p2)[0] < 0.7
    assert calibration_slope_intercept(np.zeros(5), np.full(5, 0.2)) == (None, None)


def test_ece_equal_mass_hand_example():
    # 4 patients, 2 bins of 2: bin1 p=(.1,.2) y=(0,0): |0-.15|; bin2 p=(.7,.9) y=(1,0): |.5-.8|
    y = np.array([0, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.7, 0.9])
    assert ece_equal_mass(y, p, n_bins=2) == pytest.approx(0.5 * 0.15 + 0.5 * 0.3)


# ---------------------------------------------------------------------------
# Single-split pipeline integration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    from dialong_automl.training import pipeline as pipe_mod
    from dialong_automl.training.pipeline import TrainingPipeline
    from tests.test_training_pipeline import _tiny_cfg, _tiny_csv

    tmp = tmp_path_factory.mktemp("cal")
    csv = _tiny_csv(tmp, n_patients=150)
    cfg = _tiny_cfg(tmp)
    fits: list[int] = []
    real = pipe_mod.make_calibrator

    def spy(method):
        cal = real(method)
        real_fit = cal.fit

        def fit(p, y):
            fits.append(len(y))
            return real_fit(p, y)
        cal.fit = fit
        return cal

    pipe_mod.make_calibrator = spy
    try:
        result = TrainingPipeline(cfg, full_csv_path=csv).run()
    finally:
        pipe_mod.make_calibrator = real
    return result, fits


def test_pipeline_fits_calibrator_once_on_the_validation_split(pipeline_run):
    result, fits = pipeline_run
    assert fits == [result.n_val]
    rel = result.reliability
    assert rel["enabled"] is True
    assert rel["calibration"]["method"] == "platt"
    assert rel["calibration"]["n_fit"] == result.n_val
    assert rel["calibration"]["optimistic"] is True
    assert "validation" in rel["calibration"]["fit_partition"]
    for key in ("test_calibration_raw", "test_calibration_calibrated"):
        assert rel[key]["n"] == result.n_test


def test_pipeline_writes_calibration_artifact_and_metadata(pipeline_run):
    result, _ = pipeline_run
    path = Path(result.artifact_paths["calibration"])
    saved = json.loads(path.read_text())
    meta = json.loads(Path(result.artifact_paths["model_metadata"]).read_text())
    summary = json.loads(Path(result.artifact_paths["run_summary"]).read_text())
    assert meta["reliability"] == summary["reliability"] == result.reliability
    assert saved == meta["reliability"]["calibration"]
    cal = calibrator_from_dict(saved)                                 # loadable for 14E
    t = meta["threshold"]
    assert cal.transform(np.array([t]))[0] == pytest.approx(saved["calibrated_threshold"])


def test_pipeline_prediction_rule_unchanged_and_disable_switch(tmp_path):
    from dialong_automl.training.pipeline import TrainingPipeline
    from tests.test_training_pipeline import _tiny_cfg, _tiny_csv

    csv = _tiny_csv(tmp_path, n_patients=150)
    on, off = _tiny_cfg(tmp_path / "on"), _tiny_cfg(tmp_path / "off")
    off.reliability.enabled = False
    r_on = TrainingPipeline(on, full_csv_path=csv).run()
    r_off = TrainingPipeline(off, full_csv_path=csv).run()
    assert r_off.reliability == {"enabled": False}
    assert "calibration" not in r_off.artifact_paths
    assert r_on.test_metrics == r_off.test_metrics          # raw predictions/decisions unchanged
    assert r_on.threshold == r_off.threshold


# ---------------------------------------------------------------------------
# Nested-CV integration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ncv(tmp_path_factory):
    from dialong_automl.experiments.nested_cv import NestedCVRunner
    from tests.test_nested_cv import _cfg

    tmp = tmp_path_factory.mktemp("ncvcal")
    runner = NestedCVRunner(_cfg(tmp, n=500, repeats=1, trials=2))
    gen = runner.generate()
    cohort = runner.build_cohort(gen.full_df)
    res = runner.run_on_cohort(cohort.observation_data, cohort.patient_targets,
                               truth=gen.truth_patients, write=False)
    return runner, cohort, res


def test_calibrated_predictor_keeps_ranking_and_decisions(ncv):
    from dialong_automl.experiments.nested_cv import CALIBRATED_PREDICTOR, MODEL_PREDICTOR

    *_, res = ncv
    pred = res.predictions
    raw = pred[pred.predictor == MODEL_PREDICTOR].set_index(["repeat", "patient_id"]).sort_index()
    cal = (pred[pred.predictor == CALIBRATED_PREDICTOR]
           .set_index(["repeat", "patient_id"]).sort_index())
    # Each fold has its own (monotone) calibrator, so the ranking is identical
    # WITHIN a fold. Pooled across folds it is not: fold models' raw scores sit
    # on different scales, and calibration puts them on a common one.
    for (r, k), grp in raw.groupby(["repeat", "fold"]):
        c = cal.loc[grp.index]
        assert auroc(c["y"].to_numpy(), c["score"].to_numpy()) == pytest.approx(
            auroc(grp["y"].to_numpy(), grp["score"].to_numpy()), abs=1e-12)
    np.testing.assert_array_equal((raw["score"] >= raw["threshold"]).to_numpy(),
                                  (cal["score"] >= cal["threshold"]).to_numpy())
    rows = {r["predictor"]: r for r in res.summary}
    for name in (MODEL_PREDICTOR, CALIBRATED_PREDICTOR, "b3_logreg_all"):
        for m in ("ece", "ece_equal_mass", "cal_slope", "cal_intercept"):
            assert rows[name][m] is not None, (name, m)
    assert rows["b1_last_hba1c"]["ece"] is None                     # raw scores: undefined
    assert any(c["metric"] == "ece" and c["comparator"] == CALIBRATED_PREDICTOR
               for c in res.comparisons)


def test_fold_records_calibrator_fitted_on_inner_oof(ncv):
    runner, cohort, res = ncv
    for f in res.folds:
        c = f["calibration"]
        assert c["method"] == "platt" and c["n_fit"] == f["n_train"]
        assert "out-of-fold" in c["fit_partition"]


def test_flipping_test_labels_does_not_change_the_calibrator(ncv):
    from dialong_automl.experiments.nested_cv import HeldOutFold, _subset

    runner, cohort, _ = ncv
    obs, y = cohort.observation_data, cohort.patient_targets.astype(int)
    tr, te = next(runner.outer_splits(sorted(y.index), y, 0))
    tr_obs, tr_y = _subset(obs, y, tr)
    flipped = y.copy()
    flipped.loc[te] = 1 - flipped.loc[te]
    a, _ = runner.run_fold(0, 0, tr_obs, tr_y, tr, HeldOutFold(obs, y, te))
    b, _ = runner.run_fold(0, 0, tr_obs, tr_y, tr, HeldOutFold(obs, flipped, te))
    assert a["calibration"] == b["calibration"]


def test_nested_runner_respects_disabled_reliability(tmp_path):
    from dialong_automl.experiments.nested_cv import CALIBRATED_PREDICTOR, NestedCVRunner
    from tests.test_nested_cv import _cfg

    cfg = _cfg(tmp_path, n=350, repeats=1, trials=1)
    cfg.reliability.enabled = False
    res = NestedCVRunner(cfg).run(write=False)
    assert CALIBRATED_PREDICTOR not in set(res.predictions["predictor"])
    assert all(f["calibration"] is None for f in res.folds)
