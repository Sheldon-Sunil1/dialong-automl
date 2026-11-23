"""Phase 14C: conformal sets (R3) and model-variability range (R2). Synthetic data only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from dialong_automl.config.schema import ReliabilityConfig
from dialong_automl.reliability.conformal import MondrianConformal, coverage_summary, set_labels
from dialong_automl.reliability.ensemble import member_subsamples, variability_range


def _sim(n, seed, prevalence=0.1):
    rng = np.random.default_rng(seed)
    z = rng.normal(np.log(prevalence / (1 - prevalence)), 1.5, n)
    p = 1 / (1 + np.exp(-z))
    return (rng.random(n) < p).astype(int), p


def test_hand_example_quantiles():
    # Class 1 scores 1-p: p=(.9,.8,.6,.3) -> (.1,.2,.4,.7); alpha=.25 -> k=ceil(5*.75)=4 -> .7
    # Class 0 scores p:   (.05,.1,.2,.5)            -> k=4 -> .5
    p = np.array([0.9, 0.8, 0.6, 0.3, 0.05, 0.1, 0.2, 0.5])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    m = MondrianConformal(alpha=0.25).fit(p, y)
    assert m.q[1] == pytest.approx(0.7) and m.q[0] == pytest.approx(0.5)
    has0, has1 = m.predict_sets(np.array([0.1, 0.4, 0.9, 0.25]))
    # p=.1: s0=.1<=.5 in, s1=.9>.7 out; p=.4: both in; p=.9: s0=.9 out, s1=.1 in
    assert set_labels(has0, has1) == ["no_onset", "undecided", "onset", "no_onset"]


def test_too_few_calibration_cases_always_include_the_class():
    m = MondrianConformal(alpha=0.1).fit(np.array([0.2, 0.9, 0.1]), np.array([0, 1, 0]))
    assert math.isinf(m.q[1]) and math.isinf(m.q[0])     # k > n_c for both
    has0, has1 = m.predict_sets(np.array([0.01, 0.99]))
    assert has0.all() and has1.all()
    d = json.loads(json.dumps(m.to_dict()))               # JSON-safe (no inf)
    assert d["q_onset"] is None
    assert math.isinf(MondrianConformal.from_dict(d).q[1])


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_per_class_coverage_on_exchangeable_data(alpha):
    covs = {0: [], 1: []}
    for seed in range(20):
        y, p = _sim(3000, seed)
        cal, test = slice(0, 1500), slice(1500, None)
        m = MondrianConformal(alpha).fit(p[cal], y[cal])
        s = coverage_summary(y[test], *m.predict_sets(p[test]))
        covs[0].append(s["coverage_no_onset"])
        covs[1].append(s["coverage_onset"])
    for c in (0, 1):
        assert np.mean(covs[c]) >= 1 - alpha - 0.01, (c, np.mean(covs[c]))
        assert np.mean(covs[c]) <= 1 - alpha + 0.05, (c, np.mean(covs[c]))


def test_round_trip_and_validation():
    y, p = _sim(500, 3)
    m = MondrianConformal(0.1).fit(p, y)
    m2 = MondrianConformal.from_dict(json.loads(json.dumps(m.to_dict())))
    assert m2.predict_sets(p)[0].tolist() == m.predict_sets(p)[0].tolist()
    assert m2.predict_sets(p)[1].tolist() == m.predict_sets(p)[1].tolist()
    with pytest.raises(ValueError):
        MondrianConformal(0.0)
    with pytest.raises(ValueError):
        MondrianConformal(0.1).fit(np.array([0.1, 0.2]), np.array([0, 0]))
    with pytest.raises(RuntimeError):
        MondrianConformal(0.1).predict_sets(p)


def test_member_subsamples_are_stratified_deterministic_and_distinct():
    ids = [f"p{i:03d}" for i in range(200)]
    y = np.array([1] * 20 + [0] * 180)
    a = member_subsamples(ids, y, 5, 0.8, seed=1)
    assert a == member_subsamples(ids, y, 5, 0.8, seed=1)
    assert a != member_subsamples(ids, y, 5, 0.8, seed=2)
    pos = set(ids[:20])
    for s in a:
        assert len(s) == 160 and len(set(s)) == 160
        assert len(pos & set(s)) == 16
    assert len({tuple(s) for s in a}) == 5


def test_variability_range():
    m = np.array([[0.1, 0.5], [0.2, 0.5], [0.3, 0.5]])
    r = variability_range(m)
    np.testing.assert_allclose(r["low"], [0.12, 0.5])
    np.testing.assert_allclose(r["high"], [0.28, 0.5])
    np.testing.assert_allclose(r["width"], [0.16, 0.0], atol=1e-12)
    with pytest.raises(ValueError):
        variability_range(m[:1])


def test_config_validation():
    assert ReliabilityConfig().conformal_alpha == 0.10
    assert ReliabilityConfig().ensemble_members == 10
    for bad in ({"ensemble_members": 1}, {"conformal_alpha": 1.0},
                {"ensemble_subsample_fraction": 0}):
        with pytest.raises(ValueError):
            ReliabilityConfig(**bad)


# ---------------------------------------------------------------------------
# Nested-CV integration (small run, 2 members)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ncv(tmp_path_factory):
    from dialong_automl.experiments.nested_cv import NestedCVRunner
    from tests.test_nested_cv import _cfg

    runner = NestedCVRunner(_cfg(tmp_path_factory.mktemp("ncvconf"), n=500, repeats=1, trials=2))
    gen = runner.generate()
    cohort = runner.build_cohort(gen.full_df)
    res = runner.run_on_cohort(cohort.observation_data, cohort.patient_targets,
                               generation_metadata=gen.metadata, truth=gen.truth_patients,
                               cohort_audit=cohort.audit.to_dict(), run_name="conf")
    return runner, cohort, res


def test_nested_run_outputs_sets_ranges_and_summary(ncv):
    from dialong_automl.experiments.nested_cv import CALIBRATED_PREDICTOR

    _, _, res = ncv
    cal = res.predictions[res.predictions.predictor == CALIBRATED_PREDICTOR]
    for col in ("set_no_onset", "set_onset", "range_low", "range_high", "range_width",
                "ood_percentile", "ood_category", "reliability_level"):
        assert cal[col].notna().all(), col
    assert set(cal["reliability_level"]) <= {"Low", "Moderate", "High"}
    # Undecided sets must be Low (R6).
    und = cal[cal["set_no_onset"].astype(bool) & cal["set_onset"].astype(bool)]
    assert (und["reliability_level"] == "Low").all()
    assert (cal["range_low"] <= cal["range_high"]).all()
    for f in res.folds:
        assert f["conformal"]["alpha"] == 0.10 and f["ensemble"]["members"] == 2
        assert f["test_access_count"] == 1
    rel = json.loads((res.run_dir / "reliability_summary.json").read_text())
    assert rel["target_coverage"] == pytest.approx(0.9)
    assert 0 <= rel["conformal"]["coverage_onset"] <= 1
    assert rel["ensemble"]["members"] == 2
    assert res.run_result.metadata["reliability"] == rel
    assert sum(v["share"] for v in rel["levels"].values()) == pytest.approx(1.0)
    for f in res.folds:
        lo, hi = f["ensemble"]["width_cutoffs"]
        assert 0 <= lo <= hi and f["ood"]["n_fit"] == f["n_train"]


def test_flipping_test_labels_does_not_change_conformal_or_members(ncv):
    from dialong_automl.experiments.nested_cv import (
        CALIBRATED_PREDICTOR,
        HeldOutFold,
        _subset,
    )

    runner, cohort, res = ncv
    obs, y = cohort.observation_data, cohort.patient_targets.astype(int)
    y.index = y.index.astype(str)
    tr_ids, te_ids = next(runner.outer_splits(sorted(y.index), y, 0))
    flipped = y.copy()
    flipped.loc[te_ids] = 1 - flipped.loc[te_ids]
    tr_obs, tr_t = _subset(obs, y, tr_ids)
    out = []
    for labels in (y, flipped):
        rec, rows = runner.run_fold(0, 0, tr_obs, tr_t, tr_ids, HeldOutFold(obs, labels, te_ids))
        cal = [r for r in rows if r["predictor"] == CALIBRATED_PREDICTOR]
        out.append((rec["conformal"], rec["ensemble"]["width_cutoffs"], rec["ood"],
                    [(r["set_onset"], r["range_low"], r["ood_percentile"],
                      r["reliability_level"]) for r in cal]))
    assert out[0] == out[1]
