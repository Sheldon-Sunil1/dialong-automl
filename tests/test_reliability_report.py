"""Phase 14D: input-quality flags (R4), OOD score (R5), reliability level (R6). Synthetic only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dialong_automl.config.schema import ReliabilityConfig
from dialong_automl.reliability.ood import MahalanobisOOD
from dialong_automl.reliability.quality import Flag, quality_flags
from dialong_automl.reliability.report import build_report, reliability_level, width_cutoffs


def _visits(dates, hba1c, glucose=(100, 100, 100, 100), **extra):
    n = len(dates)
    d = {"patient_id": ["p1"] * n, "visit_date": dates, "hba1c": list(hba1c)[:n],
         "fasting_glucose": list(glucose)[:n]}
    d.update({k: v[:n] for k, v in extra.items()})
    return pd.DataFrame(d)


DATES = ["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01"]


def codes(flags):
    return sorted(f.code for f in flags)


def test_clean_patient_has_no_flags():
    assert quality_flags(_visits(DATES, [5.5, 5.6, 5.7, 5.8]), max_visits=4) == []


@pytest.mark.parametrize("hba1c, expected", [
    ([None, None, None, None], ["no_hba1c"]),
    ([None, None, None, 5.8], ["single_hba1c"]),
    ([5.5, None, None, None], ["single_hba1c", "stale_hba1c"]),   # 547 days
    ([5.5, 5.6, None, None], ["old_hba1c"]),          # exactly 365 days: old, not stale
    ([5.5, 5.6, 5.7, None], ["old_hba1c"]),           # 181 days
])
def test_hba1c_flags(hba1c, expected):
    assert codes(quality_flags(_visits(DATES, hba1c), max_visits=4)) == expected


def test_other_flags_and_severities():
    f = quality_flags(_visits(DATES[:2], [5.5, 5.6], glucose=(None, None)), max_visits=4)
    assert codes(f) == ["no_glucose", "short_history"]
    assert {x.code: x.severity for x in f} == {"no_glucose": "moderate", "short_history": "info"}
    bad = quality_flags(_visits(DATES, [5.5, 5.6, 5.7, 25.0], bmi=[25, 25, 25, 90]), 4)
    imp = next(x for x in bad if x.code == "implausible_value")
    assert imp.severity == "severe" and "hba1c" in imp.message and "bmi" in imp.message
    assert codes(quality_flags(pd.DataFrame(columns=["visit_date"]), 4)) == ["no_visits"]


def test_ood_flags_an_injected_outlier_and_round_trips():
    rng = np.random.default_rng(0)
    cols = ["a", "b", "c"]
    train = pd.DataFrame(rng.normal(size=(500, 3)), columns=cols)
    train.iloc[::7, 1] = np.nan                         # missing values are imputed
    ood = MahalanobisOOD(columns=cols).fit(train)
    test = pd.DataFrame([[0.0, 0.0, 0.0], [8.0, -8.0, 8.0]], columns=cols)
    assert ood.category(test) == ["typical", "far"]
    pct = ood.percentile(test)
    assert pct[0] < 20 and pct[1] == 100
    # About 5 % / 1 % of training patients exceed the cut-offs by construction.
    assert np.mean(np.array(ood.category(train)) != "typical") == pytest.approx(0.05, abs=0.01)
    again = MahalanobisOOD.from_dict(json.loads(json.dumps(ood.to_dict())))
    np.testing.assert_allclose(again.distance(test), ood.distance(test))


SEV, MOD, INFO = (Flag("s", "severe", "S"), Flag("m", "moderate", "M"), Flag("i", "info", "I"))


@pytest.mark.parametrize("flags, pset, width, level", [
    ([], "no_onset", 0.01, "High"),
    ([INFO], "onset", 0.01, "High"),                  # info never changes the level
    ([MOD], "no_onset", 0.01, "Moderate"),
    ([], "no_onset", 0.15, "Moderate"),               # width above the 80th cut-off
    ([], "no_onset", 0.30, "Low"),                    # above the 95th
    ([], "undecided", 0.01, "Low"),
    ([], "empty", 0.01, "Low"),
    ([SEV], "no_onset", 0.01, "Low"),
    ([MOD, SEV], "no_onset", 0.15, "Low"),
    ([], None, None, "High"),                         # missing components are not used
])
def test_level_truth_table(flags, pset, width, level):
    got, reasons = reliability_level(flags, pset, width, cutoffs=(0.1, 0.2))
    assert got == level
    assert (reasons == []) == (level == "High")


def test_report_adds_ood_flag_and_lists_all_reasons():
    rep = build_report([MOD], "onset", 0.2, 0.25, (0.1, 0.2), "far", 99.6)
    assert rep.level == "Low"
    assert [f.code for f in rep.flags] == ["m", "ood_far"]
    assert len(rep.reasons) == 2 and "99th" in rep.reasons[0]
    assert json.loads(json.dumps(rep.to_dict()))["level"] == "Low"
    assert build_report([], "no_onset", ood_category="unusual", ood_percentile=97).level \
        == "Moderate"


def test_width_cutoffs_and_config():
    lo, hi = width_cutoffs(np.arange(101) / 100)
    assert (lo, hi) == pytest.approx((0.80, 0.95))
    cfg = ReliabilityConfig()
    assert cfg.ood_percentiles == (95.0, 99.0) and cfg.width_percentiles == (80.0, 95.0)
    assert cfg.hba1c_age_days == (180, 365)
    for bad in ({"ood_percentiles": (99, 95)}, {"width_percentiles": (80, 100)},
                {"hba1c_age_days": (0, 10)}):
        with pytest.raises(ValueError):
            ReliabilityConfig(**bad)
