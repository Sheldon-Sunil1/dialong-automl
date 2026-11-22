"""Phase 13C step 1 (audit item RE-7): tie handling in AUROC / AUPRC.

Tied scores must be grouped into one threshold so results never depend on
row order, and AUPRC is the step-wise average precision. Constant predictors
(the B0 prevalence baseline) must give exactly AUROC 0.5 and AUPRC equal to
the prevalence. scikit-learn is used only as an independent reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from dialong_automl.evaluation.metrics import auprc, auroc, evaluate

sk_metrics = pytest.importorskip("sklearn.metrics")


def _orders(n: int, k: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [rng.permutation(n) for _ in range(k)]


@pytest.mark.parametrize("value", [0.0, 0.0585, 0.5, 1.0])
def test_constant_predictor_auroc_is_exactly_half_for_every_label_order(value):
    y = np.array([0] * 111 + [1] * 7)
    scores = np.full(len(y), value)
    for p in _orders(len(y)):
        assert auroc(y[p], scores[p]) == 0.5


def test_constant_predictor_auprc_equals_prevalence():
    y = np.array([0] * 111 + [1] * 7)
    scores = np.full(len(y), 0.0585)
    for p in _orders(len(y)):
        assert auprc(y[p], scores[p]) == pytest.approx(7 / 118, abs=1e-12)


@pytest.mark.parametrize("levels", [2, 3, 5])
def test_tied_scores_are_order_invariant_and_match_sklearn(levels):
    rng = np.random.default_rng(levels)
    y = rng.integers(0, 2, 300)
    s = rng.integers(0, levels, 300).astype(float)
    ref_auc = sk_metrics.roc_auc_score(y, s)
    ref_ap = sk_metrics.average_precision_score(y, s)
    for p in _orders(len(y)):
        assert auroc(y[p], s[p]) == pytest.approx(ref_auc, abs=1e-12)
        assert auprc(y[p], s[p]) == pytest.approx(ref_ap, abs=1e-12)


def test_partially_tied_saturated_probabilities_match_sklearn():
    """Tree models often saturate at exactly 0 or 1 (e.g. research_v2)."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 400)
    s = np.where(rng.random(400) < 0.4, np.round(rng.random(400)), rng.random(400))
    assert auroc(y, s) == pytest.approx(sk_metrics.roc_auc_score(y, s), abs=1e-12)
    assert auprc(y, s) == pytest.approx(sk_metrics.average_precision_score(y, s), abs=1e-12)


def test_untied_scores_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = rng.random(500)
    assert auroc(y, s) == pytest.approx(sk_metrics.roc_auc_score(y, s), abs=1e-12)
    assert auprc(y, s) == pytest.approx(sk_metrics.average_precision_score(y, s), abs=1e-12)


def test_hand_computed_small_example_with_ties():
    # scores 0.9 (pos), 0.5 (pos, neg tied), 0.1 (neg)
    # AUROC: pairs (pos,neg): (0.9,0.5)=1 (0.9,0.1)=1 (0.5,0.5)=0.5 (0.5,0.1)=1 -> 3.5/4
    # AP: threshold 0.9 -> R=0.5,P=1 ; threshold 0.5 -> R=1,P=2/3 ; AP = 0.5*1 + 0.5*2/3
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.5, 0.5, 0.1])
    assert auroc(y, s) == pytest.approx(0.875, abs=1e-12)
    assert auprc(y, s) == pytest.approx(0.5 + 0.5 * (2 / 3), abs=1e-12)
    # Same result with the tied pair swapped.
    assert auroc(y[[0, 2, 1, 3]], s[[0, 2, 1, 3]]) == pytest.approx(0.875, abs=1e-12)


def test_reversed_perfect_ranking_and_single_class_unchanged():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auroc(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4])) is None
    assert auprc(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4])) is None


def test_evaluate_reports_deterministic_constant_predictor_metrics():
    y = np.array([0] * 30 + [1] * 10)
    res = evaluate(y, np.full(40, 0.25), threshold=0.5)
    assert res["auroc"] == 0.5
    assert res["auprc"] == pytest.approx(0.25)
    assert res["precision"] is None and res["f1"] is None   # no predicted positives
    assert res["brier"] == pytest.approx(0.75 * 0.25 ** 2 + 0.25 * 0.75 ** 2)
