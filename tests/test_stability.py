"""Phase 11 — explanation-stability utility tests."""

from __future__ import annotations

import pytest

from dialong_automl.evaluation.stability import (
    shared_key_count,
    spearman_rank_corr,
    stability_report,
    topk_jaccard,
)


# --- top-k Jaccard ---------------------------------------------------------

def test_jaccard_identical_topk_is_one():
    a = {"f1": 0.9, "f2": 0.5, "f3": 0.1}
    assert topk_jaccard(a, a, k=2) == 1.0


def test_jaccard_disjoint_topk_is_zero():
    a = {"f1": 0.9, "f2": 0.8, "f3": 0.0, "f4": 0.0}
    b = {"f1": 0.0, "f2": 0.0, "f3": 0.9, "f4": 0.8}
    assert topk_jaccard(a, b, k=2) == 0.0


def test_jaccard_partial_overlap():
    a = {"f1": 0.9, "f2": 0.8, "f3": 0.1}
    b = {"f1": 0.9, "f2": 0.1, "f3": 0.8}
    # top-2 of a = {f1,f2}; top-2 of b = {f1,f3}; Jaccard = 1/3.
    assert topk_jaccard(a, b, k=2) == pytest.approx(1 / 3)


def test_jaccard_uses_magnitude():
    # Large-magnitude negative influence should rank in the top-k.
    a = {"t0": -0.9, "t1": 0.2, "t2": 0.05}
    b = {"t0": 0.9, "t1": 0.2, "t2": 0.05}
    assert topk_jaccard(a, b, k=1) == 1.0  # both top-1 = t0


def test_jaccard_k_zero_is_vacuously_one():
    assert topk_jaccard({"a": 1.0}, {"b": 1.0}, k=0) == 1.0


def test_jaccard_accepts_ordered_lists():
    # Ordered key lists (most important first).
    assert topk_jaccard(["f1", "f2", "f3"], ["f1", "f2", "f9"], k=2) == 1.0


def test_jaccard_rejects_negative_k():
    with pytest.raises(ValueError):
        topk_jaccard({"a": 1.0}, {"a": 1.0}, k=-1)


# --- Spearman rank correlation ---------------------------------------------

def test_spearman_identical_is_one():
    a = {"f1": 0.9, "f2": 0.5, "f3": 0.1, "f4": 0.3}
    assert spearman_rank_corr(a, a) == pytest.approx(1.0)


def test_spearman_reversed_is_minus_one():
    a = {"f1": 1.0, "f2": 2.0, "f3": 3.0, "f4": 4.0}
    b = {"f1": 4.0, "f2": 3.0, "f3": 2.0, "f4": 1.0}
    assert spearman_rank_corr(a, b) == pytest.approx(-1.0)


def test_spearman_constant_ranking_is_none():
    a = {"f1": 0.5, "f2": 0.5, "f3": 0.5}  # all tied → constant ranks
    b = {"f1": 0.9, "f2": 0.1, "f3": 0.5}
    assert spearman_rank_corr(a, b) is None


def test_spearman_too_few_shared_keys_is_none():
    assert spearman_rank_corr({"f1": 0.9}, {"f1": 0.9}) is None  # 1 shared key
    assert spearman_rank_corr({"f1": 0.9, "f2": 0.1}, {"f3": 0.5, "f4": 0.2}) is None


def test_spearman_handles_ties_with_average_ranks():
    a = {"a": 1.0, "b": 2.0, "c": 2.0, "d": 4.0}
    # Perfect monotonic with b,c tied — correlation should be high, defined.
    b = {"a": 10.0, "b": 20.0, "c": 20.0, "d": 40.0}
    r = spearman_rank_corr(a, b)
    assert r == pytest.approx(1.0)


def test_spearman_only_shared_keys():
    a = {"f1": 1.0, "f2": 2.0, "f3": 3.0, "extra": 9.0}
    b = {"f1": 1.0, "f2": 2.0, "f3": 3.0}
    assert shared_key_count(a, b) == 3
    assert spearman_rank_corr(a, b) == pytest.approx(1.0)


def test_stability_report_bundle():
    a = {"f1": 0.9, "f2": 0.5, "f3": 0.1}
    b = {"f1": 0.8, "f2": 0.4, "f3": 0.2}
    rep = stability_report(a, b, k=2)
    assert rep["k"] == 2
    assert rep["shared_keys"] == 3
    assert 0.0 <= rep["topk_jaccard"] <= 1.0
    assert rep["spearman"] == pytest.approx(1.0)


def test_works_on_real_explanation_shapes():
    """Feature-importance (SHAP) and timestep-influence (occlusion) shapes."""
    # SHAP-style: {column_name: importance}
    shap_a = {"hba1c_latest": 0.42, "bmi_mean": 0.18, "age_at_visit_slope": 0.05}
    shap_b = {"hba1c_latest": 0.40, "bmi_mean": 0.20, "age_at_visit_slope": 0.03}
    assert topk_jaccard(shap_a, shap_b, k=2) == 1.0
    # Occlusion-style: {timestep: influence}
    occ_a = {0: 0.1, 1: 0.3, 2: -0.05}
    occ_b = {0: 0.12, 1: 0.28, 2: -0.04}
    assert spearman_rank_corr(occ_a, occ_b) == pytest.approx(1.0)
