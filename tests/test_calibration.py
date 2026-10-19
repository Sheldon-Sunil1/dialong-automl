"""Phase 11 — calibration metrics tests (ECE, MCE, reliability bins)."""

from __future__ import annotations

import numpy as np
import pytest

from dialong_automl.evaluation.calibration import (
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)


def test_perfectly_calibrated_has_near_zero_ece():
    # Half the samples at p=0.0 (all negative) and half at p=1.0 (all positive).
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-12)
    assert maximum_calibration_error(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-12)


def test_maximally_miscalibrated():
    # Confident-but-wrong: predict 1.0 for negatives and 0.0 for positives.
    y = np.array([0, 0, 1, 1])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(1.0, abs=1e-9)
    assert maximum_calibration_error(y, p, n_bins=10) == pytest.approx(1.0, abs=1e-9)


def test_known_ece_single_bin():
    # One bin: mean predicted = 0.6, observed frequency = 0.5 → ECE = 0.1.
    y = np.array([0, 1])
    p = np.array([0.6, 0.6])
    assert expected_calibration_error(y, p, n_bins=1) == pytest.approx(0.1, abs=1e-9)


def test_reliability_bins_structure_and_empty_bins():
    y = np.array([0, 1])
    p = np.array([0.05, 0.95])
    bins = reliability_bins(y, p, n_bins=10)
    assert len(bins) == 10
    # Exactly two non-empty bins (first and last).
    non_empty = [b for b in bins if b["count"] > 0]
    assert len(non_empty) == 2
    # Empty bins report None stats, not fabricated values.
    empty = [b for b in bins if b["count"] == 0]
    assert all(b["mean_predicted"] is None and b["observed_frequency"] is None for b in empty)


def test_probability_exactly_one_lands_in_last_bin():
    y = np.array([1, 1])
    p = np.array([1.0, 1.0])
    bins = reliability_bins(y, p, n_bins=5)
    assert bins[-1]["count"] == 2
    assert sum(b["count"] for b in bins) == 2


def test_input_validation():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0, 1]), np.array([0.5]))  # length mismatch
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0, 2]), np.array([0.5, 0.5]))  # non-binary
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0, 1]), np.array([0.5, 1.5]))  # out of range
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0, 1]), np.array([0.5, np.nan]))  # NaN
    with pytest.raises(ValueError):
        reliability_bins(np.array([0, 1]), np.array([0.5, 0.5]), n_bins=0)  # bad n_bins


def test_calibration_report_bundle():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.6])
    rep = calibration_report(y, p, n_bins=10)
    assert rep["n_bins"] == 10
    assert rep["n_samples"] == 5
    assert 0.0 <= rep["ece"] <= 1.0
    assert 0.0 <= rep["mce"] <= 1.0
    assert len(rep["reliability_bins"]) == 10
    assert "not comparable" in rep["note"].lower()


def test_deterministic():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.3, 0.7, 0.6, 0.4, 0.55])
    a = expected_calibration_error(y, p, n_bins=8)
    b = expected_calibration_error(y, p, n_bins=8)
    assert a == b


# ===========================================================================
# Phase 11 review fixes — MCE low-count support visibility
# ===========================================================================

def test_single_class_ece_defined():
    """ECE is well-defined when y_true has a single class (regression: this was
    untested). All-positive labels with p=0.9 → gap 0.1."""
    y = np.ones(10, dtype=int)
    p = np.full(10, 0.9)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.1, abs=1e-9)
    assert maximum_calibration_error(y, p, n_bins=10) == pytest.approx(0.1, abs=1e-9)


def test_report_exposes_mce_bin_support():
    """calibration_report must expose the count behind the MCE-driving bin so a
    low-support max gap can be recognised."""
    # 999 well-calibrated samples near p=0.5, plus a single wrong sample isolated
    # in its own high-probability bin → MCE is driven by that 1-sample bin.
    rng = np.random.default_rng(0)
    p_main = np.full(999, 0.5)
    y_main = (rng.random(999) < 0.5).astype(int)
    p = np.concatenate([p_main, [0.975]])   # lands in last of 20 bins
    y = np.concatenate([y_main, [0]])        # confident-but-wrong
    rep = calibration_report(y, p, n_bins=20)
    assert "mce_bin_count" in rep and "mce_bin_index" in rep
    # The bin driving MCE holds exactly the single isolated sample.
    assert rep["mce_bin_count"] == 1
    assert rep["mce"] == pytest.approx(
        maximum_calibration_error(y, p, n_bins=20), abs=1e-12
    )
    assert "count-insensitive" in rep["note"].lower()


def test_report_mce_bin_matches_reliability_bins():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.85, 0.9, 0.95, 0.99])
    rep = calibration_report(y, p, n_bins=10)
    # The reported mce_bin_count equals the count of the bin at mce_bin_index.
    target = next(b for b in rep["reliability_bins"]
                  if b["bin_index"] == rep["mce_bin_index"])
    assert target["count"] == rep["mce_bin_count"]
