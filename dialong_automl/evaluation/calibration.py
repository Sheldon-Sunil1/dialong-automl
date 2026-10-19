"""Calibration metrics for DiaLong-AutoML (Phase 11 — research foundation).

Adds reliability-aware evaluation on top of the existing metrics in
``dialong_automl.evaluation.metrics`` (which already provides the Brier score;
it is NOT re-implemented here). These functions are additive and do not change
model training, selection, or the default metric set.

Pure NumPy, deterministic, no sklearn dependency.

Binning
-------
Reliability is assessed with ``n_bins`` equal-width bins over the probability
range ``[0, 1]``. Bin ``i`` covers ``[i/n_bins, (i+1)/n_bins)``; the final bin
is closed on the right so a probability of exactly ``1.0`` falls in the last
bin. Empty bins contribute zero to ECE/MCE and are reported with ``count == 0``.

Caveat
------
ECE and MCE values depend on the number and placement of bins. They are NOT
universally comparable across different binning configurations; always report
``n_bins`` alongside the values and only compare runs that used the same
binning. MCE additionally is count-insensitive — a single low-support bin can
dominate it — so inspect per-bin counts (``calibration_report`` exposes
``mce_bin_count``) before trusting a high MCE.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _validate_inputs(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce/validate binary labels and probabilities in [0, 1]."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_proba, dtype=np.float64)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_proba length mismatch: {yt.shape[0]} vs {yp.shape[0]}."
        )
    if yt.size == 0:
        raise ValueError("Empty inputs: need at least one sample.")
    yt_int = yt.astype(np.int64)
    uniq = set(np.unique(yt_int).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"y_true must be binary (0/1); got values {sorted(uniq)}.")
    if np.isnan(yp).any():
        raise ValueError("y_proba contains NaN.")
    if (yp < 0.0).any() or (yp > 1.0).any():
        raise ValueError("y_proba values must lie in [0, 1].")
    return yt_int, yp


def reliability_bins(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Per-bin reliability statistics for an equal-width binning.

    Returns
    -------
    list of dicts (length ``n_bins``), each with:
        bin_index         : 0-based bin index
        lower, upper      : bin edges
        count             : number of samples in the bin
        mean_predicted    : mean predicted probability in the bin (None if empty)
        observed_frequency: observed fraction of positives in the bin (None if empty)
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    yt, yp = _validate_inputs(y_true, y_proba)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign each probability to a bin; clamp so p == 1.0 lands in the last bin.
    idx = np.minimum((yp * n_bins).astype(np.int64), n_bins - 1)
    idx = np.maximum(idx, 0)

    bins: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            mean_pred: float | None = None
            obs_freq: float | None = None
        else:
            mean_pred = float(yp[mask].mean())
            obs_freq = float(yt[mask].mean())
        bins.append({
            "bin_index": b,
            "lower": float(edges[b]),
            "upper": float(edges[b + 1]),
            "count": count,
            "mean_predicted": mean_pred,
            "observed_frequency": obs_freq,
        })
    return bins


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE): count-weighted mean |acc - conf| per bin.

    ECE = sum_b (count_b / N) * |observed_freq_b - mean_pred_b|.
    Range ``[0, 1]`` (0 = perfectly calibrated under this binning). Empty bins
    contribute 0. Depends on ``n_bins`` — report it alongside the value.
    """
    yt, yp = _validate_inputs(y_true, y_proba)
    n = yt.shape[0]
    total = 0.0
    for b in reliability_bins(yt, yp, n_bins):
        if b["count"] == 0:
            continue
        gap = abs(b["observed_frequency"] - b["mean_predicted"])
        total += (b["count"] / n) * gap
    return float(total)


def maximum_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Maximum Calibration Error (MCE): max over non-empty bins of |acc - conf|.

    Range ``[0, 1]``. Returns 0.0 when there are no non-empty bins (cannot
    occur for valid non-empty inputs). Depends on ``n_bins``.

    Interpretation caveat
    ---------------------
    MCE is the single largest per-bin gap and is **count-insensitive**: a bin
    containing very few samples can dominate the value, so a high MCE may
    reflect low-support noise rather than systematic miscalibration. Always
    inspect the per-bin ``count`` (see :func:`reliability_bins` /
    :func:`calibration_report`, which exposes the support behind the max-gap
    bin) before drawing conclusions. This function intentionally does **not**
    exclude low-count bins or apply a minimum-count threshold — doing so would
    change the mathematically defined MCE; any such filtering must be done
    explicitly by the caller.
    """
    yt, yp = _validate_inputs(y_true, y_proba)
    gaps = [
        abs(b["observed_frequency"] - b["mean_predicted"])
        for b in reliability_bins(yt, yp, n_bins)
        if b["count"] > 0
    ]
    return float(max(gaps)) if gaps else 0.0


def calibration_report(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Convenience bundle: ECE, MCE, bin stats, and the binning configuration."""
    yt, yp = _validate_inputs(y_true, y_proba)
    bins = reliability_bins(yt, yp, n_bins)

    # Identify the bin that drives MCE and expose its support so callers can
    # judge whether the max gap is backed by enough samples. This is reporting
    # only; it does not alter the ECE/MCE values themselves.
    mce_bin_index: int | None = None
    mce_bin_count: int | None = None
    max_gap = -1.0
    for b in bins:
        if b["count"] == 0:
            continue
        gap = abs(b["observed_frequency"] - b["mean_predicted"])
        if gap > max_gap:
            max_gap = gap
            mce_bin_index = b["bin_index"]
            mce_bin_count = b["count"]

    return {
        "n_bins": n_bins,
        "n_samples": int(yt.shape[0]),
        "ece": expected_calibration_error(yt, yp, n_bins),
        "mce": maximum_calibration_error(yt, yp, n_bins),
        "mce_bin_index": mce_bin_index,
        "mce_bin_count": mce_bin_count,
        "reliability_bins": bins,
        "note": (
            "ECE/MCE depend on n_bins and are not comparable across different "
            "binning configurations. MCE is count-insensitive: inspect "
            "mce_bin_count (and per-bin 'count') before trusting a high MCE, as "
            "a low-support bin can dominate it."
        ),
    }
