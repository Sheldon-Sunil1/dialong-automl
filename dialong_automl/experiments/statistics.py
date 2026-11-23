"""Uncertainty estimates for the nested-CV runner (Phase 13E).

Pooled out-of-fold predictions are summarised per repeat; the point estimate
is the mean over repeats. Uncertainty comes from a PATIENT-LEVEL (cluster)
bootstrap: each resample draws patients with replacement and recomputes every
metric on every repeat with the same patients, then averages over repeats.
Using the same resampled patients for every predictor gives PAIRED
comparisons. Two-sided bootstrap p-values are approximate; secondary
comparisons are Holm-adjusted.

These are resampling intervals over patients for a fixed synthetic dataset;
they do not capture variability across datasets or populations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from dialong_automl.evaluation.calibration import expected_calibration_error
from dialong_automl.evaluation.metrics import auprc, auroc, brier_score, f1_score
from dialong_automl.reliability.calibration import calibration_slope_intercept, ece_equal_mass

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], "float | None"]


def _thresholded(fn: Callable[..., Any]) -> MetricFn:
    def metric(y: np.ndarray, s: np.ndarray, thr: np.ndarray) -> float | None:
        # Per-patient thresholds (each outer fold has its own): predict s >= thr.
        pred = (s >= thr).astype(float)
        return fn(y, pred, threshold=0.5)
    return metric


def _recall(y, p, threshold=0.5):
    from dialong_automl.evaluation.metrics import recall
    return recall(y, p, threshold)


def _precision(y, p, threshold=0.5):
    from dialong_automl.evaluation.metrics import precision
    return precision(y, p, threshold)


def _ece(y, s, t):
    return expected_calibration_error(y, s, 10)


def _ece_mass(y, s, t):
    return ece_equal_mass(y, s, 10)


def _cal_slope(y, s, t):
    return calibration_slope_intercept(y, s)[0]


def _cal_intercept(y, s, t):
    return calibration_slope_intercept(y, s)[1]


METRICS: dict[str, tuple[MetricFn, bool]] = {
    # name: (function(y, score, threshold), needs_probabilities)
    "auroc": (lambda y, s, t: auroc(y, s), False),
    "auprc": (lambda y, s, t: auprc(y, s), False),
    "brier": (lambda y, s, t: brier_score(y, s), True),
    "f1": (_thresholded(f1_score), False),
    "recall": (_thresholded(_recall), False),
    "precision": (_thresholded(_precision), False),
    # Phase 14B calibration metrics (probabilities only; 10 bins).
    "ece": (_ece, True),
    "ece_equal_mass": (_ece_mass, True),
    "cal_slope": (_cal_slope, True),
    "cal_intercept": (_cal_intercept, True),
}
#: Fold-aware metrics, computed only when outer-fold ids are supplied.
#: ``auroc_fold_mean`` is the mean of per-outer-fold AUROCs (then averaged over
#: repeats). Added in Phase 14B as a POST-HOC methodological correction: pooled
#: out-of-fold AUROC mixes score scales of independently fitted fold models and
#: penalises models whose scale varies between folds. The pre-specified primary
#: endpoint remains pooled ``auroc``.
FOLD_METRICS: tuple[str, ...] = ("auroc_fold_mean",)
#: All metric names that may appear in summary rows.
ALL_METRICS: tuple[str, ...] = tuple(METRICS) + FOLD_METRICS
#: Metrics with a paired comparison; "lower" = smaller is better.
COMPARED_METRICS: dict[str, str] = {"auroc": "higher", "auprc": "higher", "brier": "lower",
                                    "ece": "lower", "auroc_fold_mean": "higher"}


def fold_mean_auroc(y: np.ndarray, s: np.ndarray, folds: np.ndarray) -> float | None:
    """Mean AUROC over the distinct fold ids in ``folds``; None if any fold is single-class."""
    vals = []
    for f in np.unique(folds):
        m = folds == f
        v = auroc(y[m], s[m])
        if v is None:
            return None
        vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def _per_repeat(metric: MetricFn, y: np.ndarray, scores: np.ndarray, thr: np.ndarray,
                idx: np.ndarray | None = None) -> float | None:
    """Mean over repeats of a metric; None if undefined on any repeat."""
    vals = []
    for r in range(scores.shape[0]):
        s, t, yy = scores[r], thr[r], y
        if idx is not None:
            s, t, yy = s[idx], t[idx], y[idx]
        v = metric(yy, s, t)
        if v is None:
            return None
        vals.append(float(v))
    return float(np.mean(vals))


def _per_repeat_folds(y: np.ndarray, scores: np.ndarray, folds: np.ndarray,
                      idx: np.ndarray | None = None) -> float | None:
    """Mean over repeats of :func:`fold_mean_auroc`; fold membership travels with patients."""
    vals = []
    for r in range(scores.shape[0]):
        s, f, yy = scores[r], folds[r], y
        if idx is not None:
            s, f, yy = s[idx], f[idx], y[idx]
        v = fold_mean_auroc(yy, s, f)
        if v is None:
            return None
        vals.append(v)
    return float(np.mean(vals))


def bootstrap_summary(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, np.ndarray],
    probabilistic: dict[str, bool],
    n_boot: int,
    seed: int,
    reference: str,
    primary: str | None = None,
    not_comparators: dict[str, str] | tuple[str, ...] = (),
    fold_ids: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Point estimates, bootstrap CIs and paired comparisons vs ``reference``.

    ``scores[name]`` and ``thresholds[name]`` have shape ``(repeats, n_patients)``
    and share the patient order of ``y``. Predictors in ``not_comparators``
    (name -> role, e.g. an oracle "ceiling" or a "calibrated_variant") are
    compared but excluded from the Holm family. ``fold_ids`` (shape
    ``(repeats, n_patients)``, outer-fold id of each patient per repeat) enables
    the fold-aware metrics in :data:`FOLD_METRICS`; they are reported and
    compared but never enter the Holm family or the primary comparison.
    Returns ``(summary_rows, comparison_rows)``.
    """
    roles = (dict(not_comparators) if isinstance(not_comparators, dict)
             else {name: "ceiling" for name in not_comparators})
    y = np.asarray(y).astype(int)
    names = list(scores)
    if reference not in scores:
        raise ValueError(f"reference {reference!r} not among predictors {names}")
    n = len(y)
    rng = np.random.default_rng(seed)
    boot_idx = [rng.integers(0, n, n) for _ in range(n_boot)]

    usable = {m: [p for p in names if probabilistic[p] or not needs]
              for m, (_, needs) in METRICS.items()}
    point: dict[tuple[str, str], float | None] = {}
    draws: dict[tuple[str, str], np.ndarray] = {}
    for m, (fn, _) in METRICS.items():
        for p in usable[m]:
            point[(p, m)] = _per_repeat(fn, y, scores[p], thresholds[p])
            vals = [_per_repeat(fn, y, scores[p], thresholds[p], idx) for idx in boot_idx]
            draws[(p, m)] = np.array([np.nan if v is None else v for v in vals], dtype=float)
    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids)
        for p in names:
            if fold_ids.shape != scores[p].shape:
                raise ValueError(f"fold_ids shape {fold_ids.shape} != scores[{p!r}] "
                                 f"shape {scores[p].shape}")
            point[(p, "auroc_fold_mean")] = _per_repeat_folds(y, scores[p], fold_ids)
            vals = [_per_repeat_folds(y, scores[p], fold_ids, idx) for idx in boot_idx]
            draws[(p, "auroc_fold_mean")] = np.array(
                [np.nan if v is None else v for v in vals], dtype=float)

    summary = []
    for p in names:
        row: dict[str, Any] = {"predictor": p, "n_patients": n, "n_positive": int(y.sum()),
                               "repeats": int(scores[p].shape[0])}
        for m in ALL_METRICS:
            if (p, m) not in point:
                row[m] = row[f"{m}_ci_low"] = row[f"{m}_ci_high"] = None
                continue
            row[m] = point[(p, m)]
            d = draws[(p, m)]
            d = d[~np.isnan(d)]
            row[f"{m}_ci_low"] = float(np.percentile(d, 2.5)) if d.size else None
            row[f"{m}_ci_high"] = float(np.percentile(d, 97.5)) if d.size else None
        summary.append(row)

    comparisons = []
    for p in names:
        if p == reference:
            continue
        for m in COMPARED_METRICS:
            if (reference, m) not in draws or (p, m) not in draws:
                continue
            d = draws[(reference, m)] - draws[(p, m)]
            d = d[~np.isnan(d)]
            diff = (point[(reference, m)] - point[(p, m)]
                    if point[(reference, m)] is not None and point[(p, m)] is not None else None)
            comparisons.append({
                "comparison": f"{reference} - {p}", "comparator": p, "metric": m,
                "difference": diff,
                "ci_low": float(np.percentile(d, 2.5)) if d.size else None,
                "ci_high": float(np.percentile(d, 97.5)) if d.size else None,
                "p_value": bootstrap_p_value(d) if d.size else None,
                "primary": bool(p == primary and m == "auroc"),
                "role": roles.get(p, "baseline"),
                "better_direction": COMPARED_METRICS[m],
            })
    # Holm family: secondary AUROC comparisons against baselines (not the
    # primary comparison, not ceilings such as the oracle).
    _holm_inplace([c for c in comparisons if c["metric"] == "auroc" and not c["primary"]
                   and c["role"] == "baseline"])
    for c in comparisons:
        c.setdefault("p_value_holm", c["p_value"] if c["primary"] else None)
    return summary, comparisons


def bootstrap_p_value(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0 (add-one corrected)."""
    d = np.asarray(diffs, dtype=float)
    b = d.size
    lo = (np.sum(d <= 0) + 1) / (b + 1)
    hi = (np.sum(d >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _holm_inplace(rows: list[dict[str, Any]]) -> None:
    ps = [r["p_value"] for r in rows if r["p_value"] is not None]
    adj = iter(holm_adjust(ps))
    for r in rows:
        r["p_value_holm"] = next(adj) if r["p_value"] is not None else None
