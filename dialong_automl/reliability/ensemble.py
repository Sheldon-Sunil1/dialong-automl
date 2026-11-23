"""Model-variability range from refitted members (Phase 14C, item R2).

Each member is the selected configuration refitted on a stratified random
subsample of the training patients (WITHOUT replacement). The design proposed
bootstrap resamples; subsampling is used instead because representations,
vocabularies and preprocessors are keyed by patient id, so duplicated patients
cannot be represented without renaming them. The range is the 10th-90th
percentile of the members' calibrated probabilities. It describes how much the
prediction moves when the training data change. It is NOT a confidence
interval for the patient's true risk.

Deterministic given the seed. Synthetic-data project.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RANGE_PERCENTILES = (10.0, 90.0)
RANGE_NOTE = ("Model variability range: 10th-90th percentile of calibrated probabilities "
              "from models refitted on random subsamples of the training patients. "
              "Not a confidence interval for the patient's true risk.")


def member_subsamples(ids: list[str], y: np.ndarray, n_members: int, fraction: float,
                      seed: int) -> list[list[str]]:
    """Stratified subsamples (without replacement), one per member, sorted ids."""
    if n_members < 0 or not 0 < fraction <= 1:
        raise ValueError("n_members must be >= 0 and fraction in (0, 1].")
    ids_arr, y = np.asarray(ids), np.asarray(y).astype(int)
    if ids_arr.shape != y.shape:
        raise ValueError("ids and labels differ in length.")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_members):
        keep = []
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            k = max(1, int(round(fraction * idx.size))) if idx.size else 0
            keep.append(rng.choice(idx, size=k, replace=False))
        out.append(sorted(ids_arr[np.concatenate(keep)].tolist()))
    return out


def variability_range(member_proba: np.ndarray) -> dict[str, np.ndarray]:
    """``member_proba`` shape (members, patients) -> low, high, width per patient."""
    m = np.asarray(member_proba, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] < 2:
        raise ValueError("Need a (members >= 2, patients) array.")
    lo, hi = np.percentile(m, RANGE_PERCENTILES, axis=0)
    return {"low": lo, "high": hi, "width": hi - lo}


def width_summary(width: np.ndarray) -> dict[str, Any]:
    w = np.asarray(width, dtype=np.float64)
    return {"n": int(w.size), "mean": float(w.mean()), "median": float(np.median(w)),
            "p80": float(np.percentile(w, 80)), "p95": float(np.percentile(w, 95)),
            "max": float(w.max())}
