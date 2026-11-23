"""Class-conditional (Mondrian) split-conformal prediction sets (Phase 14C, item R3).

Nonconformity of a candidate label c for probability p (of onset):
``s(c) = 1 - p(c)``, i.e. ``1 - p`` for onset and ``p`` for no onset.
For each class c the threshold is the k-th smallest calibration score of class
c with ``k = ceil((n_c + 1) * (1 - alpha))``; if ``k > n_c`` the threshold is
+inf (the class is always included). Label c enters the set if ``s(c) <= q_c``.

Guarantee: with exchangeable calibration and test patients, P(true label in
set | true label = c) >= 1 - alpha for each class. In nested CV the calibration
scores come from inner-fold models rather than the final model, so the
guarantee there is APPROXIMATE and is checked empirically.

Pure NumPy and deterministic. Synthetic-data project.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

CONFORMAL_MODE = "class_conditional"
SET_LABELS = {(False, False): "empty", (True, False): "no_onset",
              (False, True): "onset", (True, True): "undecided"}


def _check(p: np.ndarray, y: np.ndarray | None = None):
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.size == 0 or np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be a non-empty array in [0, 1].")
    if y is None:
        return p, None
    y = np.asarray(y).astype(int).ravel()
    if y.shape != p.shape or not set(np.unique(y)) <= {0, 1}:
        raise ValueError("Labels must be binary and match the probabilities in length.")
    return p, y


def _scores(p: np.ndarray, label: int) -> np.ndarray:
    return 1.0 - p if label == 1 else p


class MondrianConformal:
    """Per-class split-conformal sets for binary onset prediction."""

    def __init__(self, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1).")
        self.alpha = float(alpha)
        self.q: dict[int, float] = {}
        self.n: dict[int, int] = {}
        self.fitted = False

    def fit(self, p: np.ndarray, y: np.ndarray) -> MondrianConformal:
        p, y = _check(p, y)
        for c in (0, 1):
            s = np.sort(_scores(p[y == c], c))
            n_c = int(s.size)
            if n_c == 0:
                raise ValueError(f"No calibration patients of class {c}.")
            k = math.ceil((n_c + 1) * (1 - self.alpha))
            self.q[c] = float("inf") if k > n_c else float(s[k - 1])
            self.n[c] = n_c
        self.fitted = True
        return self

    def predict_sets(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Boolean arrays ``(contains_no_onset, contains_onset)``."""
        if not self.fitted:
            raise RuntimeError("MondrianConformal must be fitted first.")
        p, _ = _check(p)
        return _scores(p, 0) <= self.q[0], _scores(p, 1) <= self.q[1]

    def to_dict(self) -> dict[str, Any]:
        def enc(v: float) -> float | None:
            return None if math.isinf(v) else v      # JSON has no infinity
        return {"alpha": self.alpha, "mode": CONFORMAL_MODE,
                "q_no_onset": enc(self.q[0]), "q_onset": enc(self.q[1]),
                "n_no_onset": self.n[0], "n_onset": self.n[1],
                "score": "1 - calibrated probability of the candidate label",
                "null_q_means": "class always included (too few calibration patients)"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MondrianConformal:
        m = cls(d["alpha"])
        m.q = {c: float("inf") if d[k] is None else float(d[k])
               for c, k in ((0, "q_no_onset"), (1, "q_onset"))}
        m.n = {0: int(d["n_no_onset"]), 1: int(d["n_onset"])}
        m.fitted = True
        return m


def set_labels(has0: np.ndarray, has1: np.ndarray) -> list[str]:
    return [SET_LABELS[(bool(a), bool(b))] for a, b in zip(has0, has1)]


def coverage_summary(y: np.ndarray, has0: np.ndarray, has1: np.ndarray) -> dict[str, Any]:
    """Per-class and overall coverage plus the set-type distribution (JSON-safe)."""
    y = np.asarray(y).astype(int)
    has0, has1 = np.asarray(has0, bool), np.asarray(has1, bool)
    covered = np.where(y == 1, has1, has0)
    out: dict[str, Any] = {"n": int(y.size), "coverage": float(covered.mean())}
    for c, name in ((0, "no_onset"), (1, "onset")):
        m = y == c
        out[f"coverage_{name}"] = float(covered[m].mean()) if m.any() else None
        out[f"n_{name}"] = int(m.sum())
    labels = set_labels(has0, has1)
    for k in ("no_onset", "onset", "undecided", "empty"):
        out[f"share_{k}"] = float(np.mean([lab == k for lab in labels]))
    out["mean_set_size"] = float((has0.astype(int) + has1.astype(int)).mean())
    return out
