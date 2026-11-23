"""Post-hoc probability calibration and calibration metrics (Phase 14B, item R1).

Calibrators map a model's raw probability to a calibrated probability. They are
fitted on predictions for patients the model was NOT trained on (inner-CV
out-of-fold predictions in nested CV; validation-split predictions in the
single-split pipeline) and are never fitted on the patients they are evaluated on.

Methods
-------
``platt``    logistic regression of y on logit(p): p' = sigmoid(a + b * logit(p)).
             Strictly increasing (b > 0 always: a non-positive fitted slope
             falls back to identity, constant fit input to an intercept-only
             shift with b = 1), so AUROC computed on one model's scores is
             unchanged up to floating-point resolution. Default.
``isotonic`` non-decreasing step function (pool-adjacent-violators). Can create
             ties and slightly change AUROC; use for sensitivity analysis.
``none``     identity.

Calibrators serialise to plain JSON (``to_dict`` / ``calibrator_from_dict``).

Metrics
-------
``calibration_slope_intercept``: slope = b in y ~ a + b*logit(p) (1 = ideal);
calibration-in-the-large = a in y ~ a + offset(logit(p)) (0 = ideal; negative
means the model predicts too high on average).
``ece_equal_mass``: ECE over bins holding (nearly) equal numbers of patients.

Pure NumPy and deterministic. Synthetic-data project: calibration here describes
model outputs on synthetic data only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dialong_automl.evaluation.calibration import expected_calibration_error

CALIBRATION_METHODS: tuple[str, ...] = ("platt", "isotonic", "none")
# Logit clip. Small enough that distinct model outputs (e.g. 1e-12 vs 1e-9 from
# a confident logistic regression) keep distinct logits; a 1e-6 clip tied them
# and changed AUROC.
_EPS = 1e-15


def _check(p: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("Empty probability array.")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    if y is None:
        return p, None
    y = np.asarray(y).astype(np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"Length mismatch: {y.shape[0]} labels vs {p.shape[0]} probabilities.")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("Labels must be binary 0/1.")
    return p, y


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, _EPS, 1 - _EPS)
    return np.log(q / (1 - q))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-free logistic function."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def fit_logistic(x: np.ndarray | None, y: np.ndarray, offset: np.ndarray | None = None,
                 ridge: float = 1e-8, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Newton-Raphson logistic regression with intercept (and optional slope on x).

    Returns ``[intercept]`` or ``[intercept, slope]``. A tiny ridge keeps the
    solution finite under (quasi-)separation.
    """
    n = y.shape[0]
    X = np.ones((n, 1)) if x is None else np.column_stack([np.ones(n), x])
    off = np.zeros(n) if offset is None else offset
    beta = np.zeros(X.shape[1])
    eye = np.eye(X.shape[1])
    for _ in range(max_iter):
        mu = _sigmoid(X @ beta + off)
        w = mu * (1 - mu)
        grad = X.T @ (y - mu) - ridge * beta
        hess = X.T @ (X * w[:, None]) + ridge * eye
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

class _Calibrator:
    method = ""

    def __init__(self) -> None:
        self.fitted = False
        self.n_fit: int | None = None
        self.n_positive_fit: int | None = None
        self.fallback_reason: str | None = None

    def _record(self, y: np.ndarray) -> None:
        self.n_fit, self.n_positive_fit = int(y.size), int(y.sum())
        self.fitted = True

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform().")

    def fit(self, p: np.ndarray, y: np.ndarray) -> _Calibrator:
        raise NotImplementedError

    def transform(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _base_dict(self) -> dict[str, Any]:
        return {"method": self.method, "n_fit": self.n_fit, "n_positive_fit": self.n_positive_fit,
                "fallback_reason": self.fallback_reason}


class IdentityCalibrator(_Calibrator):
    method = "none"

    def fit(self, p, y):
        p, y = _check(p, y)
        self._record(y)
        return self

    def transform(self, p):
        self._require_fitted()
        return _check(p)[0].copy()

    def to_dict(self) -> dict[str, Any]:
        return self._base_dict()


class PlattCalibrator(_Calibrator):
    """p' = sigmoid(a + b * logit(p)), fitted by maximum likelihood."""

    method = "platt"

    def __init__(self) -> None:
        super().__init__()
        self.intercept: float | None = None
        self.slope: float | None = None

    def fit(self, p, y):
        p, y = _check(p, y)
        if len(np.unique(y)) < 2:
            raise ValueError("Platt calibration needs both classes in the fit data.")
        z = _logit(p)
        if np.ptp(z) == 0:
            # Constant fit input: no slope can be estimated. Recalibrate in the
            # large only (slope 1, logit shift), which maps the fit value to the
            # prevalence and keeps the ranking of later, non-constant inputs.
            # (A slope-0 fit would collapse every prediction to one value.)
            self.intercept = float(fit_logistic(None, y, offset=z)[0])
            self.slope = 1.0
            self.fallback_reason = ("constant input probabilities: intercept-only shift to the "
                                    "prevalence (slope fixed at 1)")
        else:
            a, b = fit_logistic(z, y)
            if b <= 0:
                self.intercept, self.slope = 0.0, 1.0
                self.fallback_reason = (
                    f"fitted slope {b:.4f} <= 0 (scores not positively related to outcome); "
                    "identity mapping kept"
                )
            else:
                self.intercept, self.slope = float(a), float(b)
        self._record(y)
        return self

    def transform(self, p):
        self._require_fitted()
        p = _check(p)[0]
        return _sigmoid(self.intercept + self.slope * _logit(p))

    def to_dict(self) -> dict[str, Any]:
        return {**self._base_dict(), "intercept": self.intercept, "slope": self.slope}


class IsotonicCalibrator(_Calibrator):
    """Non-decreasing step function fitted by pool-adjacent-violators.

    Prediction for a new p: value of the block whose input range contains p;
    between blocks the value of the block to the left; clipped at both ends.
    """

    method = "isotonic"

    def __init__(self) -> None:
        super().__init__()
        self.x_lower: list[float] = []
        self.values: list[float] = []

    def fit(self, p, y):
        p, y = _check(p, y)
        if len(np.unique(y)) < 2:
            raise ValueError("Isotonic calibration needs both classes in the fit data.")
        order = np.argsort(p, kind="mergesort")
        xs, ys = p[order], y[order]
        # Merge equal x first so tied inputs always map to one value.
        ux, start = np.unique(xs, return_index=True)
        sums = np.add.reduceat(ys, start)
        counts = np.diff(np.append(start, xs.size)).astype(float)
        blocks: list[list[float]] = []          # [x_lower, sum, count]
        for x0, s, c in zip(ux, sums, counts):
            blocks.append([float(x0), float(s), float(c)])
            while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
                (x_low, s1, c1), (_, s2, c2) = blocks[-2], blocks[-1]
                blocks[-2:] = [[x_low, s1 + s2, c1 + c2]]
        self.x_lower = [b[0] for b in blocks]
        self.values = [b[1] / b[2] for b in blocks]
        self._record(y)
        return self

    def transform(self, p):
        self._require_fitted()
        p = _check(p)[0]
        idx = np.searchsorted(np.asarray(self.x_lower), p, side="right") - 1
        idx = np.clip(idx, 0, len(self.values) - 1)
        return np.asarray(self.values)[idx]

    def to_dict(self) -> dict[str, Any]:
        return {**self._base_dict(), "x_lower": list(self.x_lower), "values": list(self.values)}


def make_calibrator(method: str) -> _Calibrator:
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method {method!r}; expected {CALIBRATION_METHODS}.")
    return {"platt": PlattCalibrator, "isotonic": IsotonicCalibrator,
            "none": IdentityCalibrator}[method]()


def calibrator_from_dict(d: dict[str, Any]) -> _Calibrator:
    """Rebuild a fitted calibrator from :meth:`to_dict` output."""
    cal = make_calibrator(d["method"])
    cal.n_fit, cal.n_positive_fit = d.get("n_fit"), d.get("n_positive_fit")
    cal.fallback_reason = d.get("fallback_reason")
    if isinstance(cal, PlattCalibrator):
        cal.intercept, cal.slope = float(d["intercept"]), float(d["slope"])
    elif isinstance(cal, IsotonicCalibrator):
        cal.x_lower, cal.values = list(d["x_lower"]), list(d["values"])
        if not cal.values or len(cal.values) != len(cal.x_lower):
            raise ValueError("Corrupt isotonic calibrator.")
    cal.fitted = True
    return cal


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None]:
    """(calibration slope, calibration-in-the-large). None when undefined."""
    p, y = _check(p, y)
    if len(np.unique(y)) < 2:
        return None, None
    z = _logit(p)
    slope = float(fit_logistic(z, y)[1]) if np.ptp(z) > 0 else None
    citl = float(fit_logistic(None, y, offset=z)[0])
    return slope, citl


def ece_equal_mass(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with ``n_bins`` bins of (nearly) equal size over sorted probabilities."""
    p, y = _check(p, y)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    order = np.argsort(p, kind="mergesort")
    total = 0.0
    for chunk in np.array_split(order, min(n_bins, p.size)):
        if chunk.size:
            total += chunk.size / p.size * abs(y[chunk].mean() - p[chunk].mean())
    return float(total)


def calibration_summary(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Calibration metrics for one set of probabilities (JSON-safe)."""
    p_arr, y_arr = _check(p, y)
    slope, citl = calibration_slope_intercept(y_arr, p_arr)
    return {
        "n": int(p_arr.size), "n_positive": int(y_arr.sum()),
        "mean_predicted": float(p_arr.mean()), "observed_rate": float(y_arr.mean()),
        "ece": float(expected_calibration_error(y_arr.astype(int), p_arr, n_bins)),
        "ece_equal_mass": ece_equal_mass(y_arr, p_arr, n_bins),
        "calibration_slope": slope, "calibration_in_the_large": citl, "n_bins": n_bins,
    }
