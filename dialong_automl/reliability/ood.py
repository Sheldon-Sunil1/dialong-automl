"""Out-of-distribution score (Phase 14D, item R5).

Mahalanobis distance on the numeric B3 aggregate features (the same features
whatever model is served), median-imputed and standardised with training
statistics, with a Ledoit-Wolf shrinkage covariance fitted on the training
patients. The output is the percentile of a patient's distance among the
training patients' own distances. Above the 95th percentile: "unusual"
(moderate); above the 99th: "far" (severe).

Uses no labels. Synthetic-data project.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dialong_automl.baselines.predictors import b3_columns


def default_ood_columns() -> list[str]:
    return list(b3_columns("all")[0])


class MahalanobisOOD:
    def __init__(self, columns: list[str] | None = None,
                 percentiles: tuple[float, float] = (95.0, 99.0)) -> None:
        self.columns = list(columns) if columns is not None else default_ood_columns()
        self.percentiles = tuple(float(p) for p in percentiles)
        self.fitted = False

    def _matrix(self, wide: pd.DataFrame, fit: bool) -> np.ndarray:
        x = wide.reindex(columns=self.columns).apply(pd.to_numeric, errors="coerce")
        x = x.to_numpy(dtype=np.float64)
        if fit:
            med = np.nanmedian(x, axis=0)
            self.median = np.where(np.isnan(med), 0.0, med)
        x = np.where(np.isnan(x), self.median, x)
        if fit:
            sd = x.std(axis=0)
            self.mean, self.scale = x.mean(axis=0), np.where(sd > 0, sd, 1.0)
        return (x - self.mean) / self.scale

    def fit(self, wide: pd.DataFrame) -> MahalanobisOOD:
        from sklearn.covariance import LedoitWolf

        z = self._matrix(wide, fit=True)
        if z.shape[0] < 3:
            raise ValueError("Need at least 3 training patients for the OOD reference.")
        self.precision = LedoitWolf().fit(z).precision_
        self.reference = np.sort(self._dist(z))
        self.cutoffs = [float(np.percentile(self.reference, p)) for p in self.percentiles]
        self.n_fit = int(z.shape[0])
        self.fitted = True
        return self

    def _dist(self, z: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", z, self.precision, z), 0.0))

    def distance(self, wide: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MahalanobisOOD must be fitted first.")
        return self._dist(self._matrix(wide, fit=False))

    def percentile(self, wide: pd.DataFrame) -> np.ndarray:
        """Share (0-100) of training patients with a smaller distance."""
        d = self.distance(wide)
        return 100.0 * np.searchsorted(self.reference, d, side="left") / self.reference.size

    def category(self, wide: pd.DataFrame) -> list[str]:
        d = self.distance(wide)
        lo, hi = self.cutoffs
        return ["far" if x > hi else "unusual" if x > lo else "typical" for x in d]

    def to_dict(self) -> dict[str, Any]:
        return {"columns": self.columns, "percentiles": list(self.percentiles),
                "median": self.median.tolist(), "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "precision": self.precision.tolist(),
                "reference": self.reference.tolist(), "cutoffs": self.cutoffs,
                "n_fit": self.n_fit, "covariance": "Ledoit-Wolf shrinkage"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MahalanobisOOD:
        m = cls(d["columns"], tuple(d["percentiles"]))
        for k in ("median", "mean", "scale", "precision", "reference"):
            setattr(m, k, np.asarray(d[k], dtype=np.float64))
        m.cutoffs, m.n_fit, m.fitted = list(d["cutoffs"]), int(d["n_fit"]), True
        return m
