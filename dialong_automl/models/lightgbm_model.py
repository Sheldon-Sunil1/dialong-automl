"""LightGBM model wrapper for DiaLong-AutoML.

Uses the wide longitudinal representation.
Wraps ``lightgbm.LGBMClassifier``.

If lightgbm cannot be imported this module raises ``ImportError`` immediately
— it does not silently fall back to another model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
except ImportError as _lgbm_err:
    raise ImportError(
        "lightgbm is required for LightGBMModel but could not be imported. "
        f"Original error: {_lgbm_err}"
    ) from _lgbm_err

from dialong_automl.models.base import REPRESENTATION_WIDE, BaseModel
from dialong_automl.models.persist import dump_fitted, load_fitted

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM classifier on the wide representation.

    Optuna search space
    -------------------
    n_estimators:        int [100, 800]
    learning_rate:       float log [0.01, 0.3]
    num_leaves:          int [8, 128]
    max_depth:           int [-1, 20] (-1 = unlimited)
    min_child_samples:   int [5, 100]
    subsample:           float [0.6, 1.0]
    colsample_bytree:    float [0.6, 1.0]
    lambda_l1:           float log [1e-8, 10]
    lambda_l2:           float log [1e-8, 10]
    """

    name = "lightgbm"
    representation = REPRESENTATION_WIDE

    _DEFAULTS: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "subsample_freq": 1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        super().__init__({**self._DEFAULTS, **(hyperparams or {})})
        self._clf = None

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        hp = self.hyperparams
        self._clf = _LGBMClassifier(
            n_estimators=int(hp["n_estimators"]),
            learning_rate=float(hp["learning_rate"]),
            num_leaves=int(hp["num_leaves"]),
            max_depth=int(hp["max_depth"]),
            min_child_samples=int(hp["min_child_samples"]),
            subsample=float(hp["subsample"]),
            colsample_bytree=float(hp["colsample_bytree"]),
            lambda_l1=float(hp["lambda_l1"]),
            lambda_l2=float(hp["lambda_l2"]),
            random_state=int(hp.get("random_state", 42)),
            n_jobs=int(hp.get("n_jobs", -1)),
            verbose=int(hp.get("verbose", -1)),
            # subsample is ignored unless subsample_freq > 0
            subsample_freq=int(hp.get("subsample_freq", 1)),
        )
        self._clf.fit(X, y.ravel())
        self._fitted = True
        logger.info("LightGBMModel fitted (n=%d, features=%d)", len(y), X.shape[1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self._clf.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        dump_fitted(self, path)
        logger.info("LightGBMModel saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMModel":
        obj = load_fitted(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj)}")
        return obj
