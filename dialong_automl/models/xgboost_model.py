"""XGBoost model wrapper for DiaLong-AutoML.

Uses the wide longitudinal representation.

If xgboost (or its scipy dependency) cannot be imported due to OS
Application Control policy, the error surfaces at fit() time with a
clear message. This module always imports cleanly.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from dialong_automl.models.base import BaseModel, REPRESENTATION_WIDE

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier on the wide representation."""

    name = "xgboost"
    representation = REPRESENTATION_WIDE

    _DEFAULTS: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        super().__init__({**self._DEFAULTS, **(hyperparams or {})})
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "xgboost is required for XGBoostModel but could not be imported "
                f"(possibly blocked by OS Application Control policy): {e}"
            ) from e

        hp = self.hyperparams
        self._clf = XGBClassifier(
            n_estimators=int(hp["n_estimators"]),
            max_depth=int(hp["max_depth"]),
            learning_rate=float(hp["learning_rate"]),
            subsample=float(hp["subsample"]),
            colsample_bytree=float(hp["colsample_bytree"]),
            min_child_weight=int(hp["min_child_weight"]),
            reg_alpha=float(hp["reg_alpha"]),
            reg_lambda=float(hp["reg_lambda"]),
            eval_metric=hp.get("eval_metric", "logloss"),
            random_state=int(hp.get("random_state", 42)),
            n_jobs=int(hp.get("n_jobs", -1)),
            verbosity=0,
        )
        self._clf.fit(X, y.ravel())
        self._fitted = True
        logger.info("XGBoostModel fitted (n=%d, features=%d)", len(y), X.shape[1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self._clf.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("XGBoostModel saved -> %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostModel":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj)}")
        return obj
