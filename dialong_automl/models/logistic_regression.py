"""Logistic Regression model wrapper for DiaLong-AutoML.

Uses the wide longitudinal representation.
Wraps ``sklearn.linear_model.LogisticRegression``.

If sklearn is unavailable (e.g. blocked DLL on this machine), this module
still imports cleanly. The ImportError surfaces at ``fit()`` time with a
clear message identifying the dependency problem.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from dialong_automl.models.base import BaseModel, REPRESENTATION_WIDE

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier on the wide representation.

    Optuna search space
    -------------------
    C:            log-uniform [1e-4, 100]
    class_weight: None | "balanced"
    solver:       "lbfgs" | "saga"
    """

    name = "logistic_regression"
    representation = REPRESENTATION_WIDE

    _DEFAULTS: dict[str, Any] = {
        "C": 1.0,
        "class_weight": None,
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": 42,
    }

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        super().__init__({**self._DEFAULTS, **(hyperparams or {})})
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as e:
            raise ImportError(
                "sklearn is required for LogisticRegressionModel but could not be "
                f"imported (possibly blocked by OS Application Control policy): {e}"
            ) from e

        self._clf = LogisticRegression(
            C=float(self.hyperparams["C"]),
            class_weight=self.hyperparams.get("class_weight"),
            solver=self.hyperparams.get("solver", "lbfgs"),
            max_iter=int(self.hyperparams.get("max_iter", 2000)),
            random_state=int(self.hyperparams.get("random_state", 42)),
        )
        self._clf.fit(X, y.ravel())
        self._fitted = True
        logger.info("LogisticRegressionModel fitted (n=%d, features=%d)", len(y), X.shape[1])
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
        logger.info("LogisticRegressionModel saved -> %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionModel":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj)}")
        return obj
