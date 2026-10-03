"""Random Forest model wrapper for DiaLong-AutoML.

Uses the wide longitudinal representation.
Wraps ``sklearn.ensemble.RandomForestClassifier``.

If sklearn is unavailable, ImportError surfaces at fit() time.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from dialong_automl.models.base import BaseModel, REPRESENTATION_WIDE

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier on the wide representation."""

    name = "random_forest"
    representation = REPRESENTATION_WIDE

    _DEFAULTS: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        super().__init__({**self._DEFAULTS, **(hyperparams or {})})
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as e:
            raise ImportError(
                "sklearn is required for RandomForestModel but could not be "
                f"imported (possibly blocked by OS Application Control policy): {e}"
            ) from e

        self._clf = RandomForestClassifier(
            n_estimators=int(self.hyperparams["n_estimators"]),
            max_depth=self.hyperparams.get("max_depth"),
            min_samples_split=int(self.hyperparams.get("min_samples_split", 2)),
            min_samples_leaf=int(self.hyperparams.get("min_samples_leaf", 1)),
            max_features=self.hyperparams.get("max_features", "sqrt"),
            class_weight=self.hyperparams.get("class_weight"),
            random_state=int(self.hyperparams.get("random_state", 42)),
            n_jobs=int(self.hyperparams.get("n_jobs", -1)),
        )
        self._clf.fit(X, y.ravel())
        self._fitted = True
        logger.info("RandomForestModel fitted (n=%d, estimators=%d)", len(y), self._clf.n_estimators)
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
        logger.info("RandomForestModel saved -> %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestModel":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj)}")
        return obj
