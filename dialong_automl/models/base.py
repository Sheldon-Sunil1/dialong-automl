"""Abstract base class for all DiaLong-AutoML model wrappers.

Every model wrapper must implement this interface so the training pipeline
can treat all models uniformly regardless of their underlying library.

DISCLAIMER: Models operate on synthetic demonstration data only.
Not clinical evidence. Not suitable for medical validation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Representation type each model family expects
REPRESENTATION_WIDE = "wide"
REPRESENTATION_SEQUENCE = "sequence"


class BaseModel(ABC):
    """Abstract base for all DiaLong-AutoML model wrappers.

    Attributes
    ----------
    name:
        Unique model identifier matching the registry key.
    representation:
        ``"wide"`` for tabular models, ``"sequence"`` for GRU.
    hyperparams:
        Dict of hyperparameters this instance was created with.
    """

    name: str = "base"
    representation: str = REPRESENTATION_WIDE

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        self.hyperparams: dict[str, Any] = hyperparams or {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit the model on training data.

        Parameters
        ----------
        X: (n_samples, n_features) array — already preprocessed.
        y: (n_samples,) binary integer array.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive class.

        Parameters
        ----------
        X: (n_samples, n_features) array — preprocessed with training stats.

        Returns
        -------
        np.ndarray shape (n_samples,), values in [0, 1].
        """

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at *threshold*.

        Parameters
        ----------
        X: preprocessed feature array.
        threshold: classification cutoff (default 0.5).

        Returns
        -------
        np.ndarray shape (n_samples,) of 0/1 integers.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the fitted model to *path*."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load a previously saved model from *path*."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.hyperparams})"
