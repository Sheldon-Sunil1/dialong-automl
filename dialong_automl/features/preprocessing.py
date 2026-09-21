"""Training-only preprocessing for DiaLong-AutoML (Phase 4).

Provides :class:`WidePreprocessor` for the wide tabular representation and
:class:`SequencePreprocessor` for the padded 3-D sequence array.

Both are implemented in pure NumPy/pandas — no sklearn compiled extensions
are required — so they work in environments where compiled sklearn DLLs may
be blocked by application control policies.

Critical rule
-------------
**FIT only on training data.  NEVER refit on validation or test data.**

Both preprocessors expose:

* ``fit(X_train)``           — learns statistics from training data only
* ``transform(X)``           — applies learned transform; raises if not fitted
* ``fit_transform(X_train)`` — fit + transform in one call

The fitted state is serializable via :mod:`pickle` for later phases.

WidePreprocessor
----------------
Operates on a :class:`pandas.DataFrame` (output of
:class:`~dialong_automl.features.wide.WideRepresentationResult`).

Numeric columns:
    1. Median imputation  (fills NaN with column median from training data)
    2. Standard scaling   (zero-mean, unit-variance using training statistics)

Categorical columns:
    1. Constant imputation — missing values filled with ``"<MISSING>"``
    2. One-hot encoding    — produces one binary column per observed category

The preprocessor outputs a float32 NumPy array alongside a
``feature_names_out`` list.

SequencePreprocessor
--------------------
Operates on the raw 3-D ``X`` array produced by
:class:`~dialong_automl.features.sequence.SequenceRepresentationBuilder`.

Shape: ``(n_patients, max_seq_len, n_features)``

After the builder's one-hot encoding, the feature layout is::

    [numeric_0, ..., numeric_N,
     cat_0__<MISSING>, cat_0__<UNK>, cat_0__value_A, ...,
     cat_1__<MISSING>, ...]

Only **numeric** feature axes are scaled/imputed; one-hot categorical axes
are already binary {0, 1} and are passed through unchanged.

Steps:
    1. Identify numeric axis-2 indices (from ``feature_names`` or explicit list).
    2. Compute median and (mean, std) from non-padded training timesteps only.
    3. Impute NaN with training median, then scale with training mean/std.
    4. Leave one-hot categorical positions unchanged.
    5. Zero-out padding positions (mask == 0) so the GRU can mask reliably.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MISSING_SENTINEL = "<MISSING>"


# ---------------------------------------------------------------------------
# Wide preprocessor
# ---------------------------------------------------------------------------

class WidePreprocessor:
    """Fit-once, transform-many preprocessor for the wide tabular representation.

    Parameters
    ----------
    numeric_columns:
        Column names to treat as numeric (imputed + scaled).
        Auto-detected from float/int dtype when ``None``.
    categorical_columns:
        Column names to treat as categorical (imputed + one-hot encoded).
        Auto-detected from object dtype when ``None``.
    """

    def __init__(
        self,
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> None:
        self.numeric_columns: list[str] | None = numeric_columns
        self.categorical_columns: list[str] | None = categorical_columns

        # Fitted state
        self._num_medians: dict[str, float] = {}
        self._num_means: dict[str, float] = {}
        self._num_stds: dict[str, float] = {}
        self._cat_categories: dict[str, list[str]] = {}   # col -> sorted category list
        self._feature_names_out: list[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "WidePreprocessor":
        """Fit the preprocessor on training data.

        Parameters
        ----------
        X:
            Training feature DataFrame.  Must not contain ``target`` or
            any leakage column.
        """
        if self.numeric_columns is None:
            self.numeric_columns = [
                c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
            ]
        if self.categorical_columns is None:
            self.categorical_columns = [
                c for c in X.columns if c not in (self.numeric_columns or [])
            ]

        # --- Numeric statistics ---
        for col in self.numeric_columns:
            if col not in X.columns:
                self._num_medians[col] = 0.0
                self._num_means[col] = 0.0
                self._num_stds[col] = 1.0
                continue
            arr = pd.to_numeric(X[col], errors="coerce").dropna().values.astype(float)
            if len(arr) == 0:
                self._num_medians[col] = 0.0
                self._num_means[col] = 0.0
                self._num_stds[col] = 1.0
            else:
                self._num_medians[col] = float(np.median(arr))
                self._num_means[col] = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                self._num_stds[col] = std if std > 0 else 1.0

        # --- Categorical categories ---
        for col in self.categorical_columns:
            if col not in X.columns:
                self._cat_categories[col] = []
                continue
            raw = X[col].fillna(_MISSING_SENTINEL).astype(str).str.strip()
            raw = raw.replace({"nan": _MISSING_SENTINEL, "None": _MISSING_SENTINEL, "<NA>": _MISSING_SENTINEL})
            cats = sorted(raw.unique().tolist())
            self._cat_categories[col] = cats

        # Build feature_names_out
        names: list[str] = []
        for col in (self.numeric_columns or []):
            names.append(col)
        for col in (self.categorical_columns or []):
            for cat in self._cat_categories.get(col, []):
                names.append(f"{col}__{cat}")
        self._feature_names_out = names
        self._fitted = True

        logger.info(
            "WidePreprocessor fitted: %d numeric, %d categorical -> %d output cols",
            len(self.numeric_columns or []),
            len(self.categorical_columns or []),
            len(self._feature_names_out),
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform *X* using the fitted preprocessor.

        Parameters
        ----------
        X:
            Feature DataFrame.  Must have columns compatible with the
            training data (extra columns are ignored).

        Returns
        -------
        np.ndarray
            Transformed float32 array of shape ``(n_samples, n_features_out)``.
        """
        if not self._fitted:
            raise RuntimeError(
                "WidePreprocessor has not been fitted. Call fit() or "
                "fit_transform() on training data first."
            )

        n = len(X)
        blocks: list[np.ndarray] = []

        # --- Numeric block ---
        if self.numeric_columns:
            num_block = np.empty((n, len(self.numeric_columns)), dtype=np.float64)
            for j, col in enumerate(self.numeric_columns):
                if col in X.columns:
                    col_arr = pd.to_numeric(X[col], errors="coerce").values.astype(np.float64)
                else:
                    col_arr = np.full(n, np.nan, dtype=np.float64)
                # Impute NaN with training median
                nan_mask = np.isnan(col_arr)
                col_arr[nan_mask] = self._num_medians.get(col, 0.0)
                # Scale
                col_arr = (col_arr - self._num_means.get(col, 0.0)) / self._num_stds.get(col, 1.0)
                num_block[:, j] = col_arr
            blocks.append(num_block)

        # --- Categorical block (one-hot) ---
        if self.categorical_columns:
            cat_blocks: list[np.ndarray] = []
            for col in self.categorical_columns:
                cats = self._cat_categories.get(col, [])
                if not cats:
                    continue
                n_cats = len(cats)
                ohe_block = np.zeros((n, n_cats), dtype=np.float32)
                if col in X.columns:
                    raw = X[col].fillna(_MISSING_SENTINEL).astype(str).str.strip()
                    raw = raw.replace({
                        "nan": _MISSING_SENTINEL,
                        "None": _MISSING_SENTINEL,
                        "<NA>": _MISSING_SENTINEL,
                    })
                    cat_to_idx = {c: i for i, c in enumerate(cats)}
                    for row_idx, val in enumerate(raw):
                        col_idx = cat_to_idx.get(val, None)
                        if col_idx is not None:
                            ohe_block[row_idx, col_idx] = 1.0
                        # else: unknown category → all zeros (same as sklearn OHE "ignore")
                cat_blocks.append(ohe_block)
            if cat_blocks:
                blocks.append(np.hstack(cat_blocks).astype(np.float64))

        if not blocks:
            return np.empty((n, 0), dtype=np.float32)

        result = np.hstack(blocks).astype(np.float32)
        return result

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit on *X* and return the transformed result."""
        self.fit(X)
        return self.transform(X)

    @property
    def feature_names_out(self) -> list[str]:
        """Output feature names after transformation."""
        if not self._fitted:
            raise RuntimeError("WidePreprocessor is not fitted.")
        return list(self._feature_names_out)

    @property
    def is_fitted(self) -> bool:
        """True after ``fit()`` has been called."""
        return self._fitted

    def save(self, path: str | Path) -> None:
        """Serialize the fitted preprocessor to *path* with pickle."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("WidePreprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "WidePreprocessor":
        """Load a previously saved preprocessor."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected WidePreprocessor, got {type(obj)}")
        return obj


# ---------------------------------------------------------------------------
# Sequence preprocessor
# ---------------------------------------------------------------------------

class SequencePreprocessor:
    """Fit-once, transform-many preprocessor for the 3-D sequence array.

    After the sequence builder's one-hot encoding, the array layout is::

        axis-2: [numeric_0, ..., numeric_N, ohe_cat_0_slot_0, ..., ohe_cat_K_slot_M]

    Numeric positions are imputed (median from training non-padded timesteps)
    and standard-scaled (mean/std from training non-padded timesteps).

    One-hot categorical positions are already binary {0, 1} and require no
    further processing — they are passed through unchanged.

    Padding positions (mask == 0) are zeroed after scaling so the GRU can
    safely ignore them.

    Parameters
    ----------
    numeric_indices:
        Axis-2 indices corresponding to numeric features.  Auto-detected
        from ``feature_names`` (any name in ``NUMERIC_FEATURES``) when
        ``None``.
    feature_names:
        Ordered list of feature names matching axis-2 of the sequence
        array.  Used to auto-detect numeric indices when
        ``numeric_indices`` is not provided.  Pass
        ``SequenceRepresentationResult.feature_names`` directly.
    """

    def __init__(
        self,
        numeric_indices: list[int] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        from dialong_automl.features.wide import NUMERIC_FEATURES as _NUM
        self._numeric_feature_names: list[str] = _NUM
        self.numeric_indices: list[int] | None = numeric_indices
        self.feature_names: list[str] | None = feature_names
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None
        self._medians: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        mask: np.ndarray,
    ) -> "SequencePreprocessor":
        """Fit using only real (non-padded) training timesteps.

        Parameters
        ----------
        X:
            Training sequences of shape ``(n_patients, max_seq_len, n_features)``.
        mask:
            Boolean/uint8 mask of shape ``(n_patients, max_seq_len)``.
            1 = real visit, 0 = padding.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D X, got shape {X.shape}")
        if mask.ndim != 2:
            raise ValueError(f"Expected 2-D mask, got shape {mask.shape}")

        n_patients, max_seq_len, n_features = X.shape

        # Determine numeric indices
        if self.numeric_indices is None:
            if self.feature_names is not None:
                self.numeric_indices = [
                    i for i, name in enumerate(self.feature_names)
                    if name in self._numeric_feature_names
                ]
            else:
                self.numeric_indices = list(range(n_features))

        # Flatten to 2-D, select only real (non-padding) timesteps
        flat_X = X.reshape(-1, n_features).astype(np.float64)
        flat_mask = mask.reshape(-1).astype(bool)
        real_X = flat_X[flat_mask]

        self._medians = np.zeros(n_features, dtype=np.float64)
        self._means = np.zeros(n_features, dtype=np.float64)
        self._stds = np.ones(n_features, dtype=np.float64)

        for idx in self.numeric_indices:
            col = real_X[:, idx]
            col_valid = col[~np.isnan(col)]
            if len(col_valid) > 0:
                self._medians[idx] = float(np.median(col_valid))
                self._means[idx] = float(np.mean(col_valid))
                std = float(np.std(col_valid, ddof=1)) if len(col_valid) > 1 else 0.0
                self._stds[idx] = std if std > 0 else 1.0

        self._fitted = True
        logger.info(
            "SequencePreprocessor fitted on %d real timesteps (%d numeric indices).",
            int(flat_mask.sum()), len(self.numeric_indices),
        )
        return self

    def transform(
        self,
        X: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Apply fitted preprocessing to *X*.

        Parameters
        ----------
        X:
            Sequence array of shape ``(n_patients, max_seq_len, n_features)``.
        mask:
            Mask of shape ``(n_patients, max_seq_len)``.

        Returns
        -------
        np.ndarray
            Preprocessed float32 array of same shape.
            Padding positions are zeroed.
        """
        if not self._fitted:
            raise RuntimeError(
                "SequencePreprocessor is not fitted. Call fit() first."
            )
        if self._means is None or self._medians is None or self._stds is None:
            raise RuntimeError("Internal state is corrupted.")

        out = X.copy().astype(np.float64)

        for idx in (self.numeric_indices or []):
            col = out[:, :, idx]
            nan_mask = np.isnan(col)
            col[nan_mask] = self._medians[idx]
            col -= self._means[idx]
            col /= self._stds[idx]
            out[:, :, idx] = col

        # Zero out padding positions
        pad_mask = mask == 0
        out[pad_mask, :] = 0.0

        return out.astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Fit on *X*/*mask* and return the transformed result."""
        self.fit(X, mask)
        return self.transform(X, mask)

    @property
    def is_fitted(self) -> bool:
        """True after ``fit()`` has been called."""
        return self._fitted

    def save(self, path: str | Path) -> None:
        """Serialize the fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("SequencePreprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SequencePreprocessor":
        """Load a previously saved preprocessor."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SequencePreprocessor, got {type(obj)}")
        return obj
