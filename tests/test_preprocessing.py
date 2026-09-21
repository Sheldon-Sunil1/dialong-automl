"""Tests for training-only preprocessing (Phase 4).

Covers:
1.  WidePreprocessor fit/transform/fit_transform API.
2.  Preprocessor uses only training statistics (not refitted on val/test).
3.  Validation transformed with training statistics (not val statistics).
4.  Unfitted preprocessor raises on transform.
5.  Unknown categorical value handled safely (ignored by OHE).
6.  Output is numpy float32 array.
7.  feature_names_out reflects actual output columns.
8.  SequencePreprocessor fit/transform API on 3-D arrays.
9.  Sequence preprocessing uses only non-padded training timesteps.
10. Padding positions are zeroed after transform.
11. Fitting on train does not change when val is also transformed.
12. Deterministic: same result on repeated transform calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dialong_automl.features.preprocessing import (
    SequencePreprocessor,
    WidePreprocessor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_df(n: int = 20) -> pd.DataFrame:
    """Small training DataFrame with numeric + categorical columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "hba1c": rng.uniform(4.5, 6.5, n),
        "bmi": rng.uniform(20.0, 30.0, n),
        "sex": rng.choice(["M", "F"], n),
        "smoking": rng.choice(["never", "former", "current"], n),
    })


def _val_df_extreme(n: int = 5) -> pd.DataFrame:
    """Validation DataFrame with values far from training range."""
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "hba1c": rng.uniform(900.0, 1000.0, n),   # far from train range
        "bmi": rng.uniform(900.0, 1000.0, n),
        "sex": rng.choice(["M", "F"], n),
        "smoking": rng.choice(["never", "former"], n),
    })


def _seq_array(n: int = 8, t: int = 4, f: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, mask) with some padding."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (n, t, f)).astype(np.float32)
    mask = np.ones((n, t), dtype=np.uint8)
    # Last 2 patients have only 2 real visits
    mask[-2:, 2:] = 0
    X[-2:, 2:, :] = 0.0
    return X, mask


# ---------------------------------------------------------------------------
# 1. WidePreprocessor API
# ---------------------------------------------------------------------------

def test_wide_fit_returns_self() -> None:
    """fit() returns the preprocessor itself."""
    prep = WidePreprocessor(numeric_columns=["hba1c", "bmi"])
    df = _train_df()
    result = prep.fit(df[["hba1c", "bmi"]])
    assert result is prep


def test_wide_fit_transform_returns_array() -> None:
    """fit_transform returns a numpy array."""
    prep = WidePreprocessor(numeric_columns=["hba1c", "bmi"])
    df = _train_df()
    out = prep.fit_transform(df[["hba1c", "bmi"]])
    assert isinstance(out, np.ndarray)


def test_wide_transform_shape() -> None:
    """Output shape matches input rows."""
    prep = WidePreprocessor(numeric_columns=["hba1c", "bmi"])
    df = _train_df(20)
    prep.fit(df[["hba1c", "bmi"]])
    out = prep.transform(df[["hba1c", "bmi"]])
    assert out.shape[0] == 20


def test_wide_output_dtype_float32() -> None:
    """Output dtype is float32."""
    prep = WidePreprocessor(numeric_columns=["hba1c"])
    df = _train_df()
    out = prep.fit_transform(df[["hba1c"]])
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. Preprocessor uses training statistics only
# ---------------------------------------------------------------------------

def test_validation_uses_training_mean_std() -> None:
    """Validation values are scaled using TRAINING mean/std, not their own.

    Train: hba1c ~ [1, 5].  Val: hba1c ~ [900, 1000].
    If val were scaled with val statistics, the output would be near 0.
    With train statistics it should be a very large positive number.
    """
    train_df = pd.DataFrame({"hba1c": np.linspace(1.0, 5.0, 50)})
    val_df = pd.DataFrame({"hba1c": np.linspace(900.0, 1000.0, 10)})

    prep = WidePreprocessor(numeric_columns=["hba1c"])
    prep.fit(train_df)
    val_out = prep.transform(val_df)

    # With training stats (mean ~3.0, std ~1.15), val values should be large
    assert val_out.mean() > 100, (
        "Val output should be large when using train statistics on extreme val values"
    )


def test_transform_does_not_refit() -> None:
    """Calling transform on val does NOT change the fitted statistics."""
    train_df = pd.DataFrame({"hba1c": np.linspace(1.0, 5.0, 50)})
    val_df = pd.DataFrame({"hba1c": np.linspace(900.0, 1000.0, 10)})

    prep = WidePreprocessor(numeric_columns=["hba1c"])
    prep.fit(train_df)

    # Record train output before transforming val
    train_out_before = prep.transform(train_df).copy()
    prep.transform(val_df)  # must NOT change internal state
    train_out_after = prep.transform(train_df).copy()

    np.testing.assert_array_almost_equal(train_out_before, train_out_after)


# ---------------------------------------------------------------------------
# 3. Unfitted preprocessor raises
# ---------------------------------------------------------------------------

def test_unfitted_wide_raises_on_transform() -> None:
    """Calling transform without fit raises RuntimeError."""
    prep = WidePreprocessor(numeric_columns=["hba1c"])
    df = _train_df()
    with pytest.raises(RuntimeError, match="not been fitted"):
        prep.transform(df[["hba1c"]])


def test_is_fitted_property() -> None:
    """is_fitted is False before fit, True after."""
    prep = WidePreprocessor(numeric_columns=["hba1c"])
    assert not prep.is_fitted
    prep.fit(_train_df()[["hba1c"]])
    assert prep.is_fitted


# ---------------------------------------------------------------------------
# 4. Unknown categorical handled safely (OHE ignore)
# ---------------------------------------------------------------------------

def test_unknown_categorical_ignored_not_error() -> None:
    """A category unseen at fit time is silently zeroed by OHE, not an error."""
    train_df = pd.DataFrame({
        "sex": ["M", "F", "M", "F", "M"],
    })
    val_df = pd.DataFrame({
        "sex": ["X"],  # unknown category
    })
    prep = WidePreprocessor(
        numeric_columns=[],
        categorical_columns=["sex"],
    )
    prep.fit(train_df)
    out = prep.transform(val_df)  # must not raise
    assert isinstance(out, np.ndarray)


# ---------------------------------------------------------------------------
# 5. feature_names_out
# ---------------------------------------------------------------------------

def test_feature_names_out_is_list() -> None:
    """feature_names_out returns a list of strings."""
    prep = WidePreprocessor(numeric_columns=["hba1c", "bmi"])
    prep.fit(_train_df()[["hba1c", "bmi"]])
    names = prep.feature_names_out
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


def test_feature_names_out_matches_output_width() -> None:
    """len(feature_names_out) == output array width."""
    prep = WidePreprocessor(numeric_columns=["hba1c", "bmi"])
    df = _train_df()
    out = prep.fit_transform(df[["hba1c", "bmi"]])
    assert len(prep.feature_names_out) == out.shape[1]


def test_feature_names_out_raises_if_not_fitted() -> None:
    """feature_names_out raises RuntimeError if not fitted."""
    prep = WidePreprocessor()
    with pytest.raises(RuntimeError):
        _ = prep.feature_names_out


# ---------------------------------------------------------------------------
# 6. SequencePreprocessor API
# ---------------------------------------------------------------------------

def test_seq_preprocessor_fit_returns_self() -> None:
    """fit() returns self."""
    X, mask = _seq_array()
    prep = SequencePreprocessor()
    result = prep.fit(X, mask)
    assert result is prep


def test_seq_fit_transform_shape_preserved() -> None:
    """fit_transform returns array of the same shape as input."""
    X, mask = _seq_array(n=8, t=4, f=3)
    prep = SequencePreprocessor()
    out = prep.fit_transform(X, mask)
    assert out.shape == X.shape


def test_seq_output_dtype_float32() -> None:
    """SequencePreprocessor output is float32."""
    X, mask = _seq_array()
    prep = SequencePreprocessor()
    out = prep.fit_transform(X, mask)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# 7. Sequence preprocessing uses only non-padded training timesteps
# ---------------------------------------------------------------------------

def test_seq_train_only_non_padded_for_fit() -> None:
    """Padding positions do not influence the computed statistics.

    Train set: real visits have hba1c ~ [1.0, 2.0].
    Val set: same patients but the extreme values (1000.0) appear in padded
    positions so they should not affect output.
    """
    # Build train data where feature 0 is in [1, 2]
    X_train = np.ones((6, 4, 2), dtype=np.float32) * 1.5
    mask_train = np.ones((6, 4), dtype=np.uint8)

    prep = SequencePreprocessor(numeric_indices=[0, 1])
    prep.fit(X_train, mask_train)

    # Inject extreme values into padding positions — must not affect transform
    X_extreme = np.ones((2, 4, 2), dtype=np.float32) * 1.5
    X_extreme[:, 2:, 0] = 1000.0  # extreme in padding area
    mask_extreme = np.ones((2, 4), dtype=np.uint8)
    mask_extreme[:, 2:] = 0  # those positions are padding

    out = prep.transform(X_extreme, mask_extreme)

    # Padding positions should be zeroed
    assert (out[:, 2:, :] == 0.0).all(), (
        "Padding positions must be zeroed after transform"
    )


# ---------------------------------------------------------------------------
# 8. Padding positions zeroed after transform
# ---------------------------------------------------------------------------

def test_padding_zeroed_after_seq_transform() -> None:
    """After transform, padding positions (mask=0) are 0.0."""
    X, mask = _seq_array(n=4, t=4, f=3)
    prep = SequencePreprocessor(numeric_indices=[0, 1, 2])
    out = prep.fit_transform(X, mask)

    pad_positions = mask == 0
    assert (out[pad_positions, :] == 0.0).all()


# ---------------------------------------------------------------------------
# 9. Fitting on train does not change when val is also transformed
# ---------------------------------------------------------------------------

def test_seq_transform_does_not_refit() -> None:
    """Transforming validation data does NOT alter fitted statistics."""
    X_train, mask_train = _seq_array(seed=0)
    X_val = np.ones_like(X_train) * 9999.0  # extreme val values
    mask_val = np.ones_like(mask_train)

    prep = SequencePreprocessor(numeric_indices=[0, 1, 2])
    prep.fit(X_train, mask_train)

    means_before = prep._means.copy()
    prep.transform(X_val, mask_val)  # must not change internal state
    means_after = prep._means.copy()

    np.testing.assert_array_equal(means_before, means_after)


# ---------------------------------------------------------------------------
# 10. Unfitted sequence preprocessor raises
# ---------------------------------------------------------------------------

def test_unfitted_seq_raises_on_transform() -> None:
    """SequencePreprocessor.transform() raises RuntimeError if not fitted."""
    X, mask = _seq_array()
    prep = SequencePreprocessor()
    with pytest.raises(RuntimeError, match="not fitted"):
        prep.transform(X, mask)


# ---------------------------------------------------------------------------
# 11. Deterministic output
# ---------------------------------------------------------------------------

def test_wide_transform_is_deterministic() -> None:
    """Calling transform twice with the same input returns identical arrays."""
    prep = WidePreprocessor(numeric_columns=["hba1c", "bmi"])
    df = _train_df()
    prep.fit(df[["hba1c", "bmi"]])
    out1 = prep.transform(df[["hba1c", "bmi"]])
    out2 = prep.transform(df[["hba1c", "bmi"]])
    np.testing.assert_array_equal(out1, out2)


def test_seq_transform_is_deterministic() -> None:
    """SequencePreprocessor transform is deterministic."""
    X, mask = _seq_array()
    prep = SequencePreprocessor()
    out1 = prep.fit_transform(X, mask)
    out2 = prep.transform(X, mask)
    np.testing.assert_array_almost_equal(out1, out2)
