"""Test 4 — Default seed handling.

Verifies that :func:`~dialong_automl.utils.seed.set_global_seed`:
- Accepts and returns the seed value
- Defaults to 42 when called with no arguments
- Works correctly when numpy is available
- Does not raise when torch is not available (graceful degradation)
"""

from __future__ import annotations

import random

import pytest

from dialong_automl.utils.seed import set_global_seed, _DEFAULT_SEED


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_default_seed_constant() -> None:
    """_DEFAULT_SEED must equal 42."""
    assert _DEFAULT_SEED == 42


def test_set_global_seed_returns_seed() -> None:
    """set_global_seed returns the applied seed."""
    result = set_global_seed(42)
    assert result == 42


def test_set_global_seed_default() -> None:
    """Calling set_global_seed() with no arguments uses seed 42."""
    result = set_global_seed()
    assert result == _DEFAULT_SEED


def test_set_global_seed_custom() -> None:
    """set_global_seed accepts a custom seed and returns it."""
    result = set_global_seed(123)
    assert result == 123


def test_set_global_seed_zero() -> None:
    """Seed 0 is a valid seed (boundary value)."""
    result = set_global_seed(0)
    assert result == 0


# ---------------------------------------------------------------------------
# Python random determinism
# ---------------------------------------------------------------------------


def test_python_random_deterministic() -> None:
    """The same seed produces the same sequence from Python's random module."""
    set_global_seed(42)
    seq_a = [random.random() for _ in range(10)]

    set_global_seed(42)
    seq_b = [random.random() for _ in range(10)]

    assert seq_a == seq_b


def test_different_seeds_different_sequences() -> None:
    """Different seeds produce different sequences (probabilistic sanity check)."""
    set_global_seed(42)
    seq_a = [random.random() for _ in range(10)]

    set_global_seed(99)
    seq_b = [random.random() for _ in range(10)]

    assert seq_a != seq_b


# ---------------------------------------------------------------------------
# NumPy determinism (if available)
# ---------------------------------------------------------------------------


def test_numpy_seed_deterministic() -> None:
    """The same seed produces the same NumPy random array."""
    np = pytest.importorskip("numpy")

    set_global_seed(42)
    arr_a = np.random.rand(5)

    set_global_seed(42)
    arr_b = np.random.rand(5)

    assert list(arr_a) == list(arr_b)


# ---------------------------------------------------------------------------
# Config-driven seed
# ---------------------------------------------------------------------------


def test_seed_from_config() -> None:
    """set_global_seed reads correctly from a loaded config."""
    from dialong_automl.config import load_config

    cfg = load_config(None)
    result = set_global_seed(cfg.reproducibility.seed)
    assert result == cfg.reproducibility.seed
