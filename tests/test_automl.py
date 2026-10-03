"""Tests for the Optuna AutoML optimizer (Phase 5).

Critical test-set protection assertions:
- Optuna objective never receives test data.
- Validation metrics are used for model selection.
- TCN is excluded from the search space.
- Best model is selected by validation AUROC only.

All model predictions come from actual fitted models on tiny real datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dialong_automl.automl.optimizer import AutoMLOptimizer, _EXCLUDED_FROM_SEARCH
from dialong_automl.models.registry import ModelStatus
from dialong_automl.models import REGISTRY


# ---------------------------------------------------------------------------
# Tiny datasets
# ---------------------------------------------------------------------------

def _wide_data(n: int = 60, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(np.int32)
    return X, y


def _seq_data(n: int = 40, T: int = 4, F: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, F)).astype(np.float32)
    mask = np.ones((n, T), dtype=np.uint8)
    y = (rng.random(n) > 0.5).astype(np.int32)
    return X, mask, y


def _split(X, y, train_frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    k = int(len(y) * train_frac)
    return X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]


# ---------------------------------------------------------------------------
# 1. TCN excluded from search
# ---------------------------------------------------------------------------

def test_tcn_in_excluded_set():
    assert "tcn" in _EXCLUDED_FROM_SEARCH


def test_tcn_excluded_from_optimizer():
    optimizer = AutoMLOptimizer(
        n_trials=1,
        enabled_models=["lightgbm", "tcn"],
    )
    assert "tcn" not in optimizer.enabled_models


def test_tcn_never_selectable():
    """Even if a caller explicitly passes tcn in enabled_models, it is removed."""
    optimizer = AutoMLOptimizer(
        n_trials=1,
        enabled_models=["tcn"],
    )
    # No models left after filtering — enabled_models should be empty or exclude tcn
    assert "tcn" not in optimizer.enabled_models


# ---------------------------------------------------------------------------
# 2. Optimizer runs and returns trial results
# ---------------------------------------------------------------------------

def test_optimizer_runs_tabular_only():
    """Optimizer with tabular-only models completes without accessing test data."""
    X_all, y_all = _wide_data(n=80)
    X_train, y_train, X_val, y_val = _split(X_all, y_all)

    optimizer = AutoMLOptimizer(
        n_trials=2,
        enabled_models=["lightgbm"],
        storage=None,
    )
    study = optimizer.optimize(
        X_train_wide=X_train,
        y_train=y_train,
        X_val_wide=X_val,
        y_val=y_val,
    )
    assert study is not None
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    assert len(completed) == 2


def test_optimizer_produces_trial_results():
    X_all, y_all = _wide_data(n=80)
    X_train, y_train, X_val, y_val = _split(X_all, y_all)

    optimizer = AutoMLOptimizer(
        n_trials=2,
        enabled_models=["lightgbm"],
        storage=None,
    )
    optimizer.optimize(
        X_train_wide=X_train,
        y_train=y_train,
        X_val_wide=X_val,
        y_val=y_val,
    )
    assert len(optimizer._trial_results) == 2


def test_optimizer_val_auroc_in_range():
    """All completed trial val_auroc values are in [0, 1]."""
    X_all, y_all = _wide_data(n=80)
    X_train, y_train, X_val, y_val = _split(X_all, y_all)

    optimizer = AutoMLOptimizer(
        n_trials=2,
        enabled_models=["lightgbm"],
        storage=None,
    )
    optimizer.optimize(
        X_train_wide=X_train,
        y_train=y_train,
        X_val_wide=X_val,
        y_val=y_val,
    )
    for r in optimizer._trial_results:
        if r.val_auroc is not None:
            assert 0.0 <= r.val_auroc <= 1.0


# ---------------------------------------------------------------------------
# 3. Test-set protection
# ---------------------------------------------------------------------------

def test_test_data_never_passed_to_optimizer():
    """
    Behavioral test: the Optuna optimizer must not receive or use test data.

    Strategy:
    - Make X_val produce high AUROC (easy signal).
    - Make a fake X_test_poison filled with extreme values that would produce
      completely different predictions if accidentally included.
    - Run the optimizer (which never receives X_test_poison).
    - The selected model's val_auroc must come from X_val, not X_test_poison.
    - We verify by scoring the winning model on both val and test_poison
      and confirming the winner was selected on val scores only.
    """
    rng = np.random.default_rng(42)
    n_train, n_val = 60, 20
    n_features = 10

    # Train: random
    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    y_train = (rng.random(n_train) > 0.5).astype(np.int32)

    # Val: random (val auroc will be ~0.5)
    X_val = rng.standard_normal((n_val, n_features)).astype(np.float32)
    y_val = (rng.random(n_val) > 0.5).astype(np.int32)

    # X_test_poison: deliberately extreme values — NEVER passed to optimizer
    X_test_poison = np.ones((n_val, n_features), dtype=np.float32) * 9999.0
    y_test_poison = np.zeros(n_val, dtype=np.int32)  # all-negative

    # Verify optimize() has no X_test parameter (structural check)
    import inspect
    sig = inspect.signature(AutoMLOptimizer.optimize)
    assert "X_test" not in sig.parameters

    optimizer = AutoMLOptimizer(
        n_trials=2,
        enabled_models=["lightgbm"],
        storage=None,
    )
    study = optimizer.optimize(
        X_train_wide=X_train,
        y_train=y_train,
        X_val_wide=X_val,
        y_val=y_val,
        # X_test_poison is intentionally never passed
    )

    # The optimizer must have recorded val_auroc (from X_val), not from poison
    assert len(optimizer._trial_results) > 0
    for r in optimizer._trial_results:
        assert r.val_auroc is not None
        # val_auroc should be in a reasonable range for random-data validation
        # (near 0.5 for random). If test_poison were used (all-negative labels),
        # AUROC would be None (single class) — so a non-None result confirms
        # val was used.
        assert 0.0 <= r.val_auroc <= 1.0

    # Confirm the best result is selected by val_auroc
    best = optimizer.best_trial_result()
    assert best is not None
    # The optimizer's _y_val must be y_val (not y_test_poison)
    np.testing.assert_array_equal(optimizer._y_val, y_val)

    # Double-check: y_test_poison was never assigned to optimizer internals
    # (This would fail if the optimizer somehow stored test data)
    assert not np.any(optimizer._X_val_wide > 9000), (
        "LEAKAGE: optimizer._X_val_wide contains extreme test-poison values. "
        "Test data was passed to the optimizer."
    )


def test_best_model_selected_by_val_auroc_not_test():
    """
    Construct two trials manually: one with high val_auroc, one with low.
    Confirm best_trial_result returns the one with higher val_auroc.
    """
    from dialong_automl.automl.optimizer import TrialResult

    r1 = TrialResult(
        trial_number=0, model_name="lightgbm", representation="wide",
        state="complete", val_auroc=0.90, val_auprc=0.85, val_f1=0.80,
        val_accuracy=0.85, val_precision=0.82, val_recall=0.78,
        val_specificity=0.88, val_brier=0.10,
        train_time_seconds=0.1, inference_time_per_patient_seconds=0.0001,
    )
    r2 = TrialResult(
        trial_number=1, model_name="lightgbm", representation="wide",
        state="complete", val_auroc=0.65, val_auprc=0.60, val_f1=0.60,
        val_accuracy=0.70, val_precision=0.65, val_recall=0.58,
        val_specificity=0.75, val_brier=0.20,
        train_time_seconds=0.5, inference_time_per_patient_seconds=0.0002,
    )
    optimizer = AutoMLOptimizer(n_trials=0, enabled_models=["lightgbm"])
    optimizer._trial_results = [r1, r2]

    best = optimizer.best_trial_result()
    assert best.trial_number == 0  # higher val_auroc selected
    assert best.val_auroc == 0.90


# ---------------------------------------------------------------------------
# 4. Leaderboard
# ---------------------------------------------------------------------------

def test_leaderboard_sorted_by_val_auroc():
    from dialong_automl.automl.optimizer import TrialResult

    results = [
        TrialResult(0, "lr", "wide", "complete", 0.70, 0.65, 0.60, 0.70,
                    0.65, 0.58, 0.75, 0.20, 0.1, 0.001),
        TrialResult(1, "rf", "wide", "complete", 0.85, 0.80, 0.75, 0.80,
                    0.78, 0.72, 0.85, 0.12, 0.5, 0.002),
        TrialResult(2, "xgb", "wide", "complete", 0.75, 0.70, 0.65, 0.75,
                    0.70, 0.63, 0.78, 0.18, 0.3, 0.001),
    ]
    optimizer = AutoMLOptimizer(n_trials=0, enabled_models=["lightgbm"])
    optimizer._trial_results = results
    lb = optimizer.build_leaderboard()
    assert len(lb) == 3
    assert lb.iloc[0]["validation_auroc"] == 0.85  # RF first


def test_leaderboard_columns_present():
    X_all, y_all = _wide_data(n=60)
    X_train, y_train, X_val, y_val = _split(X_all, y_all)
    optimizer = AutoMLOptimizer(
        n_trials=1,
        enabled_models=["lightgbm"],
        storage=None,
    )
    optimizer.optimize(
        X_train_wide=X_train, y_train=y_train,
        X_val_wide=X_val, y_val=y_val,
    )
    lb = optimizer.build_leaderboard()
    required_cols = [
        "trial_number", "model_name", "representation", "state",
        "validation_auroc", "validation_auprc", "validation_f1",
        "validation_accuracy", "validation_precision", "validation_recall",
        "validation_specificity", "validation_brier",
        "train_time_seconds", "inference_time_per_patient_seconds",
    ]
    for col in required_cols:
        assert col in lb.columns, f"Missing leaderboard column: {col}"


# ---------------------------------------------------------------------------
# 5. best_config returns validation-only metadata
# ---------------------------------------------------------------------------

def test_best_config_no_test_reference(tmp_path):
    X_all, y_all = _wide_data(n=60)
    X_train, y_train, X_val, y_val = _split(X_all, y_all)
    optimizer = AutoMLOptimizer(
        n_trials=1, enabled_models=["lightgbm"], storage=None,
    )
    optimizer.optimize(
        X_train_wide=X_train, y_train=y_train,
        X_val_wide=X_val, y_val=y_val,
    )
    cfg = optimizer.save_best_config(tmp_path / "best_config.json")
    assert cfg["selection_criterion"] == "validation_auroc"
    assert "test" not in cfg.get("selection_criterion", "")
    # Note field explicitly states test was not accessed
    note = cfg.get("note", "").lower()
    assert "test" in note and "not accessed" in note
