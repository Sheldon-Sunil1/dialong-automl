"""Tests for Phase 5 model wrappers.

Each model wrapper is tested for correct API behaviour.  When a model
dependency is blocked by OS Application Control policy (e.g. sklearn's
compiled extensions, torch DLLs), the affected tests are skipped with a
clear reason rather than failing with an unrelated error.

This ensures the test suite reports accurate results on machines with and
without OS restrictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from dialong_automl.models import (
    REGISTRY,
    build_model,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    GRUModel,
)
from dialong_automl.models.base import REPRESENTATION_WIDE, REPRESENTATION_SEQUENCE
from dialong_automl.models.registry import ModelStatus


# ---------------------------------------------------------------------------
# Availability checks — probe each dependency once at collection time
# ---------------------------------------------------------------------------

def _sklearn_available() -> bool:
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        # Perform a real tiny fit to confirm the full execution path works
        import numpy as _np
        _X = _np.zeros((6, 3), dtype="float32")
        _y = _np.array([0, 0, 0, 1, 1, 1], dtype="int32")
        LogisticRegression(max_iter=50).fit(_X, _y)
        return True
    except (ImportError, OSError):
        return False

def _sklearn_ensemble_available() -> bool:
    """RandomForest uses sklearn.ensemble which pulls different DLLs."""
    try:
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
        import numpy as _np
        _X = _np.zeros((6, 3), dtype="float32")
        _y = _np.array([0, 0, 0, 1, 1, 1], dtype="int32")
        RandomForestClassifier(n_estimators=2, random_state=0).fit(_X, _y)
        return True
    except (ImportError, OSError):
        return False

def _xgboost_available() -> bool:
    try:
        from xgboost import XGBClassifier  # noqa: F401
        import numpy as _np
        _X = _np.zeros((6, 3), dtype="float32")
        _y = _np.array([0, 0, 0, 1, 1, 1], dtype="int32")
        XGBClassifier(n_estimators=2, verbosity=0).fit(_X, _y)
        return True
    except (ImportError, OSError):
        return False

def _xgboost_available() -> bool:
    try:
        from xgboost import XGBClassifier  # noqa: F401
        return True
    except (ImportError, OSError):
        return False

def _torch_available() -> bool:
    try:
        import torch
        # Verify a basic tensor operation succeeds (DLL fully loaded)
        t = torch.tensor([1.0])
        _ = t + 1
        return True
    except (ImportError, OSError):
        return False

_SKLEARN_OK       = _sklearn_available()
_SKLEARN_ENSEM_OK = _sklearn_ensemble_available()
_XGBOOST_OK       = _xgboost_available()
_TORCH_OK         = _torch_available()

_SKIP_SKLEARN       = pytest.mark.skipif(not _SKLEARN_OK, reason="sklearn blocked by OS policy")
_SKIP_SKLEARN_ENSEM = pytest.mark.skipif(not _SKLEARN_ENSEM_OK, reason="sklearn.ensemble blocked by OS policy")
_SKIP_XGBOOST       = pytest.mark.skipif(not _XGBOOST_OK, reason="xgboost blocked by OS policy")
_SKIP_TORCH         = pytest.mark.skipif(not _TORCH_OK,   reason="torch blocked by OS policy")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _tabular_data(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10)).astype(np.float32)
    y = (rng.random(n) > 0.6).astype(np.int32)
    return X, y


def _seq_data(n: int = 40, T: int = 4, F: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, F)).astype(np.float32)
    mask = np.ones((n, T), dtype=np.uint8)
    mask[:5, 3] = 0
    mask[:2, 2] = 0
    y = (rng.random(n) > 0.5).astype(np.int32)
    return X, mask, y


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class TestLogisticRegression:
    @_SKIP_SKLEARN
    def test_fit_predict_proba(self):
        X, y = _tabular_data()
        m = LogisticRegressionModel({"C": 1.0, "max_iter": 200})
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (60,)
        assert (proba >= 0).all() and (proba <= 1).all()

    @_SKIP_SKLEARN
    def test_predict_binary(self):
        X, y = _tabular_data()
        m = LogisticRegressionModel()
        m.fit(X, y)
        preds = m.predict(X)
        assert set(preds).issubset({0, 1})

    def test_representation_is_wide(self):
        assert LogisticRegressionModel.representation == REPRESENTATION_WIDE

    def test_not_fitted_raises(self):
        m = LogisticRegressionModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict_proba(np.zeros((5, 10)))

    @_SKIP_SKLEARN
    def test_reproducible_with_seed(self):
        X, y = _tabular_data()
        m1 = LogisticRegressionModel({"C": 1.0, "random_state": 42, "max_iter": 200})
        m2 = LogisticRegressionModel({"C": 1.0, "random_state": 42, "max_iter": 200})
        m1.fit(X, y)
        m2.fit(X, y)
        np.testing.assert_array_almost_equal(
            m1.predict_proba(X), m2.predict_proba(X), decimal=5
        )

    def test_fit_raises_importerror_when_sklearn_blocked(self):
        """If sklearn is unavailable, fit() raises ImportError (not crash)."""
        if _SKLEARN_OK:
            pytest.skip("sklearn is available — testing blocked path is not applicable")
        m = LogisticRegressionModel()
        with pytest.raises(ImportError, match="sklearn"):
            m.fit(np.zeros((10, 5), dtype=np.float32), np.zeros(10, dtype=np.int32))


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class TestRandomForest:
    @_SKIP_SKLEARN_ENSEM
    def test_fit_predict_proba(self):
        X, y = _tabular_data()
        m = RandomForestModel({"n_estimators": 20})
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (60,)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_representation_is_wide(self):
        assert RandomForestModel.representation == REPRESENTATION_WIDE

    def test_not_fitted_raises(self):
        m = RandomForestModel()
        with pytest.raises(RuntimeError):
            m.predict_proba(np.zeros((3, 10)))


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class TestXGBoost:
    @_SKIP_XGBOOST
    def test_fit_predict_proba(self):
        X, y = _tabular_data()
        m = XGBoostModel({"n_estimators": 20, "max_depth": 3})
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (60,)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_representation_is_wide(self):
        assert XGBoostModel.representation == REPRESENTATION_WIDE

    @_SKIP_XGBOOST
    def test_handles_nan_in_features(self):
        X, y = _tabular_data()
        X_nan = X.copy()
        X_nan[0, 0] = float("nan")
        m = XGBoostModel({"n_estimators": 10})
        m.fit(X_nan, y)
        proba = m.predict_proba(X_nan)
        assert not np.any(np.isnan(proba))

    def test_fit_raises_importerror_when_blocked(self):
        """If xgboost is unavailable, fit() raises ImportError (not crash)."""
        if _XGBOOST_OK:
            pytest.skip("xgboost is available")
        m = XGBoostModel()
        with pytest.raises(ImportError, match="xgboost"):
            m.fit(np.zeros((10, 5), dtype=np.float32), np.zeros(10, dtype=np.int32))


# ---------------------------------------------------------------------------
# LightGBM (always works on this machine)
# ---------------------------------------------------------------------------

class TestLightGBM:
    def test_fit_predict_proba(self):
        X, y = _tabular_data()
        m = LightGBMModel({"n_estimators": 20})
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (60,)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_representation_is_wide(self):
        assert LightGBMModel.representation == REPRESENTATION_WIDE

    def test_predict_binary(self):
        X, y = _tabular_data()
        m = LightGBMModel({"n_estimators": 10})
        m.fit(X, y)
        preds = m.predict(X)
        assert set(preds).issubset({0, 1})

    def test_not_fitted_raises(self):
        m = LightGBMModel()
        with pytest.raises(RuntimeError):
            m.predict_proba(np.zeros((3, 10)))

    def test_reproducible_same_seed(self):
        X, y = _tabular_data()
        m1 = LightGBMModel({"n_estimators": 20, "random_state": 42})
        m2 = LightGBMModel({"n_estimators": 20, "random_state": 42})
        m1.fit(X, y)
        m2.fit(X, y)
        np.testing.assert_array_almost_equal(
            m1.predict_proba(X), m2.predict_proba(X), decimal=5
        )


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class TestGRU:
    def test_representation_is_sequence(self):
        assert GRUModel.representation == REPRESENTATION_SEQUENCE

    def test_init_raises_importerror_when_torch_blocked(self):
        """If torch is unavailable, GRUModel.__init__ raises ImportError."""
        if _TORCH_OK:
            pytest.skip("torch is available")
        with pytest.raises(ImportError, match="torch"):
            GRUModel()

    @_SKIP_TORCH
    def test_fit_predict_proba(self):
        X, mask, y = _seq_data(n=30)
        m = GRUModel({
            "hidden_size": 16, "num_layers": 1, "dropout": 0.0,
            "learning_rate": 1e-3, "batch_size": 16,
            "epochs": 3, "patience": 2, "seed": 42,
        })
        m.fit(X, y, mask=mask)
        proba = m.predict_proba_with_mask(X, mask)
        assert proba.shape == (30,)
        assert (proba >= 0).all() and (proba <= 1).all()

    @_SKIP_TORCH
    def test_early_stopping_with_val(self):
        X, mask, y = _seq_data(n=40)
        m = GRUModel({
            "hidden_size": 8, "num_layers": 1, "dropout": 0.0,
            "learning_rate": 1e-3, "batch_size": 16,
            "epochs": 10, "patience": 2, "seed": 7,
        })
        m.fit(X[:30], y[:30], mask=mask[:30],
              X_val=X[30:], y_val=y[30:], mask_val=mask[30:])
        assert m.is_fitted

    @_SKIP_TORCH
    def test_not_fitted_raises(self):
        m = GRUModel()
        with pytest.raises(RuntimeError):
            m.predict_proba(np.zeros((3, 4, 16)))

    @_SKIP_TORCH
    def test_reproducible_same_seed(self):
        X, mask, y = _seq_data(n=24)
        hp = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0,
              "learning_rate": 1e-3, "batch_size": 8, "epochs": 3,
              "patience": 10, "seed": 42}
        m1 = GRUModel(hp)
        m2 = GRUModel(hp)
        m1.fit(X, y)
        m2.fit(X, y)
        p1 = m1.predict_proba(X)
        p2 = m2.predict_proba(X)
        np.testing.assert_array_almost_equal(p1, p2, decimal=4)


# ---------------------------------------------------------------------------
# Registry + TCN
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_five_models_available(self):
        for name in ["logistic_regression", "random_forest", "xgboost", "lightgbm", "gru"]:
            entry = REGISTRY.get(name)
            assert entry.status == ModelStatus.AVAILABLE, f"{name} not AVAILABLE"

    def test_tcn_planned_not_implemented(self):
        entry = REGISTRY.get("tcn")
        assert entry.status == ModelStatus.PLANNED_NOT_IMPLEMENTED

    def test_tcn_raises_not_implemented_error(self):
        with pytest.raises(NotImplementedError):
            REGISTRY.get_factory("tcn")

    def test_tcn_not_in_list_available(self):
        assert "tcn" not in REGISTRY.list_available()

    def test_build_model_lr_dispatches_correctly(self):
        """build_model dispatches to LogisticRegressionModel."""
        m = build_model("logistic_regression", {"max_iter": 100})
        assert isinstance(m, LogisticRegressionModel)

    def test_build_model_lgbm_dispatches_correctly(self):
        """build_model dispatches to LightGBMModel (always available)."""
        m = build_model("lightgbm", {"n_estimators": 10})
        assert isinstance(m, LightGBMModel)

    def test_build_model_gru_raises_importerror_when_blocked(self):
        """When torch is blocked, build_model('gru') raises ImportError."""
        if _TORCH_OK:
            pytest.skip("torch is available — testing blocked path not applicable")
        with pytest.raises(ImportError, match="torch"):
            build_model("gru")

    @_SKIP_TORCH
    def test_build_model_gru_dispatches_when_torch_available(self):
        m = build_model("gru", {"epochs": 1})
        assert isinstance(m, GRUModel)

    def test_build_model_tcn_raises(self):
        with pytest.raises(NotImplementedError):
            build_model("tcn")

    def test_tcn_excluded_from_available_search(self):
        available = REGISTRY.list_available()
        assert "tcn" not in available
        for name in ["logistic_regression", "random_forest", "xgboost", "lightgbm", "gru"]:
            assert name in available
