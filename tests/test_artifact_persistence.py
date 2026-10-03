"""Tests for model and preprocessor persistence (Phase 5).

Covers:
- Every model can save and reload.
- Reloaded model produces identical predictions.
- WidePreprocessor saves/loads with identical transform output.
- SequencePreprocessor saves/loads with identical transform output.
- GRU saves to .pt and loads correctly.
- Tabular models save to .joblib-compatible pickle.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tabular(n: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10)).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(np.int32)
    return X, y


def _seq(n: int = 30, T: int = 4, F: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, F)).astype(np.float32)
    mask = np.ones((n, T), dtype=np.uint8)
    y = (rng.random(n) > 0.5).astype(np.int32)
    return X, mask, y


# ---------------------------------------------------------------------------
# Tabular model persistence
# ---------------------------------------------------------------------------

class TestTabularPersistence:
    @pytest.mark.parametrize("model_name,cls_name", [
        ("logistic_regression", "LogisticRegressionModel"),
        ("random_forest", "RandomForestModel"),
        ("xgboost", "XGBoostModel"),
        ("lightgbm", "LightGBMModel"),
    ])
    def test_save_load_identical_predictions(self, tmp_path, model_name, cls_name):
        from dialong_automl.models import build_model

        # Check if the underlying dependency is available before attempting fit
        _deps = {
            "logistic_regression": "sklearn.linear_model",
            "random_forest": "sklearn.ensemble",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
        }
        dep = _deps.get(model_name)
        if dep:
            try:
                __import__(dep)
            except (ImportError, OSError):
                pytest.skip(f"{dep} blocked by OS policy")

        X, y = _tabular()
        model = build_model(model_name, {"n_estimators": 10, "max_iter": 100})
        model.fit(X, y)
        proba_before = model.predict_proba(X)

        path = tmp_path / f"{model_name}.joblib"
        model.save(path)
        assert path.exists() and path.stat().st_size > 0

        cls = model.__class__
        loaded = cls.load(path)
        assert loaded.is_fitted
        proba_after = loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(proba_before, proba_after, decimal=5)

    def test_load_wrong_type_raises(self, tmp_path):
        """Loading a file saved by a different model class raises TypeError."""
        import pickle
        from dialong_automl.models.random_forest import RandomForestModel
        from dialong_automl.models.lightgbm_model import LightGBMModel

        # Manually write a pickle containing a LightGBMModel stub
        # (not a real fitted model — just a placeholder to verify type check)
        stub = object.__new__(LightGBMModel)
        stub.hyperparams = {}
        stub._fitted = False
        stub._clf = None

        path = tmp_path / "stub.joblib"
        with open(path, "wb") as fh:
            pickle.dump(stub, fh)

        # RandomForestModel.load should reject a LightGBMModel object
        with pytest.raises(TypeError):
            RandomForestModel.load(path)

    def test_save_unfitted_raises(self, tmp_path):
        from dialong_automl.models import build_model
        m = build_model("lightgbm")
        with pytest.raises(RuntimeError, match="not fitted"):
            m.save(tmp_path / "lgbm.joblib")


# ---------------------------------------------------------------------------
# GRU persistence
# ---------------------------------------------------------------------------

class TestGRUPersistence:
    def _torch_ok(self):
        try:
            import torch  # noqa: F401
            return True
        except (ImportError, OSError):
            return False

    def test_gru_save_load_pt(self, tmp_path):
        if not self._torch_ok():
            pytest.skip("torch blocked by OS policy")
        from dialong_automl.models.gru import GRUModel

        X, mask, y = _seq()
        m = GRUModel({
            "hidden_size": 8, "num_layers": 1, "dropout": 0.0,
            "learning_rate": 1e-3, "batch_size": 16,
            "epochs": 2, "patience": 10, "seed": 42,
        })
        m.fit(X, y, mask=mask)
        proba_before = m.predict_proba_with_mask(X, mask)

        path = tmp_path / "gru.pt"
        m.save(path)
        assert path.exists() and path.stat().st_size > 0
        assert path.suffix == ".pt"

        loaded = GRUModel.load(path)
        assert loaded.is_fitted
        proba_after = loaded.predict_proba_with_mask(X, mask)
        np.testing.assert_array_almost_equal(proba_before, proba_after, decimal=4)

    def test_gru_init_raises_when_torch_blocked(self, tmp_path):
        if self._torch_ok():
            pytest.skip("torch is available")
        from dialong_automl.models.gru import GRUModel
        with pytest.raises(ImportError, match="torch"):
            GRUModel()

    def test_gru_save_unfitted_raises(self, tmp_path):
        if not self._torch_ok():
            pytest.skip("torch blocked by OS policy")
        from dialong_automl.models.gru import GRUModel
        m = GRUModel()
        with pytest.raises(RuntimeError):
            m.save(tmp_path / "gru.pt")


# ---------------------------------------------------------------------------
# WidePreprocessor persistence
# ---------------------------------------------------------------------------

class TestWidePreprocessorPersistence:
    def test_save_load_identical_transform(self, tmp_path):
        import pandas as pd
        from dialong_automl.features.preprocessing import WidePreprocessor

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a": rng.standard_normal(40),
            "b": rng.standard_normal(40),
            "c": rng.choice(["x", "y", "z"], 40),
        })
        prep = WidePreprocessor(numeric_columns=["a", "b"], categorical_columns=["c"])
        out_before = prep.fit_transform(df)

        path = tmp_path / "wide_prep.joblib"
        prep.save(path)
        assert path.exists()

        loaded = WidePreprocessor.load(path)
        assert loaded.is_fitted
        out_after = loaded.transform(df)
        np.testing.assert_array_almost_equal(out_before, out_after, decimal=6)

    def test_feature_names_out_preserved(self, tmp_path):
        import pandas as pd
        from dialong_automl.features.preprocessing import WidePreprocessor

        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.standard_normal(20), "y": rng.standard_normal(20)})
        prep = WidePreprocessor(numeric_columns=["x", "y"], categorical_columns=[])
        prep.fit(df)
        names_before = prep.feature_names_out

        prep.save(tmp_path / "p.joblib")
        loaded = WidePreprocessor.load(tmp_path / "p.joblib")
        assert loaded.feature_names_out == names_before

    def test_save_unfitted_raises(self, tmp_path):
        from dialong_automl.features.preprocessing import WidePreprocessor
        prep = WidePreprocessor()
        with pytest.raises(RuntimeError):
            prep.save(tmp_path / "p.joblib")


# ---------------------------------------------------------------------------
# SequencePreprocessor persistence
# ---------------------------------------------------------------------------

class TestSequencePreprocessorPersistence:
    def test_save_load_identical_transform(self, tmp_path):
        from dialong_automl.features.preprocessing import SequencePreprocessor

        rng = np.random.default_rng(2)
        X = rng.standard_normal((20, 4, 6)).astype(np.float32)
        mask = np.ones((20, 4), dtype=np.uint8)

        prep = SequencePreprocessor(numeric_indices=[0, 1, 2, 3, 4, 5])
        out_before = prep.fit_transform(X, mask)

        path = tmp_path / "seq_prep.joblib"
        prep.save(path)
        assert path.exists()

        loaded = SequencePreprocessor.load(path)
        assert loaded.is_fitted
        out_after = loaded.transform(X, mask)
        np.testing.assert_array_almost_equal(out_before, out_after, decimal=6)

    def test_seq_prep_save_unfitted_raises(self, tmp_path):
        from dialong_automl.features.preprocessing import SequencePreprocessor
        prep = SequencePreprocessor()
        with pytest.raises(RuntimeError):
            prep.save(tmp_path / "seq.joblib")
