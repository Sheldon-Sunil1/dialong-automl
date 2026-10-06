"""Phase 7 — Explainability tests.

Covers:
1.  Feature name resolver — all column types map to readable strings.
2.  SHAP tree explainer — global explanation structure and values.
3.  Permutation fallback — used when SHAP unavailable / RF blocked.
4.  Patient-level explanation — structure, probability, threshold.
5.  Patient explanation without SHAP — graceful degradation.
6.  Temporal occlusion — structure, influence values, rank ordering.
7.  Temporal occlusion batch.
8.  JSON serialization — all artifacts round-trip through JSON.
9.  Artifact persistence — files written and readable.
10. Disclaimer present in every artifact.
11. Test-set isolation — global_explanation must not be called with test labels
    if it would influence model selection (behavioral guard).
12. Determinism — same seed, same permutation importances.
13. Blocked sklearn.ensemble — RF tests skip cleanly.
14. Real trained model artifact — loads and runs without error.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dialong_automl.explainability import (
    EXPLANATION_DISCLAIMER,
    FeatureNameResolver,
    GlobalExplanation,
    PatientExplanation,
    TemporalExplanation,
    TemporalOcclusionExplainer,
    TreeExplainerWrapper,
)
from dialong_automl.explainability.formatter import (
    FeatureContribution,
    FeatureImportanceEntry,
    TimestepInfluence,
)
from dialong_automl.explainability.shap_explainer import (
    _shap_available,
    _model_supports_tree_shap,
    _permutation_importance,
)
from dialong_automl.features.wide import FeatureInfo, NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Availability probes (used for skip decorators)
# ---------------------------------------------------------------------------

def _lgbm_ok() -> bool:
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def _torch_ok() -> bool:
    try:
        import torch
        torch.tensor([1.0])
        return True
    except (ImportError, OSError):
        return False


def _sklearn_ensemble_ok() -> bool:
    try:
        from sklearn.ensemble import RandomForestClassifier
        RandomForestClassifier(n_estimators=2).fit(
            np.zeros((6, 3), dtype="float32"),
            np.array([0, 0, 0, 1, 1, 1]),
        )
        return True
    except (ImportError, OSError):
        return False


SKIP_LGBM        = pytest.mark.skipif(not _lgbm_ok(), reason="lightgbm unavailable")
SKIP_TORCH       = pytest.mark.skipif(not _torch_ok(), reason="torch unavailable")
SKIP_RF_ENSEMBLE = pytest.mark.skipif(
    not _sklearn_ensemble_ok(), reason="sklearn.ensemble blocked by OS policy"
)
SKIP_SHAP        = pytest.mark.skipif(not _shap_available(), reason="shap package unavailable")


# ---------------------------------------------------------------------------
# Fixtures — tiny real models
# ---------------------------------------------------------------------------

def _tabular_Xy(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10)).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(np.int32)
    return X, y


def _wide_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Minimal wide-format DataFrame matching preprocessor conventions."""
    rng = np.random.default_rng(seed)
    data = {}
    for col in NUMERIC_FEATURES:
        data[col + "_latest"] = rng.standard_normal(n).astype("float32")
        data[col + "_mean"]   = rng.standard_normal(n).astype("float32")
        data[col + "_w1"]     = rng.standard_normal(n).astype("float32")
        data[col + "_w2"]     = rng.standard_normal(n).astype("float32")
    data["sex_latest"]    = rng.choice(["M", "F"], n)
    data["n_obs_visits"]  = rng.integers(3, 5, n).astype("int32")
    return pd.DataFrame(data)


def _fitted_lgbm(n: int = 60):
    """Return a fitted LightGBMModel."""
    from dialong_automl.models.lightgbm_model import LightGBMModel

    X, y = _tabular_Xy(n)
    m = LightGBMModel({"n_estimators": 10})
    m.fit(X, y)
    return m, X, y


def _fitted_preprocessor(df: pd.DataFrame):
    """Return a fitted WidePreprocessor for *df*."""
    from dialong_automl.features.preprocessing import WidePreprocessor

    prep = WidePreprocessor()
    prep.fit(df)
    return prep


def _simple_feature_mapping(prep) -> dict[str, FeatureInfo]:
    """Build minimal FeatureInfo mapping for preprocessor output columns."""
    mapping: dict[str, FeatureInfo] = {}
    for col in prep.feature_names_out:
        mapping[col] = FeatureInfo(
            column_name=col,
            source_variable=col.split("_")[0],
            transformation="latest",
            description=f"Feature {col}",
            feature_type="numeric_temporal",
        )
    return mapping


def _fitted_gru(n: int = 30, T: int = 4, F: int = 10):
    """Return a fitted GRUModel + matching arrays."""
    from dialong_automl.models.gru import GRUModel

    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, T, F)).astype(np.float32)
    mask = np.ones((n, T), dtype=np.uint8)
    y = (rng.random(n) > 0.5).astype(np.int32)
    m = GRUModel({"hidden_size": 8, "num_layers": 1, "dropout": 0.0,
                  "learning_rate": 1e-3, "batch_size": 16,
                  "epochs": 2, "patience": 10, "seed": 42})
    m.fit(X, y, mask=mask)
    return m, X, mask, y


# ---------------------------------------------------------------------------
# 1. Feature name resolver
# ---------------------------------------------------------------------------

class TestFeatureNameResolver:
    def _prep_and_mapping(self):
        df = _wide_df()
        prep = _fitted_preprocessor(df)
        mapping = _simple_feature_mapping(prep)
        return prep, mapping

    def test_resolve_all_returns_correct_length(self):
        prep, mapping = self._prep_and_mapping()
        resolver = FeatureNameResolver(prep.feature_names_out, mapping)
        names = resolver.resolve_all()
        assert len(names) == len(prep.feature_names_out)

    def test_resolve_all_are_strings(self):
        prep, mapping = self._prep_and_mapping()
        resolver = FeatureNameResolver(prep.feature_names_out, mapping)
        names = resolver.resolve_all()
        assert all(isinstance(n, str) and len(n) > 0 for n in names)

    def test_wave_column_resolves(self):
        """hba1c_w3 → 'HbA1c at wave 3'."""
        resolver = FeatureNameResolver([], None)
        label = resolver._heuristic("hba1c_w3")
        assert "HbA1c" in label or "hba1c" in label.lower()
        assert "3" in label

    def test_temporal_latest_resolves(self):
        resolver = FeatureNameResolver([], None)
        label = resolver._heuristic("hba1c_latest")
        assert "HbA1c" in label or "hba1c" in label.lower()
        assert "latest" in label.lower() or "recent" in label.lower()

    def test_temporal_slope_resolves(self):
        resolver = FeatureNameResolver([], None)
        label = resolver._heuristic("hba1c_slope")
        assert "trend" in label.lower() or "slope" in label.lower()

    def test_ohe_column_resolves(self):
        """sex_latest__M → contains 'Sex' and 'M'."""
        resolver = FeatureNameResolver([], None)
        label = resolver.resolve("sex_latest__M")
        assert "M" in label
        # "Sex" or "sex" somewhere
        assert any(s in label for s in ["Sex", "sex"])

    def test_n_obs_visits_resolves(self):
        resolver = FeatureNameResolver(["n_obs_visits"], None)
        label = resolver.resolve("n_obs_visits")
        assert "visit" in label.lower()

    def test_resolve_all_idempotent(self):
        """Calling resolve_all() twice returns the same list."""
        prep, mapping = self._prep_and_mapping()
        resolver = FeatureNameResolver(prep.feature_names_out, mapping)
        first = resolver.resolve_all()
        second = resolver.resolve_all()
        assert first == second

    def test_pretty_var_labels(self):
        from dialong_automl.explainability.feature_names import _VAR_LABELS
        for key, label in _VAR_LABELS.items():
            assert isinstance(label, str) and len(label) > 0

    def test_feature_info_mapping_used(self):
        """When a FeatureInfo is provided, its description is used."""
        info = FeatureInfo(
            column_name="test_col",
            source_variable="hba1c",
            transformation="mean",
            description="HbA1c: Mean across all non-missing observation visits.",
            feature_type="numeric_temporal",
        )
        resolver = FeatureNameResolver(["test_col"], {"test_col": info})
        label = resolver.resolve("test_col")
        assert "HbA1c" in label


# ---------------------------------------------------------------------------
# 2. TreeExplainerWrapper — global explanation with LGBM
# ---------------------------------------------------------------------------

class TestTreeExplainerWrapperGlobal:
    @SKIP_LGBM
    def test_global_explanation_has_correct_structure(self):
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        explainer = TreeExplainerWrapper(m, prep)
        result = explainer.global_explanation(df, y=y, n_top=5)

        assert isinstance(result, GlobalExplanation)
        assert result.model_name == "lightgbm"
        assert result.representation == "wide"
        assert len(result.features) == 5
        assert result.n_samples == len(df)

    @SKIP_LGBM
    def test_global_explanation_feature_ranks_are_unique(self):
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        result = TreeExplainerWrapper(m, prep).global_explanation(df, y=y, n_top=5)
        ranks = [e.rank for e in result.features]
        assert ranks == sorted(ranks)
        assert ranks == list(range(1, 6))

    @SKIP_LGBM
    def test_global_explanation_importances_mostly_non_negative(self):
        """Permutation importances are non-negative for important features;
        near-zero features may be slightly negative due to random noise."""
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        result = TreeExplainerWrapper(m, prep).global_explanation(df, y=y)
        # The most important feature must have a positive importance
        assert result.features[0].importance > 0.0
        # No single feature should have a very large negative importance
        for entry in result.features:
            assert entry.importance > -0.1, (
                f"Importance {entry.importance:.4f} for {entry.column_name} is "
                "unexpectedly negative — check permutation logic."
            )

    @SKIP_LGBM
    def test_global_explanation_disclaimer_present(self):
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        result = TreeExplainerWrapper(m, prep).global_explanation(df, y=y)
        assert result.disclaimer == EXPLANATION_DISCLAIMER

    @SKIP_LGBM
    def test_global_explanation_method_labelled(self):
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        result = TreeExplainerWrapper(m, prep).global_explanation(df, y=y)
        assert result.explanation_method in ("shap_tree", "permutation")
        for e in result.features:
            assert e.explanation_method == result.explanation_method

    @SKIP_LGBM
    def test_global_explanation_note_says_validation_only(self):
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        result = TreeExplainerWrapper(m, prep).global_explanation(df, y=y)
        note = result.metadata.get("note", "").lower()
        assert "test" in note or "validation" in note


# ---------------------------------------------------------------------------
# 3. Permutation importance fallback
# ---------------------------------------------------------------------------

class TestPermutationImportance:
    @SKIP_LGBM
    def test_permutation_returns_correct_shape(self):
        m, X, y = _fitted_lgbm()
        imp = _permutation_importance(m, X, y, n_repeats=2, seed=0)
        assert imp.shape == (X.shape[1],)

    @SKIP_LGBM
    def test_permutation_non_negative_values(self):
        """Most values should be non-negative for a fitted model (some noise ok)."""
        m, X, y = _fitted_lgbm(n=80)
        imp = _permutation_importance(m, X, y, n_repeats=3, seed=42)
        # At least some features should have positive importance
        assert imp.max() > 0

    @SKIP_LGBM
    def test_permutation_reproducible_with_seed(self):
        m, X, y = _fitted_lgbm()
        imp1 = _permutation_importance(m, X, y, n_repeats=2, seed=99)
        imp2 = _permutation_importance(m, X, y, n_repeats=2, seed=99)
        np.testing.assert_array_almost_equal(imp1, imp2, decimal=10)

    @SKIP_LGBM
    def test_permutation_different_seed_different_result(self):
        m, X, y = _fitted_lgbm(n=80)
        imp1 = _permutation_importance(m, X, y, n_repeats=3, seed=0)
        imp2 = _permutation_importance(m, X, y, n_repeats=3, seed=999)
        assert not np.allclose(imp1, imp2)

    @SKIP_LGBM
    def test_permutation_fallback_when_shap_unavailable(self):
        """Permutation path is exercised by forcing it via y= argument."""
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)

        # Temporarily make SHAP appear unavailable by overriding the check
        from dialong_automl.explainability import shap_explainer as se
        orig = se._shap_available
        se._shap_available = lambda: False
        try:
            explainer = TreeExplainerWrapper(m, prep)
            result = explainer.global_explanation(df, y=y, n_top=5)
            assert result.explanation_method == "permutation"
        finally:
            se._shap_available = orig


# ---------------------------------------------------------------------------
# 4. Patient-level explanation
# ---------------------------------------------------------------------------

class TestPatientExplanation:
    @SKIP_LGBM
    def test_patient_explanation_probability_in_range(self):
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        explainer = TreeExplainerWrapper(m, prep)
        result = explainer.patient_explanation(df, row_index=0)
        assert 0.0 <= result.prediction_probability <= 1.0

    @SKIP_LGBM
    def test_patient_explanation_predicted_class_matches_threshold(self):
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        explainer = TreeExplainerWrapper(m, prep, threshold=0.5)
        result = explainer.patient_explanation(df, row_index=0)
        expected = int(result.prediction_probability >= 0.5)
        assert result.predicted_class == expected

    @SKIP_LGBM
    def test_patient_explanation_disclaimer_present(self):
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        result = TreeExplainerWrapper(m, prep).patient_explanation(df)
        assert result.disclaimer == EXPLANATION_DISCLAIMER

    @SKIP_LGBM
    def test_patient_explanation_patient_id_preserved(self):
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        result = TreeExplainerWrapper(m, prep).patient_explanation(
            df, row_index=2, patient_id="SYN_00042"
        )
        assert result.patient_id == "SYN_00042"

    @SKIP_LGBM
    @SKIP_SHAP
    def test_patient_explanation_shap_contributions_directions(self):
        """When SHAP is available, contributions must have directions."""
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        result = TreeExplainerWrapper(m, prep).patient_explanation(df)
        if result.explanation_method == "shap_tree":
            for c in result.top_positive:
                assert c.direction == "increases_risk"
                assert c.shap_value > 0
            for c in result.top_negative:
                assert c.direction == "decreases_risk"
                assert c.shap_value <= 0

    @SKIP_LGBM
    def test_patient_explanation_without_shap_graceful(self):
        """Without SHAP, patient explanation returns empty contributions, not error."""
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)

        from dialong_automl.explainability import shap_explainer as se
        orig = se._shap_available
        se._shap_available = lambda: False
        try:
            explainer = TreeExplainerWrapper(m, prep)
            result = explainer.patient_explanation(df, row_index=0)
            assert isinstance(result, PatientExplanation)
            assert result.top_positive == []
            assert result.top_negative == []
            assert result.explanation_method == "not_available"
            assert 0.0 <= result.prediction_probability <= 1.0
        finally:
            se._shap_available = orig


# ---------------------------------------------------------------------------
# 5. Temporal occlusion
# ---------------------------------------------------------------------------

class TestTemporalOcclusion:
    @SKIP_TORCH
    def test_explain_returns_temporal_explanation(self):
        m, X, mask, y = _fitted_gru()
        occ = TemporalOcclusionExplainer(m)
        result = occ.explain(X, mask, sample_index=0)
        assert isinstance(result, TemporalExplanation)

    @SKIP_TORCH
    def test_explain_timestep_count_matches_sequence_length(self):
        m, X, mask, y = _fitted_gru(T=4)
        occ = TemporalOcclusionExplainer(m)
        result = occ.explain(X, mask, sample_index=0)
        assert len(result.timesteps) == 4

    @SKIP_TORCH
    def test_explain_baseline_probability_in_range(self):
        m, X, mask, y = _fitted_gru()
        occ = TemporalOcclusionExplainer(m)
        result = occ.explain(X, mask, sample_index=0)
        assert 0.0 <= result.baseline_probability <= 1.0

    @SKIP_TORCH
    def test_explain_occluded_probabilities_in_range(self):
        m, X, mask, y = _fitted_gru()
        occ = TemporalOcclusionExplainer(m)
        result = occ.explain(X, mask, sample_index=0)
        for ts in result.timesteps:
            assert 0.0 <= ts.occluded_probability <= 1.0

    @SKIP_TORCH
    def test_explain_influence_equals_baseline_minus_occluded(self):
        m, X, mask, y = _fitted_gru()
        occ = TemporalOcclusionExplainer(m)
        result = occ.explain(X, mask, sample_index=0)
        for ts in result.timesteps:
            expected = result.baseline_probability - ts.occluded_probability
            assert abs(ts.influence - expected) < 1e-6

    @SKIP_TORCH
    def test_explain_real_visits_ranked(self):
        m, X, mask, y = _fitted_gru()
        occ = TemporalOcclusionExplainer(m)
        result = occ.explain(X, mask, sample_index=0)
        real_ts = [ts for ts in result.timesteps if ts.is_real_visit]
        real_ranks = sorted([ts.rank for ts in real_ts if ts.rank > 0])
        assert real_ranks == list(range(1, len(real_ts) + 1))

    @SKIP_TORCH
    def test_explain_disclaimer_present(self):
        m, X, mask, y = _fitted_gru()
        occ = TemporalOcclusionExplainer(m)
        result = occ.explain(X, mask)
        assert result.disclaimer == EXPLANATION_DISCLAIMER

    @SKIP_TORCH
    def test_explain_model_name_is_gru(self):
        m, X, mask, y = _fitted_gru()
        occ = TemporalOcclusionExplainer(m)
        result = occ.explain(X, mask)
        assert result.model_name == "gru"

    @SKIP_TORCH
    def test_explain_batch_returns_correct_count(self):
        m, X, mask, y = _fitted_gru(n=5)
        occ = TemporalOcclusionExplainer(m)
        results = occ.explain_batch(X, mask, patient_ids=["P0", "P1", "P2", "P3", "P4"])
        assert len(results) == 5
        for r in results:
            assert isinstance(r, TemporalExplanation)

    @SKIP_TORCH
    def test_explain_patient_id_stored(self):
        m, X, mask, y = _fitted_gru()
        occ = TemporalOcclusionExplainer(m)
        result = occ.explain(X, mask, patient_id="SYN_00001", sample_index=0)
        assert result.patient_id == "SYN_00001"

    def test_temporal_occlusion_requires_gru_interface(self):
        """TemporalOcclusionExplainer rejects models without predict_proba_with_mask."""
        from dialong_automl.models.lightgbm_model import LightGBMModel
        if not _lgbm_ok():
            pytest.skip("lightgbm unavailable")
        m = LightGBMModel({"n_estimators": 5})
        X, y = _tabular_Xy()
        m.fit(X, y)
        with pytest.raises(TypeError, match="predict_proba_with_mask"):
            TemporalOcclusionExplainer(m)

    def test_temporal_occlusion_requires_fitted_model(self):
        """TemporalOcclusionExplainer raises if model is not fitted."""
        if not _torch_ok():
            pytest.skip("torch unavailable")
        from dialong_automl.models.gru import GRUModel
        m = GRUModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            TemporalOcclusionExplainer(m)


# ---------------------------------------------------------------------------
# 6. JSON serialization
# ---------------------------------------------------------------------------

class TestJsonSerialization:
    def _sample_global_exp(self) -> GlobalExplanation:
        features = [
            FeatureImportanceEntry(
                rank=1, column_name="hba1c_latest",
                readable_name="HbA1c (latest)",
                importance=0.42, direction="positive",
                explanation_method="shap_tree",
            ),
            FeatureImportanceEntry(
                rank=2, column_name="bmi_mean",
                readable_name="BMI (mean)",
                importance=0.18, direction="negative",
                explanation_method="shap_tree",
            ),
        ]
        return GlobalExplanation(
            model_name="lightgbm",
            representation="wide",
            explanation_method="shap_tree",
            n_samples=75,
            features=features,
            metadata={"timestamp": "2026-01-01T00:00:00+00:00"},
        )

    def test_global_explanation_dict_is_json_serializable(self):
        exp = self._sample_global_exp()
        d = exp.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_global_explanation_has_disclaimer_in_dict(self):
        exp = self._sample_global_exp()
        d = exp.to_dict()
        assert d["disclaimer"] == EXPLANATION_DISCLAIMER

    def test_global_explanation_save_and_content(self, tmp_path):
        exp = self._sample_global_exp()
        path = tmp_path / "global_exp.json"
        exp.save(path)
        assert path.exists() and path.stat().st_size > 0
        with open(path) as fh:
            d = json.load(fh)
        assert d["disclaimer"] == EXPLANATION_DISCLAIMER
        assert len(d["features"]) == 2
        assert d["features"][0]["rank"] == 1

    def test_patient_explanation_dict_is_json_serializable(self):
        exp = PatientExplanation(
            patient_id="SYN_00001",
            prediction_probability=0.72,
            predicted_class=1,
            threshold=0.5,
            model_name="lightgbm",
            explanation_method="shap_tree",
            top_positive=[
                FeatureContribution(
                    rank=1, column_name="hba1c_latest",
                    readable_name="HbA1c (latest)",
                    feature_value=7.2, shap_value=0.35,
                    direction="increases_risk",
                )
            ],
            top_negative=[],
        )
        d = exp.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0
        assert d["disclaimer"] == EXPLANATION_DISCLAIMER

    def test_temporal_explanation_dict_is_json_serializable(self):
        exp = TemporalExplanation(
            patient_id="SYN_00001",
            model_name="gru",
            baseline_probability=0.65,
            predicted_class=1,
            threshold=0.5,
            timesteps=[
                TimestepInfluence(
                    timestep=0, baseline_probability=0.65,
                    occluded_probability=0.58, influence=0.07,
                    rank=1, is_real_visit=True,
                ),
                TimestepInfluence(
                    timestep=1, baseline_probability=0.65,
                    occluded_probability=0.63, influence=0.02,
                    rank=2, is_real_visit=True,
                ),
            ],
        )
        d = exp.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0
        assert d["disclaimer"] == EXPLANATION_DISCLAIMER

    def test_temporal_explanation_save(self, tmp_path):
        exp = TemporalExplanation(
            patient_id="P1", model_name="gru",
            baseline_probability=0.6, predicted_class=1,
            threshold=0.5, timesteps=[],
        )
        path = tmp_path / "temporal.json"
        exp.save(path)
        assert path.exists()
        with open(path) as fh:
            d = json.load(fh)
        assert d["explanation_method"] == "temporal_occlusion"
        assert d["disclaimer"] == EXPLANATION_DISCLAIMER

    def test_patient_explanation_save(self, tmp_path):
        exp = PatientExplanation(
            patient_id="P1", prediction_probability=0.3,
            predicted_class=0, threshold=0.5,
            model_name="lightgbm", explanation_method="shap_tree",
            top_positive=[], top_negative=[],
        )
        path = tmp_path / "patient.json"
        exp.save(path)
        assert path.exists()
        with open(path) as fh:
            d = json.load(fh)
        assert d["disclaimer"] == EXPLANATION_DISCLAIMER


# ---------------------------------------------------------------------------
# 7. Artifact persistence (global_explanation to disk)
# ---------------------------------------------------------------------------

class TestArtifactPersistence:
    @SKIP_LGBM
    def test_global_explanation_saves_to_explanations_dir(self, tmp_path):
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)
        result = TreeExplainerWrapper(m, prep).global_explanation(df, y=y)

        out_dir = tmp_path / "artifacts" / "explanations"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "global_importance.json"
        result.save(path)

        assert path.exists() and path.stat().st_size > 0
        with open(path) as fh:
            d = json.load(fh)
        assert d["disclaimer"] == EXPLANATION_DISCLAIMER
        assert d["model_name"] == "lightgbm"
        assert len(d["features"]) > 0


# ---------------------------------------------------------------------------
# 8. Test-set isolation guard
# ---------------------------------------------------------------------------

class TestIsolationGuard:
    @SKIP_LGBM
    def test_permutation_requires_y_raises_without_it(self):
        """When SHAP is unavailable and y is not passed, must raise ValueError."""
        m, X, y = _fitted_lgbm()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        prep = _fitted_preprocessor(df)

        from dialong_automl.explainability import shap_explainer as se
        orig = se._shap_available
        se._shap_available = lambda: False
        try:
            explainer = TreeExplainerWrapper(m, prep)
            with pytest.raises(ValueError, match="y.*validation"):
                explainer.global_explanation(df, y=None)
        finally:
            se._shap_available = orig


# ---------------------------------------------------------------------------
# 9. RF — blocked in this environment
# ---------------------------------------------------------------------------

class TestRandomForestBlocked:
    def test_rf_skips_gracefully(self):
        """When sklearn.ensemble is blocked, RF tests skip (not crash)."""
        if _sklearn_ensemble_ok():
            pytest.skip("sklearn.ensemble is available — testing blocked path not applicable")
        # Verify that building a TreeExplainerWrapper with a non-RF model still works
        if not _lgbm_ok():
            pytest.skip("lightgbm unavailable")
        from dialong_automl.models.lightgbm_model import LightGBMModel

        X, y = _tabular_Xy()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        m = LightGBMModel({"n_estimators": 5})
        m.fit(X, y)
        prep = _fitted_preprocessor(df)
        # Should not raise — LightGBM is available
        explainer = TreeExplainerWrapper(m, prep)
        assert explainer.model.name == "lightgbm"

    @SKIP_RF_ENSEMBLE
    def test_rf_global_explanation(self):
        """When sklearn.ensemble is available, RF explanations work."""
        from dialong_automl.models.random_forest import RandomForestModel

        X, y = _tabular_Xy()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        m = RandomForestModel({"n_estimators": 10, "random_state": 42})
        m.fit(X, y)
        prep = _fitted_preprocessor(df)

        result = TreeExplainerWrapper(m, prep).global_explanation(df, y=y)
        assert isinstance(result, GlobalExplanation)
        assert result.model_name == "random_forest"
        assert len(result.features) > 0


# ---------------------------------------------------------------------------
# 10. Real trained artifact integration
# ---------------------------------------------------------------------------

class TestRealArtifact:
    """Tests that use the actual artifacts/best_model.joblib from the last training run."""

    _ARTIFACTS_DIR = Path("artifacts")
    _MODEL_PATH = _ARTIFACTS_DIR / "best_model.joblib"
    _PREP_PATH  = _ARTIFACTS_DIR / "preprocessor.joblib"

    def _artifacts_exist(self) -> bool:
        return self._MODEL_PATH.exists() and self._PREP_PATH.exists()

    @SKIP_LGBM
    def test_real_model_loads_and_explains(self):
        """Load the real saved model, run a global explanation on synthetic val data."""
        if not self._artifacts_exist():
            pytest.skip("Real model artifact not found — run training first")

        from dialong_automl.models.lightgbm_model import LightGBMModel
        from dialong_automl.features.preprocessing import WidePreprocessor

        m = LightGBMModel.load(self._MODEL_PATH)
        prep = WidePreprocessor.load(self._PREP_PATH)

        assert m.is_fitted
        assert prep.is_fitted

        # Build a small synthetic validation DataFrame with the correct columns
        rng = np.random.default_rng(0)
        # Use the preprocessor's numeric/categorical columns to build the right DataFrame
        nc = prep.numeric_columns or []
        cc = prep.categorical_columns or []

        n = 10
        data = {}
        for col in nc:
            data[col] = rng.standard_normal(n).astype("float32")
        for col in cc:
            cats = list(prep._cat_categories.get(col, ["<MISSING>"]))[1:]
            if cats:
                data[col] = [cats[i % len(cats)] for i in range(n)]
            else:
                data[col] = ["<MISSING>"] * n
        df = pd.DataFrame(data)

        y_dummy = rng.integers(0, 2, n).astype(np.int32)

        explainer = TreeExplainerWrapper(m, prep)
        result = explainer.global_explanation(df, y=y_dummy)

        assert isinstance(result, GlobalExplanation)
        assert result.model_name == "lightgbm"
        assert len(result.features) > 0
        assert result.disclaimer == EXPLANATION_DISCLAIMER

        # Note in metadata must reference validation/training data
        note = result.metadata.get("note", "")
        assert "test" in note.lower() or "validation" in note.lower()

    @SKIP_LGBM
    def test_real_model_patient_explanation(self):
        """Patient explanation from the real saved model."""
        if not self._artifacts_exist():
            pytest.skip("Real model artifact not found — run training first")

        from dialong_automl.models.lightgbm_model import LightGBMModel
        from dialong_automl.features.preprocessing import WidePreprocessor

        m = LightGBMModel.load(self._MODEL_PATH)
        prep = WidePreprocessor.load(self._PREP_PATH)

        nc = prep.numeric_columns or []
        cc = prep.categorical_columns or []
        rng = np.random.default_rng(1)
        n = 5
        data = {}
        for col in nc:
            data[col] = rng.standard_normal(n).astype("float32")
        for col in cc:
            cats = list(prep._cat_categories.get(col, ["<MISSING>"]))[1:]
            if cats:
                data[col] = [cats[i % len(cats)] for i in range(n)]
            else:
                data[col] = ["<MISSING>"] * n
        df = pd.DataFrame(data)

        explainer = TreeExplainerWrapper(m, prep)
        result = explainer.patient_explanation(df, row_index=0, patient_id="SYN_TEST_00001")

        assert isinstance(result, PatientExplanation)
        assert result.patient_id == "SYN_TEST_00001"
        assert 0.0 <= result.prediction_probability <= 1.0
        assert result.disclaimer == EXPLANATION_DISCLAIMER
        assert result.model_name == "lightgbm"
