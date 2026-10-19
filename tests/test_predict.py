"""Phase 8 — Prediction tests (dialong_automl.predict).

Covers:
- Prediction from saved artifacts returns a probability + binary prediction.
- The predict feature pipeline matches the training pipeline (same model,
  same preprocessor, same probability) for the same input.
- Invalid input (missing visits, leakage column, bad JSON) raises clearly.
- Missing / incompatible artifacts raise clearly and never train.
- The documented CLI entry point works end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dialong_automl.predict as predict_mod
from dialong_automl.predict import PredictionError, predict_from_artifacts
from dialong_automl.config.schema import AppConfig
from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator


def _lgbm_ok() -> bool:
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


SKIP_LGBM = pytest.mark.skipif(not _lgbm_ok(), reason="lightgbm unavailable")


# ---------------------------------------------------------------------------
# Fixtures — a tiny trained artifact set (wide / lightgbm)
# ---------------------------------------------------------------------------

def _tiny_full_csv(tmp_path: Path, n_patients: int = 80) -> Path:
    gen_cfg = GeneratorConfig(
        n_patients=n_patients,
        min_obs_visits=2, max_obs_visits=3,
        min_future_visits=2, max_future_visits=3,
        seed=42, progression_prevalence=0.40, missingness_rate=0.0,
    )
    res = SyntheticGenerator(config=gen_cfg).generate()
    paths = res.save(output_dir=tmp_path / "data" / "synthetic")
    return paths["full_csv"]


def _tiny_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.outputs_dir = str(tmp_path / "outputs")
    cfg.paths.logs_dir = str(tmp_path / "logs")
    cfg.cohort.min_observation_visits = 2
    cfg.cohort.max_observation_visits = 3
    cfg.cohort.prediction_horizon_days = 365 * 5
    cfg.cohort.exclude_known_diabetes_at_baseline = False
    cfg.cohort.require_followup_after_cutoff = True
    cfg.split.train_size = 0.60
    cfg.split.validation_size = 0.20
    cfg.split.test_size = 0.20
    cfg.split.stratify = False
    cfg.training.seed = 42
    cfg.training.threshold = 0.5
    cfg.model.enabled_models = ["lightgbm"]
    cfg.optuna.n_trials = 1
    cfg.optuna.storage = None
    cfg.optuna.study_name = "predict_test_study"
    cfg.optuna.seed = 42
    return cfg


@pytest.fixture()
def trained_artifacts(tmp_path):
    """Train a tiny wide model and return (artifact_dir, csv_path, cfg)."""
    if not _lgbm_ok():
        pytest.skip("lightgbm unavailable")
    from dialong_automl.training.pipeline import TrainingPipeline

    csv = _tiny_full_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    TrainingPipeline(cfg, full_csv_path=csv).run()
    return Path(cfg.paths.artifacts_dir), csv, cfg


def _example_visits() -> list[dict]:
    base = {
        "sex": "F", "smoking_status": "former",
        "physical_activity_level": "moderate",
        "family_history_diabetes": "yes", "metformin_use": "no",
        "hypertension_diagnosis": "no",
    }
    return [
        {"visit_date": "2019-02-11", "hba1c": 5.6, "fasting_glucose": 94.0,
         "bmi": 27.4, "systolic_bp": 124.0, "diastolic_bp": 79.0,
         "ldl_cholesterol": 118.0, "hdl_cholesterol": 51.0,
         "triglycerides": 142.0, "heart_rate": 72.0, "age_at_visit": 54.2, **base},
        {"visit_date": "2019-08-19", "hba1c": 5.8, "fasting_glucose": 99.0,
         "bmi": 27.9, "systolic_bp": 127.0, "diastolic_bp": 80.0,
         "ldl_cholesterol": 121.0, "hdl_cholesterol": 50.0,
         "triglycerides": 150.0, "heart_rate": 74.0, "age_at_visit": 54.7, **base},
    ]


# ---------------------------------------------------------------------------
# Prediction from artifacts
# ---------------------------------------------------------------------------

def test_predict_returns_probability_and_class(trained_artifacts):
    artifact_dir, _csv, _cfg = trained_artifacts
    result = predict_from_artifacts(artifact_dir, "P1", _example_visits())
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["prediction"] == int(result["probability"] >= result["threshold"])
    assert result["model"] == "lightgbm"
    assert result["representation"] == "wide"
    assert "disclaimer" in result


def test_predict_matches_training_pipeline(trained_artifacts):
    """Predict path must reproduce the training feature pipeline exactly."""
    artifact_dir, _csv, cfg = trained_artifacts
    from dialong_automl.features.preprocessing import WidePreprocessor
    from dialong_automl.features.wide import WideRepresentationBuilder
    from dialong_automl.models.lightgbm_model import LightGBMModel

    visits = _example_visits()
    result = predict_from_artifacts(artifact_dir, "P1", visits)

    # Manually reconstruct using the saved artifacts and the same builder.
    model = LightGBMModel.load(artifact_dir / "best_model.joblib")
    prep = WidePreprocessor.load(artifact_dir / "preprocessor.joblib")
    df = pd.DataFrame(visits)
    df["patient_id"] = "P1"
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    wide = WideRepresentationBuilder(max_waves=cfg.cohort.max_observation_visits).build(df)
    X = prep.transform(wide.X)
    expected = float(np.asarray(model.predict_proba(X)).ravel()[0])

    assert result["probability"] == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

def test_predict_rejects_leakage_column(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"visits": [{"visit_date": "2020-01-01", "hba1c": 6.0, "target": 1}]}),
        encoding="utf-8",
    )
    with pytest.raises(PredictionError, match="leakage"):
        predict_mod._load_patient_json(bad)


def test_predict_rejects_empty_visits(tmp_path):
    bad = tmp_path / "empty.json"
    bad.write_text(json.dumps({"patient_id": "P1", "visits": []}), encoding="utf-8")
    with pytest.raises(PredictionError, match="at least one visit"):
        predict_mod._load_patient_json(bad)


def test_predict_rejects_missing_visit_date(tmp_path):
    bad = tmp_path / "noedate.json"
    bad.write_text(json.dumps({"visits": [{"hba1c": 6.0}]}), encoding="utf-8")
    with pytest.raises(PredictionError, match="visit_date"):
        predict_mod._load_patient_json(bad)


def test_predict_rejects_bad_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(PredictionError, match="not valid JSON"):
        predict_mod._load_patient_json(bad)


def test_predict_accepts_bare_list(tmp_path):
    good = tmp_path / "list.json"
    good.write_text(json.dumps(_example_visits()), encoding="utf-8")
    pid, visits = predict_mod._load_patient_json(good)
    assert pid == "UNKNOWN"
    assert len(visits) == 2


# ---------------------------------------------------------------------------
# Missing / incompatible artifacts
# ---------------------------------------------------------------------------

def test_predict_missing_artifact_dir():
    with pytest.raises(PredictionError, match="Artifact directory not found"):
        predict_from_artifacts("definitely_not_a_dir_12345", "P1", _example_visits())


def test_predict_missing_metadata(tmp_path):
    (tmp_path / "artifacts").mkdir()
    with pytest.raises(PredictionError, match="Model metadata not found"):
        predict_from_artifacts(tmp_path / "artifacts", "P1", _example_visits())


def test_predict_missing_model_file(tmp_path):
    """Metadata present but model file missing -> clear error, no training."""
    adir = tmp_path / "artifacts"
    adir.mkdir()
    (adir / "model_metadata.json").write_text(
        json.dumps({
            "selected_model": "lightgbm",
            "representation": "wide",
            "threshold": 0.5,
            "observation_visits": 3,
        }),
        encoding="utf-8",
    )
    with pytest.raises(PredictionError, match="Model artifact not found"):
        predict_from_artifacts(adir, "P1", _example_visits())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def test_predict_cli_end_to_end(trained_artifacts, tmp_path, capsys):
    artifact_dir, _csv, _cfg = trained_artifacts
    input_path = tmp_path / "patient.json"
    input_path.write_text(
        json.dumps({"patient_id": "P_CLI", "visits": _example_visits()}),
        encoding="utf-8",
    )
    rc = predict_mod.main(
        ["--artifact-dir", str(artifact_dir), "--input", str(input_path)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["patient_id"] == "P_CLI"
    assert 0.0 <= payload["probability"] <= 1.0


def test_predict_cli_missing_input_returns_error(trained_artifacts, tmp_path):
    artifact_dir, _csv, _cfg = trained_artifacts
    rc = predict_mod.main(
        ["--artifact-dir", str(artifact_dir), "--input", str(tmp_path / "nope.json")]
    )
    assert rc == 1


# ---------------------------------------------------------------------------
# Incomplete feature sets (regression for the sequence IndexError fix)
# ---------------------------------------------------------------------------

def _torch_ok() -> bool:
    try:
        import torch
        torch.tensor([1.0])
        return True
    except (ImportError, OSError):
        return False


SKIP_TORCH = pytest.mark.skipif(not _torch_ok(), reason="torch unavailable")


def test_ensure_feature_columns_adds_missing_without_fabricating():
    """_ensure_feature_columns adds every canonical feature column; absent
    numerics become NaN and absent categoricals become None (not invented)."""
    import numpy as np
    import pandas as pd
    from dialong_automl.predict import _ensure_feature_columns
    from dialong_automl.features.wide import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    df = pd.DataFrame([{"patient_id": "P", "visit_date": pd.Timestamp("2020-01-01"), "hba1c": 5.6}])
    out = _ensure_feature_columns(df)
    for col in NUMERIC_FEATURES:
        assert col in out.columns
    for col in CATEGORICAL_FEATURES:
        assert col in out.columns
    # Present value preserved; missing numerics are NaN; missing categoricals None.
    assert out.loc[0, "hba1c"] == 5.6
    assert bool(np.isnan(out.loc[0, "bmi"]))
    assert out.loc[0, "sex"] is None


@pytest.fixture()
def gru_artifacts(tmp_path):
    """Train a tiny GRU (sequence) model and return its artifact directory."""
    if not _torch_ok():
        pytest.skip("torch unavailable")
    from dialong_automl.training.pipeline import TrainingPipeline

    csv = _tiny_full_csv(tmp_path, n_patients=90)
    cfg = _tiny_cfg(tmp_path)
    cfg.model.enabled_models = ["gru"]
    cfg.gru.epochs = 2
    cfg.gru.patience = 2
    cfg.optuna.study_name = "predict_gru_test_study"
    TrainingPipeline(cfg, full_csv_path=csv).run()
    return Path(cfg.paths.artifacts_dir)


@SKIP_TORCH
def test_sequence_prediction_with_partial_features_does_not_crash(gru_artifacts):
    """Regression: a partial trajectory (missing many clinical fields) must not
    raise an IndexError in the sequence preprocessor; it should return a valid
    probability because missing features are imputed by the fitted preprocessor."""
    partial = [
        {"visit_date": "2019-02-11", "hba1c": 5.6, "sex": "F"},
        {"visit_date": "2019-08-19", "hba1c": 5.8, "sex": "F"},
    ]
    result = predict_from_artifacts(gru_artifacts, "P_PARTIAL", partial)
    assert result["representation"] == "sequence"
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in (0, 1)


@SKIP_TORCH
def test_sequence_prediction_with_full_features_still_works(gru_artifacts):
    """A complete trajectory continues to work after the fix."""
    result = predict_from_artifacts(gru_artifacts, "P_FULL", _example_visits())
    assert result["representation"] == "sequence"
    assert 0.0 <= result["probability"] <= 1.0


@SKIP_LGBM
def test_wide_prediction_with_partial_features_works(trained_artifacts):
    """The wide path also accepts a partial trajectory (imputed), unchanged by the fix."""
    artifact_dir, _csv, _cfg = trained_artifacts
    partial = [{"visit_date": "2019-02-11", "hba1c": 5.6}, {"visit_date": "2019-08-19", "hba1c": 5.8}]
    result = predict_from_artifacts(artifact_dir, "P", partial)
    assert result["representation"] == "wide"
    assert 0.0 <= result["probability"] <= 1.0
