"""Phase 9 — FastAPI API tests (dialong_automl.api).

Covers:
1. Health endpoint (loaded and not-loaded states).
2. Metadata endpoint (loaded and not-loaded states).
3. Successful prediction from real tiny artifacts.
4. Invalid request body (Pydantic 422).
5. Missing model artifacts (503, no path leakage).
6. Incompatible / corrupt artifacts (503, no stack trace).
7. Rejection of forbidden leakage/future fields (422).
8. Correct disclaimer and response schema.

Uses FastAPI TestClient. Real artifacts are built with a tiny lightgbm run in
a temp dir (no expensive training). No project artifacts are created or
overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dialong_automl.api import create_app
from dialong_automl.config.schema import AppConfig
from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator


def _lgbm_ok() -> bool:
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


SKIP_LGBM = pytest.mark.skipif(not _lgbm_ok(), reason="lightgbm unavailable")

_DISCLAIMER_MARKER = "not clinical"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_full_csv(tmp_path: Path, n_patients: int = 80) -> str:
    gen_cfg = GeneratorConfig(
        n_patients=n_patients,
        min_obs_visits=2, max_obs_visits=3,
        min_future_visits=2, max_future_visits=3,
        seed=42, progression_prevalence=0.40, missingness_rate=0.0,
    )
    res = SyntheticGenerator(config=gen_cfg).generate()
    paths = res.save(output_dir=tmp_path / "data" / "synthetic")
    return str(paths["full_csv"])


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
    cfg.optuna.study_name = "api_test_study"
    cfg.optuna.seed = 42
    return cfg


@pytest.fixture()
def artifact_dir(tmp_path) -> Path:
    """Train a tiny wide model and return its artifact directory."""
    if not _lgbm_ok():
        pytest.skip("lightgbm unavailable")
    from dialong_automl.training.pipeline import TrainingPipeline

    csv = _tiny_full_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    TrainingPipeline(cfg, full_csv_path=csv).run()
    return Path(cfg.paths.artifacts_dir)


@pytest.fixture()
def loaded_client(artifact_dir) -> TestClient:
    with TestClient(create_app(artifact_dir=artifact_dir)) as client:
        yield client


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
# 1. Health
# ---------------------------------------------------------------------------

@SKIP_LGBM
def test_health_loaded(loaded_client):
    r = loaded_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_not_loaded(tmp_path):
    with TestClient(create_app(artifact_dir=tmp_path / "empty")) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


# ---------------------------------------------------------------------------
# 2. Metadata
# ---------------------------------------------------------------------------

@SKIP_LGBM
def test_metadata_loaded(loaded_client):
    r = loaded_client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["model"] == "lightgbm"
    assert body["representation"] == "wide"
    assert body["threshold"] == 0.5
    assert _DISCLAIMER_MARKER in body["disclaimer"].lower()
    # No filesystem paths leaked anywhere in the metadata payload.
    blob = json.dumps(body)
    assert "artifact_paths" not in blob
    assert "\\\\" not in blob and "/artifacts/" not in blob


def test_metadata_not_loaded(tmp_path):
    with TestClient(create_app(artifact_dir=tmp_path / "empty")) as client:
        r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is False
    assert body["model"] is None
    assert _DISCLAIMER_MARKER in body["disclaimer"].lower()


# ---------------------------------------------------------------------------
# 3. Successful prediction + 8. disclaimer / schema
# ---------------------------------------------------------------------------

@SKIP_LGBM
def test_predict_success_and_schema(loaded_client):
    r = loaded_client.post(
        "/predict", json={"patient_id": "P_API", "visits": _example_visits()}
    )
    assert r.status_code == 200
    body = r.json()
    # Response schema keys.
    for key in ("patient_id", "probability", "prediction", "threshold",
                "model", "representation", "n_visits", "disclaimer"):
        assert key in body, f"missing key {key}"
    assert body["patient_id"] == "P_API"
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["prediction"] == int(body["probability"] >= body["threshold"])
    assert body["model"] == "lightgbm"
    assert body["representation"] == "wide"
    assert body["n_visits"] == 2
    assert _DISCLAIMER_MARKER in body["disclaimer"].lower()


@SKIP_LGBM
def test_predict_accepts_default_patient_id(loaded_client):
    r = loaded_client.post("/predict", json={"visits": _example_visits()})
    assert r.status_code == 200
    assert r.json()["patient_id"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# 4. Invalid request body
# ---------------------------------------------------------------------------

@SKIP_LGBM
def test_predict_empty_visits_rejected(loaded_client):
    r = loaded_client.post("/predict", json={"visits": []})
    assert r.status_code == 422  # Pydantic min_length=1


@SKIP_LGBM
def test_predict_missing_visits_field_rejected(loaded_client):
    r = loaded_client.post("/predict", json={"patient_id": "P1"})
    assert r.status_code == 422


@SKIP_LGBM
def test_predict_visit_missing_visit_date_rejected(loaded_client):
    r = loaded_client.post("/predict", json={"visits": [{"hba1c": 5.6}]})
    assert r.status_code == 422  # VisitModel requires visit_date


@SKIP_LGBM
def test_predict_unparseable_visit_date_rejected(loaded_client):
    r = loaded_client.post(
        "/predict", json={"visits": [{"visit_date": "not-a-date", "hba1c": 5.6}]}
    )
    assert r.status_code == 422
    # No filesystem path or stack trace in the error.
    detail = r.json()["detail"]
    assert "Traceback" not in str(detail)


# ---------------------------------------------------------------------------
# 5. Missing model artifacts
# ---------------------------------------------------------------------------

def test_predict_missing_artifacts_returns_503(tmp_path):
    with TestClient(create_app(artifact_dir=tmp_path / "empty")) as client:
        r = client.post(
            "/predict", json={"visits": [{"visit_date": "2020-01-01", "hba1c": 5.6}]}
        )
    assert r.status_code == 503
    detail = r.json()["detail"]
    # Generic message — no filesystem path leaked.
    assert "\\" not in detail and "/" not in detail
    assert "Traceback" not in detail


# ---------------------------------------------------------------------------
# 6. Incompatible / corrupt artifacts
# ---------------------------------------------------------------------------

def test_corrupt_metadata_degrades_gracefully(tmp_path):
    adir = tmp_path / "artifacts"
    adir.mkdir()
    (adir / "model_metadata.json").write_text("{ this is not valid json",
                                               encoding="utf-8")
    with TestClient(create_app(artifact_dir=adir)) as client:
        # Startup tolerates the corrupt metadata -> not loaded.
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["model_loaded"] is False
        # Prediction is unavailable (503), no stack trace leaked.
        p = client.post(
            "/predict", json={"visits": [{"visit_date": "2020-01-01", "hba1c": 5.6}]}
        )
        assert p.status_code == 503
        assert "Traceback" not in p.json()["detail"]


def test_metadata_present_but_model_file_missing_returns_503(tmp_path):
    """Metadata loads at startup, but the model file is absent -> 503 at predict."""
    adir = tmp_path / "artifacts"
    adir.mkdir()
    (adir / "model_metadata.json").write_text(
        json.dumps({
            "selected_model": "lightgbm",
            "representation": "wide",
            "threshold": 0.5,
            "observation_visits": 3,
            "synthetic_data": True,
            "disclaimer": "Synthetic demonstration data. Not clinical evidence.",
        }),
        encoding="utf-8",
    )
    with TestClient(create_app(artifact_dir=adir)) as client:
        # Startup sees valid metadata -> loaded True; metadata endpoint works.
        assert client.get("/health").json()["model_loaded"] is True
        md = client.get("/metadata").json()
        assert md["model"] == "lightgbm"
        # But scoring fails because the model file is missing -> 503, no path.
        p = client.post(
            "/predict", json={"visits": [{"visit_date": "2020-01-01", "hba1c": 5.6}]}
        )
        assert p.status_code == 503
        assert "\\" not in p.json()["detail"] and "/" not in p.json()["detail"]


# ---------------------------------------------------------------------------
# 7. Forbidden leakage/future fields
# ---------------------------------------------------------------------------

@SKIP_LGBM
def test_predict_rejects_leakage_field(loaded_client):
    r = loaded_client.post(
        "/predict",
        json={"visits": [{"visit_date": "2020-01-01", "hba1c": 6.0, "target": 1}]},
    )
    assert r.status_code == 422
    assert "future" in r.json()["detail"].lower() or "forbidden" in r.json()["detail"].lower()


@SKIP_LGBM
def test_predict_rejects_future_outcome_field(loaded_client):
    r = loaded_client.post(
        "/predict",
        json={"visits": [{"visit_date": "2020-01-01", "hba1c": 6.0, "future_hba1c": 9.0}]},
    )
    assert r.status_code == 422
