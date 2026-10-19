"""Phase 10 — patient/visit API + DB-backed prediction tests.

Uses a temporary SQLite DB and (for prediction) a tiny trained wide model, so
the developer's real database and artifacts are never touched.
"""

from __future__ import annotations

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


def _train_tiny_wide(tmp_path: Path) -> Path:
    from dialong_automl.training.pipeline import TrainingPipeline

    gen = SyntheticGenerator(config=GeneratorConfig(
        n_patients=80, min_obs_visits=2, max_obs_visits=3,
        min_future_visits=2, max_future_visits=3, seed=42,
        progression_prevalence=0.40, missingness_rate=0.0,
    )).generate()
    csv = gen.save(output_dir=tmp_path / "data" / "synthetic")["full_csv"]

    cfg = AppConfig()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.outputs_dir = str(tmp_path / "outputs")
    cfg.paths.logs_dir = str(tmp_path / "logs")
    cfg.cohort.min_observation_visits = 2
    cfg.cohort.max_observation_visits = 3
    cfg.cohort.prediction_horizon_days = 365 * 5
    cfg.cohort.exclude_known_diabetes_at_baseline = False
    cfg.split.train_size = 0.6
    cfg.split.validation_size = 0.2
    cfg.split.test_size = 0.2
    cfg.split.stratify = False
    cfg.training.seed = 42
    cfg.model.enabled_models = ["lightgbm"]
    cfg.optuna.n_trials = 1
    cfg.optuna.storage = None
    cfg.optuna.study_name = "patients_api_test"
    cfg.optuna.seed = 42
    TrainingPipeline(cfg, full_csv_path=csv).run()
    return Path(cfg.paths.artifacts_dir)


@pytest.fixture()
def client_no_model(tmp_path):
    """App with a temp DB but no model artifacts (predict unavailable)."""
    app = create_app(artifact_dir=tmp_path / "empty_artifacts", db_path=tmp_path / "p.db")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_with_model(tmp_path):
    if not _lgbm_ok():
        pytest.skip("lightgbm unavailable")
    artifacts = _train_tiny_wide(tmp_path)
    app = create_app(artifact_dir=artifacts, db_path=tmp_path / "p.db")
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Patient CRUD
# ---------------------------------------------------------------------------

def test_create_patient_trims_and_reports(client_no_model):
    r = client_no_model.post("/patients", json={"patient_id": "  SYN_0001 "})
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] == "SYN_0001"
    assert body["created"] is True
    assert body["exists"] is False
    assert body["visits"] == []

    # Second call: exists, not created.
    r2 = client_no_model.post("/patients", json={"patient_id": "SYN_0001"})
    assert r2.json()["exists"] is True
    assert r2.json()["created"] is False


def test_create_patient_requires_id(client_no_model):
    assert client_no_model.post("/patients", json={"patient_id": "  "}).status_code == 422
    assert client_no_model.post("/patients", json={}).status_code == 422


def test_get_unknown_patient_404(client_no_model):
    assert client_no_model.get("/patients/NOPE").status_code == 404


def test_add_visit_requires_existing_patient(client_no_model):
    r = client_no_model.post("/patients/GHOST/visits", json={"visit_date": "2020-01-01"})
    assert r.status_code == 404


def test_visit_crud_and_history(client_no_model):
    client_no_model.post("/patients", json={"patient_id": "SYN_0001"})
    a = client_no_model.post(
        "/patients/SYN_0001/visits", json={"visit_date": "2020-06-01", "hba1c": 5.9}
    )
    b = client_no_model.post(
        "/patients/SYN_0001/visits", json={"visit_date": "2020-01-01", "hba1c": 5.6}
    )
    assert a.status_code == 201 and b.status_code == 201

    hist = client_no_model.get("/patients/SYN_0001").json()
    assert [v["visit_date"] for v in hist["visits"]] == ["2020-01-01", "2020-06-01"]

    # Update b.
    bid = b.json()["id"]
    up = client_no_model.put(
        f"/patients/SYN_0001/visits/{bid}", json={"visit_date": "2020-01-01", "hba1c": 6.0}
    )
    assert up.status_code == 200 and up.json()["hba1c"] == 6.0

    # Delete a.
    aid = a.json()["id"]
    assert client_no_model.delete(f"/patients/SYN_0001/visits/{aid}").status_code == 204
    hist2 = client_no_model.get("/patients/SYN_0001").json()
    assert len(hist2["visits"]) == 1


def test_visit_rejects_unknown_field(client_no_model):
    client_no_model.post("/patients", json={"patient_id": "SYN_0001"})
    r = client_no_model.post(
        "/patients/SYN_0001/visits",
        json={"visit_date": "2020-01-01", "target": 1},  # forbidden/unknown
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Ownership isolation
# ---------------------------------------------------------------------------

def test_cannot_touch_another_patients_visit(client_no_model):
    client_no_model.post("/patients", json={"patient_id": "SYN_0001"})
    client_no_model.post("/patients", json={"patient_id": "SYN_0002"})
    v = client_no_model.post(
        "/patients/SYN_0002/visits", json={"visit_date": "2021-01-01", "hba1c": 7.0}
    ).json()

    # Update/delete another patient's visit → 404 (not revealed).
    assert client_no_model.put(
        f"/patients/SYN_0001/visits/{v['id']}", json={"visit_date": "2021-01-01"}
    ).status_code == 404
    assert client_no_model.delete(f"/patients/SYN_0001/visits/{v['id']}").status_code == 404


# ---------------------------------------------------------------------------
# DB-backed selected-visit prediction
# ---------------------------------------------------------------------------

@SKIP_LGBM
def test_selected_visit_prediction(client_with_model):
    client_with_model.post("/patients", json={"patient_id": "SYN_0001"})
    v1 = client_with_model.post(
        "/patients/SYN_0001/visits", json={"visit_date": "2019-02-11", "hba1c": 5.6, "bmi": 27.4}
    ).json()
    v2 = client_with_model.post(
        "/patients/SYN_0001/visits", json={"visit_date": "2019-08-19", "hba1c": 5.8, "bmi": 27.9}
    ).json()

    r = client_with_model.post(
        "/patients/SYN_0001/predict", json={"visit_ids": [v2["id"], v1["id"]]}
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["patient_id"] == "SYN_0001"
    # Traced selection, chronological.
    assert body["selected_visit_ids"] == [v1["id"], v2["id"]]
    assert body["selected_visit_dates"] == ["2019-02-11", "2019-08-19"]
    assert body["n_visits"] == 2


@SKIP_LGBM
def test_prediction_rejects_cross_patient_visit(client_with_model):
    client_with_model.post("/patients", json={"patient_id": "SYN_0001"})
    client_with_model.post("/patients", json={"patient_id": "SYN_0002"})
    vx = client_with_model.post(
        "/patients/SYN_0002/visits", json={"visit_date": "2021-01-01", "hba1c": 7.0}
    ).json()
    r = client_with_model.post("/patients/SYN_0001/predict", json={"visit_ids": [vx["id"]]})
    assert r.status_code == 404


@SKIP_LGBM
def test_prediction_requires_at_least_one_visit(client_with_model):
    client_with_model.post("/patients", json={"patient_id": "SYN_0001"})
    r = client_with_model.post("/patients/SYN_0001/predict", json={"visit_ids": []})
    assert r.status_code == 422


@SKIP_LGBM
def test_prediction_enforces_observation_window(client_with_model):
    """Selecting more than observation_visits is rejected (no silent truncation)."""
    import json as _json

    # Read the model's observation window from /metadata.
    limit = client_with_model.get("/metadata").json().get("observation_visits")
    assert isinstance(limit, int) and limit >= 1

    client_with_model.post("/patients", json={"patient_id": "SYN_0001"})
    ids = []
    for i in range(limit + 1):
        vid = client_with_model.post(
            "/patients/SYN_0001/visits",
            json={"visit_date": f"2019-0{i + 1}-01" if i < 9 else f"2019-10-01", "hba1c": 5.5 + i * 0.1},
        ).json()["id"]
        ids.append(vid)

    r = client_with_model.post("/patients/SYN_0001/predict", json={"visit_ids": ids})
    assert r.status_code == 422
    assert "observation window" in _json.dumps(r.json()).lower()


# ---------------------------------------------------------------------------
# Mandatory regression: SYN_0001 -> SYN_0002 -> SYN_0001 (backend isolation)
# ---------------------------------------------------------------------------

def test_mandatory_switch_sequence_backend(tmp_path):
    """SYN_0001 → SYN_0002 → SYN_0001 with persistence across a restart."""
    db = tmp_path / "switch.db"

    # First app instance.
    app1 = create_app(artifact_dir=tmp_path / "empty", db_path=db)
    with TestClient(app1) as c:
        # A: open SYN_0001 (new, empty).
        a = c.post("/patients", json={"patient_id": "SYN_0001"}).json()
        assert a["created"] is True and a["visits"] == []
        # Give SYN_0001 one visit.
        c.post("/patients/SYN_0001/visits", json={"visit_date": "2018-01-01", "hba1c": 5.5})

        # B: switch to SYN_0002 (new, empty) — no SYN_0001 visits.
        b = c.post("/patients", json={"patient_id": "SYN_0002"}).json()
        assert b["visits"] == []

        # C: add two visits to SYN_0002.
        c.post("/patients/SYN_0002/visits", json={"visit_date": "2020-01-01", "hba1c": 6.8})
        c.post("/patients/SYN_0002/visits", json={"visit_date": "2020-06-01", "hba1c": 7.1})
        h2 = c.get("/patients/SYN_0002").json()
        assert len(h2["visits"]) == 2

        # D: switch back to SYN_0001 — original visit intact, no SYN_0002 data.
        h1 = c.get("/patients/SYN_0001").json()
        assert len(h1["visits"]) == 1
        assert h1["visits"][0]["visit_date"] == "2018-01-01"
        assert all(v["visit_date"] != "2020-01-01" for v in h1["visits"])

    # E: restart (new app, same DB file) — both histories persist independently.
    app2 = create_app(artifact_dir=tmp_path / "empty", db_path=db)
    with TestClient(app2) as c:
        assert len(c.get("/patients/SYN_0001").json()["visits"]) == 1
        assert len(c.get("/patients/SYN_0002").json()["visits"]) == 2


# ---------------------------------------------------------------------------
# Existing contracts preserved
# ---------------------------------------------------------------------------

def test_legacy_endpoints_preserved(client_no_model):
    assert client_no_model.get("/health").status_code == 200
    assert client_no_model.get("/metadata").status_code == 200
    # Legacy stateless /predict still present (503 without a model).
    r = client_no_model.post("/predict", json={"visits": [{"visit_date": "2020-01-01"}]})
    assert r.status_code == 503
