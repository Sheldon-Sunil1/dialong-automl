"""Phase 14E: reliability in predict.py and the API. Synthetic data only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dialong_automl.api.app import create_app
from dialong_automl.predict import PredictionError, predict_from_artifacts
from tests.test_predict import _example_visits, _lgbm_ok, _tiny_cfg, _tiny_full_csv


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    if not _lgbm_ok():
        pytest.skip("lightgbm unavailable")
    from dialong_automl.training.pipeline import TrainingPipeline

    tmp = tmp_path_factory.mktemp("serve")
    cfg = _tiny_cfg(tmp)
    cfg.reliability.ensemble_members = 3
    TrainingPipeline(cfg, full_csv_path=_tiny_full_csv(tmp)).run()
    return Path(cfg.paths.artifacts_dir)


def test_pipeline_writes_served_components(artifacts):
    m = json.loads((artifacts / "reliability.json").read_text())
    assert len(m["members"]) == 3 and all((artifacts / f).exists() for f in m["members"])
    for k in ("calibration", "conformal", "ood"):
        assert (artifacts / m[k]).exists()
    lo, hi = m["width_cutoffs"]
    assert 0 <= lo <= hi
    meta = json.loads((artifacts / "model_metadata.json").read_text())
    assert meta["reliability"]["served_components"]["members"] == m["members"]


def test_predict_returns_reliability_and_keeps_raw_fields(artifacts):
    r = predict_from_artifacts(artifacts, "P1", _example_visits())
    rel = r["reliability"]
    assert rel["level"] in {"Low", "Moderate", "High"}
    assert rel["prediction_set"] in {"no_onset", "onset", "undecided", "empty"}
    assert rel["range_low"] <= rel["range_high"]
    assert 0 <= rel["calibrated_probability"] <= 1
    assert 0 <= rel["ood_percentile"] <= 100
    # Decision rule unchanged: raw probability vs raw threshold.
    assert r["prediction"] == int(r["probability"] >= r["threshold"])
    assert (rel["calibrated_probability"] >= rel["calibrated_threshold"]) == bool(r["prediction"])
    # Two visits < 3-visit window -> info flag, listed but never a reason.
    codes = {f["code"]: f["severity"] for f in rel["flags"]}
    assert codes.get("short_history") == "info"
    assert all("visits were available" not in s for s in rel["reasons"])
    assert json.dumps(r)                                    # JSON-safe


def test_missing_hba1c_makes_the_prediction_low(artifacts):
    visits = [{k: v for k, v in v_.items() if k != "hba1c"} for v_ in _example_visits()]
    rel = predict_from_artifacts(artifacts, "P2", visits)["reliability"]
    assert rel["level"] == "Low"
    assert any(f["code"] == "no_hba1c" for f in rel["flags"])


def test_old_artifacts_return_null_and_corrupt_ones_fail(artifacts, tmp_path):
    old = tmp_path / "old"
    shutil.copytree(artifacts, old)
    (old / "reliability.json").unlink()
    r = predict_from_artifacts(old, "P1", _example_visits())
    assert r["reliability"] is None
    assert r["probability"] == predict_from_artifacts(artifacts, "P1", _example_visits())[
        "probability"]
    bad = tmp_path / "bad"
    shutil.copytree(artifacts, bad)
    (bad / "ood.json").unlink()
    with pytest.raises(PredictionError, match="Reliability"):
        predict_from_artifacts(bad, "P1", _example_visits())


def test_api_contract(artifacts, tmp_path):
    with TestClient(create_app(artifact_dir=artifacts)) as c:
        body = c.post("/predict", json={"patient_id": "P1", "visits": _example_visits()}).json()
    assert body["reliability"]["level"] in {"Low", "Moderate", "High"}
    assert body["reliability"]["note"] and body["reliability"]["range_note"]
    old = tmp_path / "old_api"
    shutil.copytree(artifacts, old)
    (old / "reliability.json").unlink()
    with TestClient(create_app(artifact_dir=old)) as c:
        r = c.post("/predict", json={"patient_id": "P1", "visits": _example_visits()})
    assert r.status_code == 200 and r.json()["reliability"] is None


def test_gru_members_are_saved_and_scored(tmp_path):
    pytest.importorskip("torch")
    from dialong_automl.training.pipeline import TrainingPipeline
    from tests.test_phase13b1_correctness import _gru_cfg
    from tests.test_phase13b1_correctness import _tiny_full_csv as gru_csv

    cfg = _gru_cfg(tmp_path, "gru")
    cfg.reliability.ensemble_members = 2
    TrainingPipeline(cfg, full_csv_path=gru_csv(tmp_path)).run()
    adir = Path(cfg.paths.artifacts_dir)
    m = json.loads((adir / "reliability.json").read_text())
    assert [Path(f).suffix for f in m["members"]] == [".pt", ".pt"]
    rel = predict_from_artifacts(adir, "P1", _example_visits())["reliability"]
    assert rel["range_low"] is not None and rel["range_low"] <= rel["range_high"]
