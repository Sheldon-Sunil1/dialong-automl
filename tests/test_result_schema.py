"""Phase 11 — versioned experiment-result schema tests."""

from __future__ import annotations

import json

import pytest

from dialong_automl.evaluation.result_schema import (
    SCHEMA_VERSION,
    ExperimentCondition,
    RunResult,
    config_hash,
    new_run_id,
)


def _minimal() -> RunResult:
    return RunResult(
        run_id=new_run_id(),
        config_hash=config_hash({"a": 1, "b": 2}),
        seeds={"global": 42, "optuna": 42},
        dataset_id="synthetic_v3",
        split_id="70_15_15_seed42",
        condition=ExperimentCondition(kind="baseline"),
        metrics={"auroc": 0.95, "ece": 0.08, "spearman": None},
    )


def test_round_trip_serialization():
    r = _minimal()
    d = r.to_dict()
    r2 = RunResult.from_dict(d)
    assert r2.run_id == r.run_id
    assert r2.metrics == r.metrics
    assert r2.condition.kind == "baseline"
    assert r2.schema_version == SCHEMA_VERSION


def test_deterministic_json():
    r = _minimal()
    assert r.to_json() == r.to_json()  # stable (sort_keys)


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_run_id_unique():
    assert new_run_id() != new_run_id()


def test_missing_optional_fields_default_explicitly():
    r = RunResult(run_id="r1")
    d = r.to_dict()
    # Optional fields present and explicit (None), not omitted.
    assert d["config_hash"] is None
    assert d["dataset_id"] is None
    assert d["split_id"] is None
    assert d["seeds"] == {}
    assert d["condition"]["kind"] == "baseline"
    assert d["condition"]["strength"] is None


def test_none_metrics_allowed_and_preserved():
    r = RunResult(run_id="r1", metrics={"auroc": None})
    assert r.to_dict()["metrics"]["auroc"] is None


def test_validation_rejects_bad_schema_version():
    r = RunResult(run_id="r1")
    r.schema_version = "9.9"
    with pytest.raises(ValueError):
        r.validate()


def test_validation_rejects_non_numeric_metric():
    r = RunResult(run_id="r1", metrics={"auroc": "high"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        r.validate()


def test_validation_rejects_empty_run_id():
    with pytest.raises(ValueError):
        RunResult(run_id="").validate()


def test_perturbed_condition_captured():
    r = RunResult(
        run_id="r1",
        condition=ExperimentCondition(
            kind="remove_measurements", strength=0.2, params={"seed": 1}
        ),
        metrics={"auroc": 0.80},
    )
    d = r.to_dict()
    assert d["condition"]["kind"] == "remove_measurements"
    assert d["condition"]["strength"] == 0.2
    assert d["condition"]["params"]["seed"] == 1


def test_save_and_load(tmp_path):
    r = _minimal()
    p = tmp_path / "run.json"
    r.save(p)
    loaded = RunResult.load(p)
    assert loaded.run_id == r.run_id
    assert loaded.metrics == r.metrics
    # File is valid JSON with sorted keys.
    with p.open() as fh:
        raw = json.load(fh)
    assert raw["schema_version"] == SCHEMA_VERSION


def test_no_patient_identifiers_in_schema_fields():
    """The schema has no field intended for patient-identifying data."""
    fields = set(RunResult(run_id="r1").to_dict().keys())
    assert "patient_id" not in fields
    assert "patients" not in fields
    assert "visits" not in fields
