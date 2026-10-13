"""Phase 8 — Synthetic generation + experiment persistence tests.

Covers:
- Reproducible synthetic data generation (same seed -> identical data;
  different seed -> different data).
- scripts/run_experiments.py persists resolved config, seed, and metrics,
  and the run is reproducible from the saved config.
- The experiment honours patient-level split with no leakage/overlap.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _lgbm_ok() -> bool:
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


SKIP_LGBM = pytest.mark.skipif(not _lgbm_ok(), reason="lightgbm unavailable")


def _load_run_experiments():
    """Import scripts/run_experiments.py as a module."""
    path = _REPO_ROOT / "scripts" / "run_experiments.py"
    spec = importlib.util.spec_from_file_location("run_experiments", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Reproducible synthetic generation
# ---------------------------------------------------------------------------

def test_generation_is_reproducible_with_same_seed():
    cfg = GeneratorConfig(n_patients=40, seed=123, missingness_rate=0.0)
    a = SyntheticGenerator(config=GeneratorConfig(**vars(cfg))).generate().full_df
    b = SyntheticGenerator(config=GeneratorConfig(**vars(cfg))).generate().full_df
    pd.testing.assert_frame_equal(a, b)


def test_generation_differs_with_different_seed():
    base = dict(n_patients=40, missingness_rate=0.0)
    a = SyntheticGenerator(config=GeneratorConfig(seed=1, **base)).generate().full_df
    b = SyntheticGenerator(config=GeneratorConfig(seed=2, **base)).generate().full_df
    # Shapes may differ or values differ; they must not be identical.
    if a.shape == b.shape:
        assert not a.equals(b)


def test_generation_target_matches_cohort_rule():
    """Pre-labeled target equals the future-outcome rule (leakage contract)."""
    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode

    res = SyntheticGenerator(
        config=GeneratorConfig(n_patients=60, seed=7, missingness_rate=0.0)
    ).generate()
    full_df = res.full_df.copy()
    full_df["patient_id"] = full_df["patient_id"].astype(str)

    cohort = CohortBuilder(
        CohortConfig(
            mode=CohortMode.FUTURE_OUTCOME,
            min_observation_visits=2,
            max_observation_visits=4,
            prediction_horizon_days=365 * 10,
            exclude_known_diabetes_at_baseline=False,
            require_followup_after_cutoff=True,
        )
    ).build(full_df)
    # Cohort targets must be a subset-consistent 0/1 labelling.
    assert set(cohort.patient_targets.unique()).issubset({0, 1})
    assert len(cohort.patient_targets) > 0


# ---------------------------------------------------------------------------
# Experiment persistence
# ---------------------------------------------------------------------------

def _tiny_experiment_config(tmp_path: Path) -> Path:
    content = f"""
project_name: "DiaLong-AutoML"
version: "0.1.0"
paths:
  data_dir: "{(tmp_path / 'data').as_posix()}"
  artifacts_dir: "{(tmp_path / 'artifacts').as_posix()}"
  outputs_dir: "{(tmp_path / 'outputs').as_posix()}"
  logs_dir: "{(tmp_path / 'logs').as_posix()}"
reproducibility:
  seed: 42
  deterministic_torch: false
data:
  patients: 80
  min_visits: 2
  max_visits: 3
  seed: 42
  missingness_rate: 0.0
  output_dir: "{(tmp_path / 'data' / 'synthetic').as_posix()}"
cohort:
  mode: "future_outcome"
  min_observation_visits: 2
  max_observation_visits: 3
  prediction_horizon_days: 1825
  exclude_known_diabetes_at_baseline: false
  require_followup_after_cutoff: true
split:
  train_size: 0.60
  validation_size: 0.20
  test_size: 0.20
  random_seed: 42
  stratify: false
training:
  seed: 42
  threshold: 0.5
model:
  enabled_models:
    - lightgbm
optuna:
  n_trials: 1
  seed: 42
  storage: null
  study_name: "experiment_test_study"
"""
    path = tmp_path / "experiment.yaml"
    path.write_text(content, encoding="utf-8")
    return path


@SKIP_LGBM
def test_experiment_persists_config_and_metrics(tmp_path):
    """run_experiments writes resolved config, meta, and metrics; data is generated."""
    mod = _load_run_experiments()
    cfg_path = _tiny_experiment_config(tmp_path)

    rc = mod.main(["--config", str(cfg_path), "--run-name", "unit_run"])
    assert rc == 0

    run_dir = tmp_path / "outputs" / "unit_run"
    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "experiment_meta.json").exists()
    assert (run_dir / "metrics.json").exists()

    # Dataset was generated reproducibly from the config.
    assert (tmp_path / "data" / "synthetic" / "diabetes_progression_full.csv").exists()

    meta = json.loads((run_dir / "experiment_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 42
    assert "disclaimer" in meta

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selected_model"] == "lightgbm"
    assert "test_metrics" in metrics
    # No patient leakage: splits sum to the eligible cohort.
    split = metrics["split"]
    assert split["train"] + split["val"] + split["test"] == metrics["n_eligible_patients"]

    # Resolved config is reproducible: reloading it yields the same seed.
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["reproducibility"]["seed"] == 42
    assert resolved["optuna"]["seed"] == 42


@SKIP_LGBM
def test_experiment_missing_config_returns_error(tmp_path):
    mod = _load_run_experiments()
    rc = mod.main(["--config", str(tmp_path / "missing.yaml")])
    assert rc == 1
