"""Phase 8 — Training CLI tests (dialong_automl.train).

Covers:
- Config path validation (missing file -> non-zero exit).
- Missing input data -> clear error, non-zero exit, no artifacts.
- End-to-end training via the CLI entry point produces the documented
  artifacts.
- The CLI reuses the existing TrainingPipeline (no second trainer).
- Reproducibility: two runs with the same seed select the same model and
  produce the same validation AUROC.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import dialong_automl.train as train_cli
from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator


def _lgbm_ok() -> bool:
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


SKIP_LGBM = pytest.mark.skipif(not _lgbm_ok(), reason="lightgbm unavailable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_csv(tmp_path: Path, n_patients: int = 80, seed: int = 42) -> Path:
    gen_cfg = GeneratorConfig(
        n_patients=n_patients,
        min_obs_visits=2,
        max_obs_visits=3,
        min_future_visits=2,
        max_future_visits=3,
        seed=seed,
        progression_prevalence=0.40,
        missingness_rate=0.0,
    )
    result = SyntheticGenerator(config=gen_cfg).generate()
    paths = result.save(output_dir=tmp_path / "data" / "synthetic")
    return paths["full_csv"]


def _tiny_config_yaml(tmp_path: Path) -> Path:
    """Write a minimal demo-style YAML that uses only a wide model."""
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
  study_name: "cli_test_study"
"""
    path = tmp_path / "cli_test.yaml"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Config / argument handling
# ---------------------------------------------------------------------------

def test_train_missing_config_returns_error(tmp_path):
    """A non-existent config path yields a non-zero exit code."""
    rc = train_cli.main(["--config", str(tmp_path / "nope.yaml")])
    assert rc == 1


@SKIP_LGBM
def test_train_missing_data_returns_error(tmp_path):
    """A valid config but missing dataset yields a clean non-zero exit."""
    cfg_path = _tiny_config_yaml(tmp_path)
    # No CSV generated and no default dataset in the tmp data dir.
    rc = train_cli.main(["--config", str(cfg_path), "--log-level", "WARNING"])
    assert rc == 1
    # No model artifact should have been written.
    assert not (tmp_path / "artifacts" / "best_model.joblib").exists()


# ---------------------------------------------------------------------------
# End-to-end training + artifacts
# ---------------------------------------------------------------------------

@SKIP_LGBM
def test_train_end_to_end_creates_artifacts(tmp_path):
    """Running the CLI produces the documented artifact set."""
    cfg_path = _tiny_config_yaml(tmp_path)
    csv = _tiny_csv(tmp_path)

    rc = train_cli.main(
        ["--config", str(cfg_path), "--data-csv", str(csv), "--log-level", "WARNING"]
    )
    assert rc == 0

    artifacts = tmp_path / "artifacts"
    outputs = tmp_path / "outputs"
    assert (artifacts / "best_model.joblib").exists()
    assert (artifacts / "preprocessor.joblib").exists()
    assert (artifacts / "model_metadata.json").exists()
    assert (outputs / "leaderboard.csv").exists()
    assert (outputs / "run_summary.json").exists()

    meta = json.loads((artifacts / "model_metadata.json").read_text(encoding="utf-8"))
    assert meta["selected_model"] == "lightgbm"
    assert meta["representation"] == "wide"
    assert meta["test_set_accessed_for_selection"] is False
    assert meta["synthetic_data"] is True


@SKIP_LGBM
def test_train_is_reproducible_with_seed(tmp_path):
    """Two identical-seed runs select the same model and val AUROC."""
    csv = _tiny_csv(tmp_path)

    cfg_a = _tiny_config_yaml(tmp_path)
    rc_a = train_cli.main(
        ["--config", str(cfg_a), "--data-csv", str(csv), "--log-level", "WARNING"]
    )
    assert rc_a == 0
    summary_a = json.loads(
        (tmp_path / "outputs" / "run_summary.json").read_text(encoding="utf-8")
    )

    # Second run into a fresh output location but same seed/data.
    tmp_b = tmp_path / "second"
    tmp_b.mkdir()
    content = (tmp_path / "cli_test.yaml").read_text(encoding="utf-8")
    content = content.replace(
        (tmp_path / "artifacts").as_posix(), (tmp_b / "artifacts").as_posix()
    ).replace(
        (tmp_path / "outputs").as_posix(), (tmp_b / "outputs").as_posix()
    )
    cfg_b = tmp_b / "cli_test.yaml"
    cfg_b.write_text(content, encoding="utf-8")

    rc_b = train_cli.main(
        ["--config", str(cfg_b), "--data-csv", str(csv), "--log-level", "WARNING"]
    )
    assert rc_b == 0
    summary_b = json.loads(
        (tmp_b / "outputs" / "run_summary.json").read_text(encoding="utf-8")
    )

    assert summary_a["selected_model"] == summary_b["selected_model"]
    assert summary_a["validation_auroc_best"] == summary_b["validation_auroc_best"]
