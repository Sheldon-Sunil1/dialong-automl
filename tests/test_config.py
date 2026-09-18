"""Tests 2 & 3 — YAML configuration loading and validation.

Covers:
- Loading the bundled demo_fast.yaml
- Loading from defaults (no file)
- Field values and types
- Validation errors on bad input
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dialong_automl.config import AppConfig, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repository root (parent of the tests/ directory)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Test: load from defaults (no YAML file)
# ---------------------------------------------------------------------------


def test_load_defaults_no_file() -> None:
    """load_config(None) returns a valid AppConfig with expected defaults."""
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert cfg.project_name == "DiaLong-AutoML"
    assert cfg.version == "0.1.0"
    assert cfg.reproducibility.seed == 42


def test_default_paths() -> None:
    """Default path config fields are Path instances."""
    cfg = load_config(None)
    assert isinstance(cfg.paths.data_dir, Path)
    assert isinstance(cfg.paths.artifacts_dir, Path)
    assert isinstance(cfg.paths.outputs_dir, Path)
    assert isinstance(cfg.paths.logs_dir, Path)


def test_default_log_level() -> None:
    """Default log level is INFO."""
    cfg = load_config(None)
    assert cfg.logging.level == "INFO"


def test_default_optuna_trials() -> None:
    """Default Optuna n_trials is 50."""
    cfg = load_config(None)
    assert cfg.optuna.n_trials == 50


def test_default_api_port() -> None:
    """Default API port is 8000."""
    cfg = load_config(None)
    assert cfg.api.port == 8000


# ---------------------------------------------------------------------------
# Test: load from demo_fast.yaml
# ---------------------------------------------------------------------------


def test_load_demo_fast_yaml() -> None:
    """demo_fast.yaml loads and validates without errors."""
    yaml_path = _repo_root() / "configs" / "demo_fast.yaml"
    assert yaml_path.exists(), f"demo_fast.yaml not found at {yaml_path}"
    cfg = load_config(yaml_path)
    assert isinstance(cfg, AppConfig)


def test_demo_fast_seed() -> None:
    """demo_fast.yaml uses seed 42."""
    cfg = load_config(_repo_root() / "configs" / "demo_fast.yaml")
    assert cfg.reproducibility.seed == 42


def test_demo_fast_optuna_trials() -> None:
    """demo_fast.yaml sets n_trials=10 (fast preset)."""
    cfg = load_config(_repo_root() / "configs" / "demo_fast.yaml")
    assert cfg.optuna.n_trials == 10


def test_demo_fast_enabled_models() -> None:
    """demo_fast.yaml lists exactly the expected enabled models."""
    cfg = load_config(_repo_root() / "configs" / "demo_fast.yaml")
    assert set(cfg.model.enabled_models) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "gru",
    }


def test_demo_fast_tcn_not_enabled() -> None:
    """TCN must NOT appear in demo_fast.yaml enabled_models."""
    cfg = load_config(_repo_root() / "configs" / "demo_fast.yaml")
    assert "tcn" not in cfg.model.enabled_models


# ---------------------------------------------------------------------------
# Test: validation errors on bad input
# ---------------------------------------------------------------------------


def test_invalid_log_level_raises() -> None:
    """Providing an invalid log level raises ValidationError."""
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"logging": {"level": "VERBOSE"}})


def test_invalid_seed_type_raises() -> None:
    """Providing a non-integer seed raises ValidationError."""
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"reproducibility": {"seed": "not-an-int"}})


def test_missing_config_file_raises() -> None:
    """load_config with a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_appconfig_direct_construction() -> None:
    """AppConfig can be constructed directly with keyword arguments."""
    cfg = AppConfig(project_name="Test", version="9.9.9")
    assert cfg.project_name == "Test"
    assert cfg.version == "9.9.9"
    # Nested defaults still apply
    assert cfg.reproducibility.seed == 42
