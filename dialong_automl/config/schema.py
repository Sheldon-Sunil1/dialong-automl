"""Pydantic v2 schema for the DiaLong-AutoML YAML configuration.

All fields carry defaults so the schema is usable without a YAML file
(e.g. in unit tests).  Future phases extend the placeholder sections
rather than replacing them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


class PathsConfig(BaseModel):
    """Filesystem paths used across the project."""

    data_dir: Path = Field(Path("data"), description="Root directory for raw/processed data.")
    artifacts_dir: Path = Field(
        Path("artifacts"), description="Trained model artefacts and checkpoints."
    )
    outputs_dir: Path = Field(
        Path("outputs"), description="Evaluation results, plots, and reports."
    )
    logs_dir: Path = Field(Path("logs"), description="Application and experiment log files.")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field("INFO", description="Root log level (DEBUG/INFO/WARNING/ERROR/CRITICAL).")
    format: str = Field(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Python logging format string.",
    )
    log_to_file: bool = Field(False, description="Whether to write logs to a rotating file.")
    log_filename: str = Field(
        "dialong_automl.log", description="Filename inside paths.logs_dir when log_to_file=True."
    )
    max_bytes: int = Field(10_485_760, description="Max bytes per log file (10 MiB default).")
    backup_count: int = Field(3, description="Number of rotated log files to retain.")

    @field_validator("level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log level must be one of {allowed}, got {v!r}")
        return upper


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class ReproducibilityConfig(BaseModel):
    """Random-seed and determinism settings."""

    seed: int = Field(42, description="Global random seed for NumPy, Python random, and PyTorch.")
    deterministic_torch: bool = Field(
        True,
        description=(
            "Set torch.use_deterministic_algorithms(True). "
            "May reduce performance; disable if not needed."
        ),
    )
    benchmark_cudnn: bool = Field(
        False,
        description="Set torch.backends.cudnn.benchmark. Keep False for reproducibility.",
    )


# ---------------------------------------------------------------------------
# Model placeholders (Phase 2+)
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Placeholder model configuration — populated in later phases."""

    enabled_models: list[str] = Field(
        default_factory=lambda: [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
            "gru",
        ],
        description=(
            "Models to include in the AutoML search. "
            "'tcn' is planned but not yet implemented."
        ),
    )
    # Per-model hyperparameter overrides will be nested dicts in later phases.
    hyperparameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional per-model fixed hyperparameters (bypasses Optuna for that model).",
    )


# ---------------------------------------------------------------------------
# Optuna placeholder (Phase 3+)
# ---------------------------------------------------------------------------


class OptunaConfig(BaseModel):
    """Placeholder Optuna configuration — populated in later phases."""

    n_trials: int = Field(50, description="Number of Optuna trials per model.")
    timeout_seconds: int | None = Field(
        None, description="Wall-clock timeout per Optuna study (None = unlimited)."
    )
    direction: str = Field("maximize", description="Optimisation direction: maximize or minimize.")
    metric: str = Field("roc_auc", description="Primary evaluation metric for Optuna.")
    pruner: str = Field(
        "MedianPruner", description="Optuna pruner class name (MedianPruner, HyperbandPruner, …)."
    )
    sampler: str = Field(
        "TPESampler", description="Optuna sampler class name (TPESampler, CmaEsSampler, …)."
    )
    n_jobs: int = Field(1, description="Parallel Optuna workers (-1 = all cores).")
    storage: str | None = Field(
        None,
        description="Optuna storage URL (None = in-memory). E.g. 'sqlite:///optuna.db'.",
    )


# ---------------------------------------------------------------------------
# API placeholder (Phase 5+)
# ---------------------------------------------------------------------------


class APIConfig(BaseModel):
    """Placeholder FastAPI configuration — populated in later phases."""

    host: str = Field("0.0.0.0", description="Bind host for the FastAPI server.")
    port: int = Field(8000, description="Bind port for the FastAPI server.")
    reload: bool = Field(False, description="Enable uvicorn auto-reload (development only).")
    workers: int = Field(1, description="Number of uvicorn worker processes.")
    title: str = Field("DiaLong-AutoML API", description="OpenAPI title.")
    version: str = Field("0.1.0", description="API version string.")
    docs_url: str = Field("/docs", description="Swagger UI endpoint path.")
    redoc_url: str = Field("/redoc", description="ReDoc endpoint path.")


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Top-level application configuration loaded from YAML."""

    project_name: str = Field("DiaLong-AutoML", description="Human-readable project name.")
    version: str = Field("0.1.0", description="Project version string.")
    description: str = Field(
        "Explainable Longitudinal AutoML for Diabetes Progression Prediction",
        description="Short project description.",
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    api: APIConfig = Field(default_factory=APIConfig)
