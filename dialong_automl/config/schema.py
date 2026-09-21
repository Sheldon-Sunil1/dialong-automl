"""Pydantic v2 schema for the DiaLong-AutoML YAML configuration.

All fields carry defaults so the schema is usable without a YAML file
(e.g. in unit tests).  Future phases extend the placeholder sections
rather than replacing them.

Phase 2 adds: SyntheticDataConfig, CohortConfig, SplitConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


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
# Synthetic data configuration (Phase 2)
# ---------------------------------------------------------------------------


class SyntheticDataConfig(BaseModel):
    """Configuration for the synthetic longitudinal data generator."""

    patients: int = Field(1000, description="Number of synthetic patients to generate.")
    min_visits: int = Field(4, description="Minimum observation visits per patient.")
    max_visits: int = Field(6, description="Maximum observation visits per patient.")
    min_future_visits: int = Field(
        2, description="Minimum future visits generated (used for outcome only)."
    )
    max_future_visits: int = Field(
        4, description="Maximum future visits generated (used for outcome only)."
    )
    min_days_between_visits: int = Field(
        90, description="Minimum inter-visit interval in days."
    )
    max_days_between_visits: int = Field(
        240, description="Maximum inter-visit interval in days."
    )
    missingness_rate: float = Field(
        0.05, description="Fraction of numeric values to set as missing."
    )
    progression_prevalence: float = Field(
        0.30, description="Approximate fraction of patients who progress to diabetes."
    )
    seed: int = Field(42, description="Random seed for reproducible generation.")
    output_dir: str = Field(
        "data/synthetic", description="Output directory for generated CSV and metadata."
    )


# ---------------------------------------------------------------------------
# Cohort configuration (Phase 3)
# ---------------------------------------------------------------------------


class CohortBuilderConfig(BaseModel):
    """Configuration for cohort construction (Phase 3)."""

    mode: str = Field(
        "future_outcome",
        description="Cohort mode: 'future_outcome' or 'pre_labeled'.",
    )
    min_observation_visits: int = Field(
        3, description="Minimum observation visits per patient."
    )
    max_observation_visits: int = Field(
        4, description="Maximum observation visits used per patient."
    )
    prediction_horizon_days: int = Field(
        365, description="Days after cutoff in which future outcomes count."
    )
    diabetes_onset_hba1c_threshold: float = Field(
        6.5,
        description="HbA1c threshold (%) for future diabetes onset classification.",
    )
    exclude_known_diabetes_at_baseline: bool = Field(
        True,
        description="Exclude patients already diabetic in the observation window.",
    )
    require_followup_after_cutoff: bool = Field(
        True,
        description="Exclude patients with no future visits beyond the cutoff.",
    )

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        allowed = {"future_outcome", "pre_labeled"}
        if v not in allowed:
            raise ValueError(f"cohort mode must be one of {allowed}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Split configuration (Phase 3)
# ---------------------------------------------------------------------------


class SplitterConfig(BaseModel):
    """Configuration for patient-level train/val/test splitting (Phase 3)."""

    train_size: float = Field(0.70, description="Fraction of patients for training.")
    validation_size: float = Field(0.15, description="Fraction for validation.")
    test_size: float = Field(0.15, description="Fraction for testing.")
    random_seed: int = Field(42, description="Random seed for the split.")
    stratify: bool = Field(True, description="Stratify by patient-level target.")
    chronological: bool = Field(
        False,
        description="Split in chronological order by first visit date.",
    )

    @model_validator(mode="after")
    def _validate_sizes(self) -> "SplitterConfig":
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_size + validation_size + test_size must equal 1.0, got {total:.6f}."
            )
        return self


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
    # Phase 2/3 sections
    data: SyntheticDataConfig = Field(default_factory=SyntheticDataConfig)
    cohort: CohortBuilderConfig = Field(default_factory=CohortBuilderConfig)
    split: SplitterConfig = Field(default_factory=SplitterConfig)
    # Later phases
    model: ModelConfig = Field(default_factory=ModelConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    api: APIConfig = Field(default_factory=APIConfig)
