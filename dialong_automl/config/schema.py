"""Pydantic v2 schema for the DiaLong-AutoML YAML configuration.

All fields carry defaults so the schema is usable without a YAML file
(e.g. in unit tests).  Future phases extend the placeholder sections
rather than replacing them.

Phase 2 adds: SyntheticDataConfig, CohortConfig, SplitConfig.
Phase 5 adds: TrainingConfig, GRUConfig, OptunaConfig (extended).
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
# Optuna (Phase 5 — replaces placeholder)
# ---------------------------------------------------------------------------


#: Values the optimizer actually implements. Anything else is rejected explicitly
#: so that no OptunaConfig field looks functional while being ignored.
SUPPORTED_OPTUNA_SAMPLERS: tuple[str, ...] = ("TPESampler", "RandomSampler")
SUPPORTED_OPTUNA_PRUNERS: tuple[str, ...] = ("MedianPruner", "NopPruner")
SUPPORTED_OPTUNA_METRICS: tuple[str, ...] = ("roc_auc",)
SUPPORTED_OPTUNA_DIRECTIONS: tuple[str, ...] = ("maximize",)


class OptunaConfig(BaseModel):
    """Optuna AutoML search configuration.

    Every field is applied by ``AutoMLOptimizer``. Unsupported values are
    rejected at validation time (Phase 13B.1).
    """

    @field_validator("sampler")
    @classmethod
    def _validate_sampler(cls, v: str) -> str:
        if v not in SUPPORTED_OPTUNA_SAMPLERS:
            raise ValueError(
                f"optuna.sampler must be one of {SUPPORTED_OPTUNA_SAMPLERS}, got {v!r}"
            )
        return v

    @field_validator("pruner")
    @classmethod
    def _validate_pruner(cls, v: str) -> str:
        if v not in SUPPORTED_OPTUNA_PRUNERS:
            raise ValueError(
                f"optuna.pruner must be one of {SUPPORTED_OPTUNA_PRUNERS}, got {v!r}"
            )
        return v

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, v: str) -> str:
        if v not in SUPPORTED_OPTUNA_METRICS:
            raise ValueError(
                f"optuna.metric must be one of {SUPPORTED_OPTUNA_METRICS} "
                f"(the objective is validation AUROC), got {v!r}"
            )
        return v

    @field_validator("direction")
    @classmethod
    def _validate_direction(cls, v: str) -> str:
        if v not in SUPPORTED_OPTUNA_DIRECTIONS:
            raise ValueError(
                f"optuna.direction must be one of {SUPPORTED_OPTUNA_DIRECTIONS} "
                f"(best-trial selection maximises validation AUROC), got {v!r}"
            )
        return v

    @field_validator("n_jobs")
    @classmethod
    def _validate_n_jobs(cls, v: int) -> int:
        if v != 1:
            raise ValueError(
                "optuna.n_jobs must be 1: parallel trials are not supported "
                "(they would break seeded reproducibility and trial bookkeeping)."
            )
        return v

    n_trials: int = Field(5, description="Number of Optuna trials (demo default=5).")
    timeout_seconds: int | None = Field(
        None, description="Wall-clock timeout per study (None = unlimited)."
    )
    direction: str = Field("maximize", description="Optimisation direction: maximize or minimize.")
    metric: str = Field("roc_auc", description="Primary metric used for the objective.")
    pruner: str = Field(
        "MedianPruner", description="Optuna pruner class name."
    )
    sampler: str = Field(
        "TPESampler", description="Optuna sampler class name."
    )
    seed: int = Field(42, description="Seed for the TPE sampler.")
    n_jobs: int = Field(1, description="Parallel Optuna workers.")
    storage: str | None = Field(
        None,
        description="Optuna storage URL. E.g. 'sqlite:///artifacts/optuna_study.db'.",
    )
    study_name: str = Field(
        "dialong_automl_study",
        description=(
            "Base Optuna study name. The effective study name is "
            "'<study_name>__<identity hash>' where the hash covers the dataset "
            "content, cohort/split/search configuration and seeds, so unrelated "
            "experiments can never share trials (Phase 13B)."
        ),
    )
    run_id: str | None = Field(
        None,
        description=(
            "Optional run/fold label included in the study identity. Use a "
            "different value to start an independent study for an otherwise "
            "identical configuration (for example one per fold or repeat)."
        ),
    )
    resume: bool = Field(
        False,
        description=(
            "When False (default) a persisted study with the same identity must "
            "not already exist. When True the existing study is resumed and "
            "'n_trials' is interpreted as the TOTAL number of trials. Requires "
            "persistent 'storage'."
        ),
    )


# ---------------------------------------------------------------------------
# GRU configuration (Phase 5)
# ---------------------------------------------------------------------------


class GRUConfig(BaseModel):
    """GRU training configuration."""

    epochs: int = Field(30, description="Maximum training epochs per GRU trial.")
    patience: int = Field(5, description="Early-stopping patience (validation AUROC).")
    batch_size: int = Field(32, description="Default batch size (overridden by Optuna).")
    hidden_size: int = Field(64, description="Default hidden size (overridden by Optuna).")
    num_layers: int = Field(1, description="Default GRU layers (overridden by Optuna).")
    dropout: float = Field(0.0, description="Default dropout (overridden by Optuna).")
    learning_rate: float = Field(1e-3, description="Default LR (overridden by Optuna).")
    weight_decay: float = Field(1e-5, description="Default weight decay.")


# ---------------------------------------------------------------------------
# Training configuration (Phase 5)
# ---------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    """Top-level training configuration."""

    seed: int = Field(42, description="Master seed for training reproducibility.")
    threshold: float = Field(
        0.5, description="Default classification threshold for binary predictions."
    )
    max_iter_sklearn: int = Field(
        2000, description="Max iterations for sklearn iterative models (e.g. LogReg)."
    )
    threshold_strategy: str = Field(
        "fixed",
        description=(
            "How the classification threshold is chosen. "
            "'fixed' uses training.threshold. "
            "'validation_f1' maximises F1 on the validation set only."
        ),
    )

    @field_validator("threshold_strategy")
    @classmethod
    def _validate_threshold_strategy(cls, v: str) -> str:
        allowed = {"fixed", "validation_f1"}
        if v not in allowed:
            raise ValueError(f"threshold_strategy must be one of {allowed}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# API placeholder (Phase 6+)
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
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        description=(
            "Allowed CORS origins for the frontend. Defaults to the local Vite dev "
            "server. Never use a wildcard '*' default; set explicit origins per "
            "environment."
        ),
    )


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
    require_outcome_measurement: bool = Field(
        False,
        description=(
            "Phase 13D: exclude patients whose horizon has no measured HbA1c and no "
            "recorded diagnosis (outcome unascertained) instead of labelling them 0."
        ),
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
# Evaluation (Phase 11 — research foundation; opt-in, defaults preserve behavior)
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    """Optional settings for the Phase 11 research-evaluation utilities.

    These are **opt-in**. With defaults, nothing in the baseline training,
    selection, or metric set changes — the utilities are only used when a
    research run explicitly invokes them.
    """

    calibration_bins: int = Field(
        10, description="Number of equal-width bins for ECE/MCE/reliability curves."
    )
    stability_top_k: int = Field(
        5, description="Default k for top-k explanation-stability overlap."
    )
    enable_calibration_metrics: bool = Field(
        False, description="Reserved: compute calibration metrics in research runs."
    )
    enable_stability_metrics: bool = Field(
        False, description="Reserved: compute explanation-stability metrics in research runs."
    )


# ---------------------------------------------------------------------------
# Reference baselines (Phase 13C)
# ---------------------------------------------------------------------------


class BaselinesConfig(BaseModel):
    """Fixed, pre-specified reference baselines B0-B3 (``dialong_automl.baselines``).

    Baselines are reported next to the selected model and are NEVER Optuna
    candidates or selectable models. Their parameters are fixed here in
    advance and are not tuned on any data; only the decision threshold is
    chosen, on validation data, exactly like the selected model's.
    """

    enabled: bool = Field(
        True, description="Fit and report the reference baselines in every training run."
    )
    b2_horizon_days: int | None = Field(
        None,
        description=(
            "Extrapolation horizon for B2 in days. None = cohort.prediction_horizon_days."
        ),
    )
    b3_c: float = Field(
        1.0,
        description="Fixed inverse L2 regularisation strength (C) for the B3 regressions.",
    )

    @field_validator("b2_horizon_days")
    @classmethod
    def _validate_horizon(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"baselines.b2_horizon_days must be positive, got {v}")
        return v

    @field_validator("b3_c")
    @classmethod
    def _validate_c(cls, v: float) -> float:
        if not v > 0:
            raise ValueError(f"baselines.b3_c must be positive, got {v}")
        return v


# ---------------------------------------------------------------------------
# Prediction reliability (Phase 14; see docs/research/phase_14a_reliability_design.md)
# ---------------------------------------------------------------------------


class ReliabilityConfig(BaseModel):
    """Reliability layer for the served / evaluated model.

    14B implements post-hoc calibration only. Further fields (ensemble range,
    conformal sets, quality and OOD flags) are added by the phase that
    implements them, so no field here is ever ignored.
    """

    enabled: bool = Field(True, description="Fit and report the reliability layer.")
    calibration_method: str = Field(
        "platt", description="Post-hoc calibration: 'platt' (default), 'isotonic' or 'none'."
    )
    conformal_alpha: float = Field(
        0.10, gt=0, lt=1,
        description="Miscoverage of the class-conditional conformal sets (0.10 = 90 % sets).",
    )
    ensemble_members: int = Field(
        10, ge=0, description="Refitted members for the model-variability range "
                              "(nested CV only in 14C; 0 disables, otherwise >= 2).",
    )
    ensemble_subsample_fraction: float = Field(
        0.8, gt=0, le=1, description="Share of training patients per member (no replacement).",
    )

    ood_percentiles: tuple[float, float] = Field(
        (95.0, 99.0), description="OOD distance percentiles of training patients: unusual, far.")
    hba1c_age_days: tuple[int, int] = Field(
        (180, 365), description="Days since the last HbA1c for the 'old' and 'stale' flags.")
    width_percentiles: tuple[float, float] = Field(
        (80.0, 95.0), description="Variability-range width percentiles: Moderate, Low.")

    @field_validator("ood_percentiles", "hba1c_age_days", "width_percentiles")
    @classmethod
    def _validate_pair(cls, v):
        lo, hi = v
        if not 0 < lo < hi:
            raise ValueError("reliability cut-off pairs must satisfy 0 < low < high")
        if isinstance(lo, float) and hi >= 100:
            raise ValueError("percentile cut-offs must be below 100")
        return v

    @field_validator("ensemble_members")
    @classmethod
    def _validate_members(cls, v: int) -> int:
        if v == 1:
            raise ValueError("reliability.ensemble_members must be 0 or >= 2")
        return v

    @field_validator("calibration_method")
    @classmethod
    def _validate_method(cls, v: str) -> str:
        allowed = ("platt", "isotonic", "none")
        if v not in allowed:
            raise ValueError(f"reliability.calibration_method must be one of {allowed}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Unified evaluation runner (Phase 13E)
# ---------------------------------------------------------------------------


class NestedCVConfig(BaseModel):
    """Repeated, nested, patient-level cross-validation protocol (Phase 13E).

    Used only by ``dialong_automl.experiments.nested_cv`` /
    ``scripts/run_nested_cv.py``; the single-split ``TrainingPipeline`` ignores it.
    """

    scenario: str = Field("v2_standard", description="Synthetic scenario to generate.")
    n_patients: int = Field(5000, description="Patients generated before cohort exclusions.")
    dataset_seed: int = Field(42, description="Seed of the synthetic scenario.")
    oracle_draws: int = Field(32, description="Monte Carlo draws for the v2 latent-state oracle.")
    repeats: int = Field(2, description="Independent repetitions of the outer K-fold split.")
    outer_folds: int = Field(5, description="Outer stratified patient-level folds (test folds).")
    inner_folds: int = Field(3, description="Inner folds for selection, threshold and baselines.")
    n_trials: int = Field(20, description="Optuna trials per outer fold (inner-CV objective).")
    bootstrap_resamples: int = Field(1000, description="Patient-level bootstrap resamples.")
    bootstrap_seed: int = Field(2026, description="Seed of the bootstrap.")
    primary_comparator: str = Field(
        "b3_logreg_all", description="Baseline for the pre-specified primary comparison."
    )
    output_subdir: str = Field("nested_cv", description="Run folders go under outputs_dir/<this>.")

    @field_validator("scenario")
    @classmethod
    def _validate_scenario(cls, v: str) -> str:
        allowed = ("legacy_v1", "v2_standard")
        if v not in allowed:
            raise ValueError(f"nested_cv.scenario must be one of {allowed}, got {v!r}")
        return v

    @field_validator("outer_folds", "inner_folds")
    @classmethod
    def _validate_folds(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"fold counts must be >= 2, got {v}")
        return v

    @field_validator("repeats", "n_trials", "n_patients")
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    @field_validator("bootstrap_resamples", "oracle_draws")
    @classmethod
    def _validate_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"must be >= 0, got {v}")
        return v


# ---------------------------------------------------------------------------
# Database (Phase 10 — patient/visit persistence)
# ---------------------------------------------------------------------------


class DatabaseConfig(BaseModel):
    """SQLite patient-database configuration.

    The effective path is also overridable at runtime by the
    ``DIALONG_DB_PATH`` environment variable (see ``dialong_automl.db.engine``).
    """

    db_path: str = Field(
        "data/dialong_patients.db",
        description="Path to the SQLite patient database file.",
    )
    echo: bool = Field(False, description="Echo SQL statements (debug only).")


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
    # Phase 5 sections
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    gru: GRUConfig = Field(default_factory=GRUConfig)
    # Shared
    model: ModelConfig = Field(default_factory=ModelConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    # Phase 13C: reference baselines (not part of the Optuna study identity).
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    # Phase 13E: nested-CV evaluation protocol (not part of the Optuna study identity).
    nested_cv: NestedCVConfig = Field(default_factory=NestedCVConfig)
    # Phase 14: reliability layer (not part of the Optuna study identity).
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
