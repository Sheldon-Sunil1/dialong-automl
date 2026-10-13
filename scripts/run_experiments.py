#!/usr/bin/env python
"""Run a reproducible DiaLong-AutoML research experiment.

Reuses the existing training pipeline (``TrainingPipeline``) and AutoML
optimizer — it does not reimplement training. The script:

1. Loads a YAML config (defaults to ``configs/research_experiment.yaml``).
2. Ensures a synthetic dataset exists (generates one reproducibly from the
   config's ``data`` section + seed when absent).
3. Runs the full training/AutoML pipeline (preprocessing fitted on training
   data only; test set evaluated once; no leakage).
4. Saves the resolved configuration, the random seed, and all metrics into a
   timestamped run directory so the experiment can be reproduced from the
   saved config + seed.

DISCLAIMER: Synthetic demonstration data only. Results are NOT clinical
evidence and do NOT establish clinical validity.

Usage::

    python scripts/run_experiments.py --config configs/research_experiment.yaml
    python scripts/run_experiments.py --config configs/demo_fast.yaml --patients 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the package importable when run directly from the repo root.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_experiments")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible DiaLong-AutoML research experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/research_experiment.yaml",
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=None,
        dest="data_csv",
        help=(
            "Full-timeline CSV to train on. Defaults to "
            "<data_dir>/synthetic/diabetes_progression_full.csv. "
            "Generated from the config if absent."
        ),
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=None,
        help="Override the number of synthetic patients to generate.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        dest="run_name",
        help="Optional run directory name. Defaults to a UTC timestamp.",
    )
    return parser.parse_args(argv)


def ensure_dataset(cfg, data_csv: str | None, patients_override: int | None) -> Path:
    """Return a full-timeline CSV path, generating it reproducibly if absent."""
    from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator

    if data_csv:
        csv_path = Path(data_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"--data-csv provided but not found: {csv_path}")
        logger.info("Using existing dataset: %s", csv_path)
        return csv_path

    synth_dir = Path(cfg.paths.data_dir) / "synthetic"
    csv_path = synth_dir / "diabetes_progression_full.csv"
    if csv_path.exists():
        logger.info("Found existing dataset: %s", csv_path)
        return csv_path

    n_patients = patients_override or cfg.data.patients
    logger.info(
        "Generating synthetic dataset (%d patients, seed=%d) -> %s",
        n_patients, cfg.data.seed, synth_dir,
    )
    gen_cfg = GeneratorConfig(
        n_patients=n_patients,
        min_obs_visits=cfg.data.min_visits,
        max_obs_visits=cfg.data.max_visits,
        min_future_visits=cfg.data.min_future_visits,
        max_future_visits=cfg.data.max_future_visits,
        min_days_between_visits=cfg.data.min_days_between_visits,
        max_days_between_visits=cfg.data.max_days_between_visits,
        progression_prevalence=cfg.data.progression_prevalence,
        missingness_rate=cfg.data.missingness_rate,
        seed=cfg.data.seed,
        diabetes_onset_hba1c_threshold=cfg.cohort.diabetes_onset_hba1c_threshold,
    )
    result = SyntheticGenerator(config=gen_cfg).generate()
    paths = result.save(output_dir=synth_dir)
    return paths["full_csv"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from dialong_automl.config import load_config
    from dialong_automl.utils.logging_utils import configure_logging
    from dialong_automl.utils.seed import set_global_seed
    from dialong_automl.training.pipeline import TrainingPipeline

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _repo_root / config_path
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    cfg = load_config(config_path)
    configure_logging(cfg.logging, logs_dir=Path(cfg.paths.logs_dir), force=True)

    # Honour reproducibility settings explicitly (the pipeline also seeds).
    set_global_seed(
        cfg.reproducibility.seed,
        deterministic_torch=cfg.reproducibility.deterministic_torch,
        benchmark_cudnn=cfg.reproducibility.benchmark_cudnn,
    )

    logger.info("=" * 60)
    logger.info("DiaLong-AutoML Research Experiment")
    logger.info("DISCLAIMER: Synthetic demonstration data only.")
    logger.info("  Not clinical evidence. Does not establish clinical validity.")
    logger.info("=" * 60)

    try:
        csv_path = ensure_dataset(cfg, args.data_csv, args.patients)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # Per-run output directory for reproducibility artifacts.
    run_name = args.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = Path(cfg.paths.outputs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the resolved config + seed BEFORE training so a crashed run is
    # still reproducible.
    resolved_config = cfg.model_dump(mode="json")
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as fh:
        json.dump(resolved_config, fh, indent=2, default=str)
    with (run_dir / "experiment_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config_path": str(config_path),
                "dataset_csv": str(csv_path),
                "seed": cfg.reproducibility.seed,
                "training_seed": cfg.training.seed,
                "optuna_seed": cfg.optuna.seed,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "disclaimer": (
                    "Synthetic demonstration data only. Not clinical evidence. "
                    "Does not establish clinical validity."
                ),
            },
            fh, indent=2, default=str,
        )

    pipeline = TrainingPipeline(cfg, full_csv_path=csv_path)
    try:
        result = pipeline.run()
    except Exception as exc:  # noqa: BLE001 — surface a clean experiment failure
        logger.exception("Experiment failed: %s", exc)
        return 1

    # Persist metrics for this run (alongside the pipeline's own artifacts).
    metrics_payload = {
        "selected_model": result.best_model_name,
        "representation": result.best_representation,
        "best_hyperparams": result.best_hyperparams,
        "validation_auroc_best": result.best_val_auroc,
        "validation_metrics": {
            k: v for k, v in result.val_metrics.items() if not isinstance(v, dict)
        },
        "test_metrics": {
            k: v for k, v in result.test_metrics.items() if not isinstance(v, dict)
        },
        "split": {"train": result.n_train, "val": result.n_val, "test": result.n_test},
        "n_eligible_patients": result.n_eligible_patients,
        "positive_rate": result.positive_rate,
        "n_trials_completed": result.n_trials_completed,
        "total_run_seconds": result.total_run_seconds,
        "artifact_paths": result.artifact_paths,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, default=str)

    print("\n" + result.summary())
    print("\nExperiment run directory:")
    print(f"  {run_dir}")
    print("  - resolved_config.json  (reproduce with this config + seed)")
    print("  - experiment_meta.json")
    print("  - metrics.json")

    logger.info("Experiment complete. Reproducibility artifacts in %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
