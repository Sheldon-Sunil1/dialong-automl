"""DiaLong-AutoML training CLI entry point.

Usage::

    python -m dialong_automl.train --config configs/demo_fast.yaml
    python -m dialong_automl.train --config configs/demo_fast.yaml --data-csv data/synthetic/diabetes_progression_full.csv

The CLI:
1. Loads the YAML config.
2. Sets up logging.
3. Creates required directories.
4. Delegates to TrainingPipeline.run().
5. Prints the PipelineResult summary to stdout.

DISCLAIMER: Operates on synthetic demonstration data only.
Not clinical evidence. Not suitable for medical validation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DiaLong-AutoML — train all models with Optuna AutoML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/demo_fast.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=None,
        dest="data_csv",
        help=(
            "Override path to the full-timeline CSV. "
            "Defaults to data/synthetic/diabetes_progression_full.csv."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        dest="log_level",
        help="Override log level (DEBUG/INFO/WARNING/ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Bootstrap logging early so we see all output
    # ------------------------------------------------------------------
    log_level = args.log_level or "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("dialong_automl.train")

    logger.info("=" * 60)
    logger.info("DiaLong-AutoML Training Run")
    logger.info("DISCLAIMER: Synthetic demonstration data only.")
    logger.info("  Not clinical evidence. Not for medical validation.")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    from dialong_automl.config import load_config
    from dialong_automl.utils.logging_utils import configure_logging

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    cfg = load_config(config_path)

    # Apply log-level override from CLI if given
    if args.log_level:
        cfg.logging.level = args.log_level.upper()
    configure_logging(cfg.logging, logs_dir=Path(cfg.paths.logs_dir), force=True)

    logger.info("Config loaded: project=%s version=%s seed=%d",
                cfg.project_name, cfg.version, cfg.training.seed)

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    from dialong_automl.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline(cfg, full_csv_path=args.data_csv)

    try:
        result = pipeline.run()
    except FileNotFoundError as exc:
        logger.error("Data file not found: %s", exc)
        logger.error(
            "Generate synthetic data first:\n"
            "  python scripts/generate_demo_data.py "
            "--output data/synthetic/diabetes_progression_demo.csv "
            "--patients 500 --seed 42"
        )
        return 1
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + result.summary())

    print("\nArtifact paths:")
    for name, path in result.artifact_paths.items():
        print(f"  {name:25s}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
