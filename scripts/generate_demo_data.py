#!/usr/bin/env python
"""Generate a synthetic longitudinal diabetes progression dataset.

DISCLAIMER: All generated data is synthetic demonstration data.
It is NOT clinical evidence and NOT suitable for medical validation.

Usage::

    python scripts/generate_demo_data.py \\
        --output data/synthetic/diabetes_progression_demo.csv \\
        --patients 500 \\
        --seed 42

    python scripts/generate_demo_data.py \\
        --output data/synthetic/diabetes_progression_demo.csv \\
        --patients 1000 \\
        --min-visits 4 \\
        --max-visits 6 \\
        --seed 42 \\
        --prevalence 0.30 \\
        --missingness 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the package is importable when run directly from the repo root
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("generate_demo_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic longitudinal diabetes progression data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/diabetes_progression_demo.csv",
        help="Path for the observation-only CSV output.",
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=1000,
        help="Number of synthetic patients to generate.",
    )
    parser.add_argument(
        "--min-visits",
        type=int,
        default=4,
        dest="min_visits",
        help="Minimum observation visits per patient.",
    )
    parser.add_argument(
        "--max-visits",
        type=int,
        default=6,
        dest="max_visits",
        help="Maximum observation visits per patient.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--prevalence",
        type=float,
        default=0.30,
        dest="prevalence",
        help="Approximate target prevalence (fraction who progress).",
    )
    parser.add_argument(
        "--missingness",
        type=float,
        default=0.05,
        dest="missingness",
        help="Fraction of numeric values to set as missing (NaN).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output_path = Path(args.output)
    output_dir = output_path.parent

    logger.info("=" * 60)
    logger.info("DiaLong-AutoML — Synthetic Data Generator")
    logger.info("DISCLAIMER: Synthetic demonstration data only.")
    logger.info("  Not clinical evidence. Not for medical validation.")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  patients    = %d", args.patients)
    logger.info("  min_visits  = %d", args.min_visits)
    logger.info("  max_visits  = %d", args.max_visits)
    logger.info("  seed        = %d", args.seed)
    logger.info("  prevalence  = %.2f", args.prevalence)
    logger.info("  missingness = %.2f", args.missingness)
    logger.info("  output_dir  = %s", output_dir)

    # Import after path setup
    from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator

    cfg = GeneratorConfig(
        n_patients=args.patients,
        min_obs_visits=args.min_visits,
        max_obs_visits=args.max_visits,
        progression_prevalence=args.prevalence,
        missingness_rate=args.missingness,
        seed=args.seed,
    )

    generator = SyntheticGenerator(config=cfg)
    result = generator.generate()

    # Save all files to output_dir (includes full CSV + demo CSV + metadata)
    paths = result.save(output_dir=output_dir)

    # If the requested output path differs from the default obs CSV name,
    # copy/write it to the requested location
    default_obs_path = output_dir / "diabetes_progression_demo.csv"
    if output_path.resolve() != default_obs_path.resolve():
        logger.info("Writing obs CSV to requested path: %s", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.obs_df.to_csv(output_path, index=False)

    # Print actual statistics — no fake numbers
    md = result.metadata
    print("\n" + "=" * 60)
    print("ACTUAL GENERATION STATISTICS")
    print("=" * 60)
    print(f"  Disclaimer      : {md['disclaimer']}")
    print(f"  Generator ver.  : {md['generator_version']}")
    print(f"  Seed            : {md['seed']}")
    print(f"  Patients        : {md['actual_n_patients']}")
    print(f"  Obs rows        : {md['total_observation_rows']}")
    print(f"  Full rows       : {md['total_full_rows']}")
    print(f"  Positive (target=1): {md['n_positive_patients']}")
    print(f"  Negative (target=0): {md['n_negative_patients']}")
    print(f"  Actual prevalence  : {md['actual_progression_prevalence']:.4f}")
    print(f"  Intended prevalence: {md['intended_progression_prevalence']:.4f}")
    print(f"  Visits/patient  : min={md['visits_per_patient_min']} "
          f"median={md['visits_per_patient_median']} "
          f"max={md['visits_per_patient_max']}")
    print(f"  Actual missingness : {md['actual_missingness_rate']:.4f}")
    print(f"  Intended missingness: {md['intended_missingness_rate']:.4f}")
    print(f"  Obs CSV         : {paths['obs_csv']}")
    print(f"  Full CSV        : {paths['full_csv']}")
    print(f"  Metadata JSON   : {paths['metadata_json']}")
    print("=" * 60)

    logger.info("Generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
