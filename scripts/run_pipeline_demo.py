#!/usr/bin/env python
"""Phase 2+3 end-to-end demo pipeline.

Loads the generated 500-patient CSV, runs validation, profiling,
cohort construction, and splitting, then prints ACTUAL results.

Usage::

    python scripts/run_pipeline_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import logging
logging.basicConfig(level=logging.WARNING)  # suppress INFO during demo

from dialong_automl.data.loader import load_longitudinal_csv
from dialong_automl.data.validator import DataValidator, ValidationConfig
from dialong_automl.data.profiler import DataProfiler
from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
from dialong_automl.data.splitter import PatientSplitter, SplitConfig

CSV = Path("data/synthetic/diabetes_progression_demo.csv")
FULL_CSV = Path("data/synthetic/diabetes_progression_full.csv")

def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def main() -> None:
    # ---------------------------------------------------------------
    # 1. Load
    # ---------------------------------------------------------------
    sep("1. LOADING")
    df = load_longitudinal_csv(CSV)
    print(f"  Rows loaded        : {len(df):,}")
    print(f"  Columns            : {len(df.columns)}")
    print(f"  Unique patients    : {df['patient_id'].nunique():,}")

    # ---------------------------------------------------------------
    # 2. Validate
    # ---------------------------------------------------------------
    sep("2. VALIDATION")
    validator = DataValidator(ValidationConfig(
        target_inconsistency_as_error=True,
        allow_duplicate_patient_date=False,
    ))
    val_result = validator.validate(df)
    print(f"  Passed             : {val_result.passed}")
    print(f"  Errors             : {len(val_result.errors())}")
    print(f"  Warnings           : {len(val_result.warnings())}")
    for f in val_result.findings:
        prefix = "  ERROR  " if f.level == "error" else "  WARN   " if f.level == "warning" else "  INFO   "
        print(f"{prefix}[{f.code}] {f.message}")

    # ---------------------------------------------------------------
    # 3. Profile
    # ---------------------------------------------------------------
    sep("3. PROFILING")
    profiler = DataProfiler()
    profile = profiler.profile(df)
    print(f"  Rows               : {profile.n_rows:,}")
    print(f"  Patients           : {profile.n_patients:,}")
    print(f"  Visits/patient     : min={profile.visits_per_patient_min} "
          f"median={profile.visits_per_patient_median} "
          f"max={profile.visits_per_patient_max}")
    print(f"  Date range         : {profile.date_min} to {profile.date_max}")
    if profile.target_present:
        print(f"  Positive patients  : {profile.n_positive_patients}")
        print(f"  Negative patients  : {profile.n_negative_patients}")
        print(f"  Positive rate      : {profile.positive_rate:.4f}")
    print(f"  Missing cols       : {len(profile.missingness)}")
    print("  Numeric summary (mean / std):")
    for ns in profile.numeric_summaries:
        print(f"    {ns.column:25s}  mean={ns.mean:7.2f}  std={ns.std:6.2f}  missing={ns.missing_rate:.1%}")

    # ---------------------------------------------------------------
    # 4. Cohort construction (FUTURE_OUTCOME on full CSV)
    # ---------------------------------------------------------------
    sep("4. COHORT CONSTRUCTION (future_outcome mode)")
    full_df = load_longitudinal_csv(FULL_CSV)
    print(f"  Full CSV rows      : {len(full_df):,}")

    cohort_cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        max_observation_visits=4,
        prediction_horizon_days=365,
        diabetes_onset_hba1c_threshold=6.5,
        exclude_known_diabetes_at_baseline=True,
        require_followup_after_cutoff=True,
    )
    cohort_result = CohortBuilder(cohort_cfg).build(full_df)
    audit = cohort_result.audit
    print(f"  Input patients     : {audit.input_patients}")
    print(f"  Eligible patients  : {audit.eligible_patients}")
    print(f"  Excl prior diabetes: {audit.excluded_prior_diabetes}")
    print(f"  Excl insuf visits  : {audit.excluded_insufficient_visits}")
    print(f"  Excl insuf followup: {audit.excluded_insufficient_followup}")
    print(f"  Positive patients  : {audit.positive_patients}")
    print(f"  Negative patients  : {audit.negative_patients}")
    print(f"  Positive rate      : {audit.positive_rate:.4f}")
    print(f"  Obs data rows      : {len(cohort_result.observation_data):,}")
    print(f"  Target col present : {'target' not in cohort_result.observation_data.columns}")
    print(f"  is_future_visit    : {'is_future_visit' not in cohort_result.observation_data.columns}")
    print(f"  Target criteria    : {audit.target_construction_criteria}")

    # ---------------------------------------------------------------
    # 5. Patient split
    # ---------------------------------------------------------------
    sep("5. PATIENT-LEVEL SPLITTING (70/15/15, stratified)")
    split_cfg = SplitConfig(
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        seed=42,
        stratify=True,
    )
    split_result = PatientSplitter(split_cfg).split(
        cohort_result.observation_data,
        cohort_result.patient_targets,
    )
    split_result.assert_no_overlap()
    print(split_result.summary())
    stats = split_result.split_stats
    for s in ("train", "validation", "test"):
        st = stats[s]
        print(f"  {s:12s}  {st['n_patients']:4d} patients  "
              f"pos={st['n_positive']} neg={st['n_negative']} "
              f"rate={st['positive_rate']:.3f}")

    # ---------------------------------------------------------------
    # 6. Leakage assertion
    # ---------------------------------------------------------------
    sep("6. LEAKAGE ASSERTION")
    obs = cohort_result.observation_data
    leakage_cols = ["target", "is_future_visit", "future_diabetes",
                    "diabetes_progression", "outcome"]
    found = [c for c in leakage_cols if c in obs.columns]
    print(f"  Leakage cols in obs_data : {found if found else 'NONE (clean)'}")
    print(f"  No patient in multiple splits: {not bool(set(split_result.train_ids) & set(split_result.validation_ids) & set(split_result.test_ids))}")

    print("\n" + "="*60)
    print("  Phase 2+3 end-to-end pipeline COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
