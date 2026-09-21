#!/usr/bin/env python
"""Phase 4 representation demo.

Loads the generated 500-patient synthetic data, builds the future-outcome
cohort, splits patients, builds wide and sequence representations, applies
training-only preprocessing, and prints ACTUAL statistics.

Usage::

    python scripts/run_representation_demo.py

All numbers printed come from actual computation — nothing is fabricated.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import logging
import json
import numpy as np

logging.basicConfig(level=logging.WARNING)  # suppress INFO during demo

from dialong_automl.data.loader import load_longitudinal_csv
from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
from dialong_automl.data.splitter import PatientSplitter, SplitConfig
from dialong_automl.features.wide import WideRepresentationBuilder
from dialong_automl.features.sequence import SequenceRepresentationBuilder
from dialong_automl.features.preprocessing import WidePreprocessor, SequencePreprocessor

FULL_CSV = Path("data/synthetic/diabetes_progression_full.csv")


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load full timeline
    # ------------------------------------------------------------------
    sep("1. LOAD FULL TIMELINE")
    if not FULL_CSV.exists():
        print(f"  ERROR: {FULL_CSV} not found.")
        print("  Run: python scripts/generate_demo_data.py --patients 500 --seed 42")
        sys.exit(1)

    full_df = load_longitudinal_csv(FULL_CSV)
    print(f"  Full CSV rows      : {len(full_df):,}")
    print(f"  Patients           : {full_df['patient_id'].nunique():,}")

    # ------------------------------------------------------------------
    # 2. Build cohort (future-outcome, canonical mode)
    # ------------------------------------------------------------------
    sep("2. BUILD FUTURE-OUTCOME COHORT")
    cohort_cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=3,
        max_observation_visits=4,
        prediction_horizon_days=365,
        diabetes_onset_hba1c_threshold=6.5,
        exclude_known_diabetes_at_baseline=True,
        require_followup_after_cutoff=True,
    )
    cohort = CohortBuilder(cohort_cfg).build(full_df)
    obs = cohort.observation_data
    targets = cohort.patient_targets

    print(f"  Eligible patients  : {cohort.audit.eligible_patients}")
    print(f"  Positive patients  : {cohort.audit.positive_patients}")
    print(f"  Negative patients  : {cohort.audit.negative_patients}")
    print(f"  Positive rate      : {cohort.audit.positive_rate:.4f}")
    print(f"  Observation rows   : {len(obs):,}")
    print(f"  Leakage cols in obs: {[c for c in obs.columns if c in {'target','is_future_visit','outcome','diabetes_progression'}]!r}")

    # ------------------------------------------------------------------
    # 3. Patient-level split
    # ------------------------------------------------------------------
    sep("3. PATIENT SPLIT (70/15/15)")
    split = PatientSplitter(SplitConfig(seed=42)).split(obs, targets)
    split.assert_no_overlap()
    print(split.summary())

    # ------------------------------------------------------------------
    # 4. Wide representation
    # ------------------------------------------------------------------
    sep("4. WIDE REPRESENTATION")
    wide_builder = WideRepresentationBuilder(max_waves=4)

    train_wide = wide_builder.build(split.train_df, patient_targets=split.train_targets)
    val_wide   = wide_builder.build(split.validation_df, patient_targets=split.validation_targets)
    test_wide  = wide_builder.build(split.test_df, patient_targets=split.test_targets)

    print(f"  Train X shape      : {train_wide.X.shape}")
    print(f"  Val X shape        : {val_wide.X.shape}")
    print(f"  Test X shape       : {test_wide.X.shape}")
    print(f"  Feature count      : {len(train_wide.feature_names)}")
    print(f"  Wave features      : {train_wide.metadata['n_wave_features']}")
    print(f"  Temporal features  : {train_wide.metadata['n_temporal_features']}")
    print(f"  Categorical cols   : {train_wide.metadata['n_categorical_cols']}")
    print(f"  Feature mappings   : {len(train_wide.feature_mapping)}")
    print(f"  'target' in X      : {'target' in train_wide.feature_names}")

    print("\n  Example feature mappings:")
    for col in ["hba1c_w1", "hba1c_w3", "hba1c_mean", "hba1c_delta", "hba1c_slope", "sex_latest"]:
        if col in train_wide.feature_mapping:
            info = train_wide.feature_mapping[col]
            print(f"    {col:25s}: {info.transformation!r:20s}  wave={info.visit_wave}  type={info.feature_type}")

    # ------------------------------------------------------------------
    # 5. Wide preprocessing (train-only fit)
    # ------------------------------------------------------------------
    sep("5. WIDE PREPROCESSING (fit on train only)")
    num_cols = [c for c in train_wide.feature_names
                if train_wide.feature_mapping.get(c, None) is not None
                and train_wide.feature_mapping[c].feature_type in ("numeric_wave", "numeric_temporal", "meta")]
    cat_cols = [c for c in train_wide.feature_names
                if train_wide.feature_mapping.get(c, None) is not None
                and train_wide.feature_mapping[c].feature_type == "categorical"]

    wide_prep = WidePreprocessor(
        numeric_columns=num_cols if num_cols else None,
        categorical_columns=cat_cols if cat_cols else None,
    )
    X_train_proc = wide_prep.fit_transform(train_wide.X)
    X_val_proc   = wide_prep.transform(val_wide.X)
    X_test_proc  = wide_prep.transform(test_wide.X)

    print(f"  Train preprocessed shape  : {X_train_proc.shape}")
    print(f"  Val preprocessed shape    : {X_val_proc.shape}")
    print(f"  Test preprocessed shape   : {X_test_proc.shape}")
    print(f"  Output feature count      : {len(wide_prep.feature_names_out)}")
    print(f"  Output dtype              : {X_train_proc.dtype}")
    print(f"  Train mean (should ~0)    : {float(X_train_proc[:, :2].mean()):.4f}")

    # ------------------------------------------------------------------
    # 6. Sequence representation
    # ------------------------------------------------------------------
    sep("6. SEQUENCE REPRESENTATION")
    seq_builder = SequenceRepresentationBuilder(max_seq_len=4)

    train_seq = seq_builder.build(split.train_df, patient_targets=split.train_targets)
    val_seq   = SequenceRepresentationBuilder(
        max_seq_len=4,
        categorical_vocabularies=dict(train_seq.categorical_vocabularies),
    ).build(split.validation_df, patient_targets=split.validation_targets)
    test_seq  = SequenceRepresentationBuilder(
        max_seq_len=4,
        categorical_vocabularies=dict(train_seq.categorical_vocabularies),
    ).build(split.test_df, patient_targets=split.test_targets)

    print(f"  Train X shape      : {train_seq.X.shape}")
    print(f"  Train mask shape   : {train_seq.mask.shape}")
    print(f"  Val X shape        : {val_seq.X.shape}")
    print(f"  Test X shape       : {test_seq.X.shape}")
    print(f"  Feature count      : {len(train_seq.feature_names)}")
    print(f"  Feature names      : {train_seq.feature_names}")
    print(f"  'target' in names  : {'target' in train_seq.feature_names}")

    lengths = train_seq.lengths
    print(f"\n  Sequence length distribution (train):")
    print(f"    min     = {int(lengths.min())}")
    print(f"    max     = {int(lengths.max())}")
    print(f"    median  = {float(np.median(lengths)):.1f}")
    print(f"    mean    = {float(lengths.mean()):.2f}")
    for v in sorted(np.unique(lengths)):
        count = int((lengths == v).sum())
        print(f"    len={v}: {count} patients")

    # ------------------------------------------------------------------
    # 7. Sequence preprocessing (fit on train only)
    # ------------------------------------------------------------------
    sep("7. SEQUENCE PREPROCESSING (fit on train only)")
    seq_prep = SequencePreprocessor(feature_names=train_seq.feature_names)
    X_seq_train_proc = seq_prep.fit_transform(train_seq.X, train_seq.mask)
    X_seq_val_proc   = seq_prep.transform(val_seq.X, val_seq.mask)
    X_seq_test_proc  = seq_prep.transform(test_seq.X, test_seq.mask)

    print(f"  Train preprocessed shape  : {X_seq_train_proc.shape}")
    print(f"  Val preprocessed shape    : {X_seq_val_proc.shape}")
    print(f"  Test preprocessed shape   : {X_seq_test_proc.shape}")
    print(f"  Padding zeroed (val)      : {bool((X_seq_val_proc[val_seq.mask == 0, :] == 0.0).all())}")

    # ------------------------------------------------------------------
    # 8. Leakage assertions
    # ------------------------------------------------------------------
    sep("8. LEAKAGE ASSERTIONS")
    leakage_cols = {"target", "is_future_visit", "diabetes_progression",
                    "outcome", "future_hba1c", "future_glucose",
                    "diagnosis_after", "post_outcome"}

    wide_leakage = [c for c in train_wide.feature_names if c in leakage_cols]
    seq_leakage  = [c for c in train_seq.feature_names  if c in leakage_cols]

    print(f"  Leakage cols in wide X    : {wide_leakage if wide_leakage else 'NONE (clean)'}")
    print(f"  Leakage cols in seq X     : {seq_leakage  if seq_leakage  else 'NONE (clean)'}")
    print(f"  No patient overlap splits : {not bool(set(split.train_ids) & set(split.validation_ids) & set(split.test_ids))}")

    # Future values must not appear in wide wave columns
    future_df = full_df[full_df["is_future_visit"].astype(str).str.lower() == "true"]
    future_hba1c_max = float(future_df["hba1c"].max()) if len(future_df) > 0 else float("nan")
    wave_hba1c_max = float(train_wide.X[[c for c in train_wide.feature_names if "_w" in c and "hba1c" in c]].max().max())
    print(f"  Future HbA1c max          : {future_hba1c_max:.2f}")
    print(f"  Wide wave HbA1c max       : {wave_hba1c_max:.2f}")
    print(f"  Future val in wide waves  : {future_hba1c_max in train_wide.X[[c for c in train_wide.feature_names if '_w' in c and 'hba1c' in c]].values.tolist()}")

    print("\n" + "=" * 60)
    print("  Phase 4 representation demo COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
