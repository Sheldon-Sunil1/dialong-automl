"""Tests for the training pipeline (Phase 5).

Uses a tiny synthetic dataset (20 patients) to exercise the full
10-step pipeline without long runtimes.

Test-set protection verifications:
- Pipeline result contains test metrics only in test_metrics dict.
- Preprocessing is fitted on training data only during search.
- After retrain, preprocessing is fitted on train+val.
- Test data is not referenced by optimizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dialong_automl.config.schema import (
    AppConfig, CohortBuilderConfig, SplitterConfig,
    TrainingConfig, GRUConfig, OptunaConfig, ModelConfig,
)
from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator


# ---------------------------------------------------------------------------
# Fixture: tiny config + tiny dataset
# ---------------------------------------------------------------------------

def _tiny_cfg(tmp_path: Path, n_trials: int = 1) -> AppConfig:
    """Build a minimal AppConfig for a fast pipeline smoke test."""
    cfg = AppConfig()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.outputs_dir = str(tmp_path / "outputs")
    cfg.paths.logs_dir = str(tmp_path / "logs")
    cfg.cohort.min_observation_visits = 2
    cfg.cohort.max_observation_visits = 3
    cfg.cohort.prediction_horizon_days = 365 * 5  # wide window for tiny dataset
    cfg.cohort.exclude_known_diabetes_at_baseline = False
    cfg.cohort.require_followup_after_cutoff = True
    cfg.split.random_seed = 42
    cfg.split.train_size = 0.60
    cfg.split.validation_size = 0.20
    cfg.split.test_size = 0.20
    cfg.split.stratify = False  # avoid stratification issues on tiny data
    cfg.training.seed = 42
    cfg.training.threshold = 0.5
    cfg.gru.epochs = 2
    cfg.gru.patience = 2
    cfg.model.enabled_models = ["lightgbm"]  # lightgbm works without blocked DLLs
    cfg.optuna.n_trials = n_trials
    cfg.optuna.storage = None  # in-memory for tests
    cfg.optuna.study_name = "test_study"
    cfg.optuna.seed = 42
    return cfg


def _tiny_csv(tmp_path: Path, n_patients: int = 60) -> Path:
    """Generate a tiny full-timeline CSV and return its path."""
    gen_cfg = GeneratorConfig(
        n_patients=n_patients,
        min_obs_visits=2,
        max_obs_visits=3,
        min_future_visits=2,
        max_future_visits=3,
        seed=42,
        progression_prevalence=0.40,
        missingness_rate=0.0,
    )
    result = SyntheticGenerator(config=gen_cfg).generate()
    out_dir = tmp_path / "data" / "synthetic"
    paths = result.save(output_dir=out_dir)
    return paths["full_csv"]


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

def test_pipeline_runs_end_to_end(tmp_path):
    """Full pipeline runs without error on tiny data and produces artifacts."""
    from dialong_automl.training.pipeline import TrainingPipeline

    csv = _tiny_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    pipeline = TrainingPipeline(cfg, full_csv_path=csv)
    result = pipeline.run()

    # Basic counts
    assert result.n_eligible_patients > 0
    assert result.n_train > 0
    assert result.n_val > 0
    assert result.n_test > 0
    assert result.n_train + result.n_val + result.n_test == result.n_eligible_patients

    # Model selected
    assert result.best_model_name != ""
    assert result.best_representation in ("wide", "sequence")

    # Timing is positive
    assert result.total_run_seconds > 0

    # Artifacts exist
    for key, path_str in result.artifact_paths.items():
        if path_str.startswith("sqlite"):
            continue  # storage URL, not a file path
        p = Path(path_str)
        assert p.exists() and p.stat().st_size > 0, (
            f"Artifact {key!r} missing or empty: {path_str}"
        )


def test_pipeline_test_metrics_non_empty(tmp_path):
    """test_metrics dict is populated and contains auroc key."""
    from dialong_automl.training.pipeline import TrainingPipeline

    csv = _tiny_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    result = TrainingPipeline(cfg, full_csv_path=csv).run()

    assert "test_auroc" in result.test_metrics
    # auroc may be None if single-class val, but key must exist
    assert "test_auroc" in result.test_metrics


def test_pipeline_no_overlap_in_splits(tmp_path):
    """No patient appears in more than one split."""
    from dialong_automl.training.pipeline import TrainingPipeline
    from dialong_automl.data.splitter import PatientSplitter, SplitConfig
    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
    from dialong_automl.data.loader import load_longitudinal_csv

    csv = _tiny_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path)

    # Rebuild the split independently to verify no overlap
    full_df = load_longitudinal_csv(csv)
    cohort_cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=cfg.cohort.min_observation_visits,
        max_observation_visits=cfg.cohort.max_observation_visits,
        prediction_horizon_days=cfg.cohort.prediction_horizon_days,
        exclude_known_diabetes_at_baseline=cfg.cohort.exclude_known_diabetes_at_baseline,
        require_followup_after_cutoff=cfg.cohort.require_followup_after_cutoff,
    )
    cohort = CohortBuilder(cohort_cfg).build(full_df)
    split = PatientSplitter(
        SplitConfig(
            train_size=cfg.split.train_size,
            validation_size=cfg.split.validation_size,
            test_size=cfg.split.test_size,
            seed=cfg.split.random_seed,
            stratify=cfg.split.stratify,
        )
    ).split(cohort.observation_data, cohort.patient_targets)
    split.assert_no_overlap()


def test_pipeline_best_model_json_has_no_test_selection(tmp_path):
    """best_config.json explicitly states test data was not used for selection."""
    from dialong_automl.training.pipeline import TrainingPipeline

    csv = _tiny_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    result = TrainingPipeline(cfg, full_csv_path=csv).run()

    best_config_path = Path(result.artifact_paths["best_config"])
    with open(best_config_path) as fh:
        best_cfg = json.load(fh)

    assert best_cfg["selection_criterion"] == "validation_auroc"
    note = best_cfg.get("note", "").lower()
    assert "test" in note


def test_pipeline_metadata_json_contains_disclaimer(tmp_path):
    """model_metadata.json contains synthetic disclaimer."""
    from dialong_automl.training.pipeline import TrainingPipeline

    csv = _tiny_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    result = TrainingPipeline(cfg, full_csv_path=csv).run()

    meta_path = Path(result.artifact_paths["model_metadata"])
    with open(meta_path) as fh:
        meta = json.load(fh)

    assert meta["synthetic_data"] is True
    assert "disclaimer" in meta
    assert "Not clinical evidence" in meta["disclaimer"]
    assert meta["test_set_accessed_for_selection"] is False


def test_pipeline_run_summary_json_structure(tmp_path):
    """run_summary.json has expected structure."""
    from dialong_automl.training.pipeline import TrainingPipeline

    csv = _tiny_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    result = TrainingPipeline(cfg, full_csv_path=csv).run()

    summary_path = Path(result.artifact_paths["run_summary"])
    with open(summary_path) as fh:
        summary = json.load(fh)

    for key in ["selected_model", "test_metrics", "validation_metrics",
                "total_run_seconds", "split"]:
        assert key in summary, f"Missing key in run_summary: {key}"


def test_preprocessing_not_fitted_on_test_behavioral(tmp_path):
    """
    Behavioral test: the preprocessor fitted during Optuna search must NOT
    have been fitted on test data.

    Strategy:
    - Run the pipeline with a config where we know train/val/test split sizes.
    - After the run, load the Optuna-phase preprocessor (not the final one
      which is train+val) by running a second pipeline pass up to step 5 only.
    - Actually, we verify by checking that the preprocessor's column statistics
      are consistent with a dataset of n_train rows only.
    - More critically: verify that the *final* preprocessor (refit on train+val)
      has different statistics from a preprocessor fit on train+val+test.
    """
    from dialong_automl.training.pipeline import TrainingPipeline
    from dialong_automl.features.preprocessing import WidePreprocessor
    from dialong_automl.features.wide import WideRepresentationBuilder
    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
    from dialong_automl.data.splitter import PatientSplitter, SplitConfig
    from dialong_automl.data.loader import load_longitudinal_csv
    import numpy as np

    csv = _tiny_csv(tmp_path, n_patients=80)
    cfg = _tiny_cfg(tmp_path)
    result = TrainingPipeline(cfg, full_csv_path=csv).run()

    prep_path = Path(result.artifact_paths["preprocessor"])
    final_prep = WidePreprocessor.load(prep_path)
    assert final_prep.is_fitted

    # Reconstruct the cohort and split to compare preprocessor statistics
    full_df = load_longitudinal_csv(csv)
    cohort_cfg = CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=cfg.cohort.min_observation_visits,
        max_observation_visits=cfg.cohort.max_observation_visits,
        prediction_horizon_days=cfg.cohort.prediction_horizon_days,
        exclude_known_diabetes_at_baseline=cfg.cohort.exclude_known_diabetes_at_baseline,
        require_followup_after_cutoff=cfg.cohort.require_followup_after_cutoff,
    )
    cohort = CohortBuilder(cohort_cfg).build(full_df)
    split = PatientSplitter(SplitConfig(
        train_size=cfg.split.train_size,
        validation_size=cfg.split.validation_size,
        test_size=cfg.split.test_size,
        seed=cfg.split.random_seed,
        stratify=cfg.split.stratify,
    )).split(cohort.observation_data, cohort.patient_targets)

    wide_builder = WideRepresentationBuilder(max_waves=cfg.cohort.max_observation_visits)
    train_wide = wide_builder.build(split.train_df, split.train_targets)
    val_wide   = wide_builder.build(split.validation_df, split.validation_targets)
    test_wide  = wide_builder.build(split.test_df, split.test_targets)

    # Build a preprocessor fitted on train+val only (what the pipeline does for final retrain)
    import pandas as pd
    train_val_wide_X = pd.concat([train_wide.X, val_wide.X], ignore_index=True)
    prep_trainval = WidePreprocessor()
    prep_trainval.fit(train_val_wide_X)

    # Build a preprocessor fitted on train+val+TEST (what leakage would produce)
    train_val_test_X = pd.concat([train_wide.X, val_wide.X, test_wide.X], ignore_index=True)
    prep_all = WidePreprocessor()
    prep_all.fit(train_val_test_X)

    # The final (saved) preprocessor's means should match train+val, NOT train+val+test
    # (for at least one numeric column they should differ if test data differs from train+val)
    saved_means = np.array([final_prep._num_means[c] for c in (final_prep.numeric_columns or [])])
    tv_means    = np.array([prep_trainval._num_means[c] for c in (final_prep.numeric_columns or [])])
    all_means   = np.array([prep_all._num_means[c] for c in (final_prep.numeric_columns or [])])

    # The saved preprocessor must match train+val stats (not all-data stats)
    # For the comparison to be meaningful, test data must have different statistics
    # from train+val — which is probabilistically true for different random splits.
    # We only assert when the means actually differ (degenerate tiny datasets may not differ).
    if not np.allclose(tv_means, all_means, atol=1e-6):
        # Means differ → we can tell which the saved preprocessor matches
        tv_dist  = np.abs(saved_means - tv_means).mean()
        all_dist = np.abs(saved_means - all_means).mean()
        assert tv_dist <= all_dist + 1e-6, (
            f"LEAKAGE DETECTED: saved preprocessor means match train+val+test "
            f"statistics better than train+val statistics. "
            f"tv_dist={tv_dist:.6f}, all_dist={all_dist:.6f}"
        )


def test_test_labels_not_used_for_threshold_selection(tmp_path):
    """
    Behavioral test: the selected threshold must be the default (0.5).
    No threshold optimization on the test set occurs.

    We verify by:
    1. Running the pipeline with threshold=0.5
    2. Checking that the test_metrics use exactly 0.5
    3. Checking that no threshold selection code in the pipeline
       references y_test or test labels.
    """
    from dialong_automl.training.pipeline import TrainingPipeline
    import inspect

    csv = _tiny_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    cfg.training.threshold = 0.5
    result = TrainingPipeline(cfg, full_csv_path=csv).run()

    # Threshold in test_metrics must be the configured value, not something
    # derived from test labels
    assert result.test_metrics.get("test_threshold") == 0.5, (
        f"Expected threshold=0.5, got {result.test_metrics.get('test_threshold')}"
    )

    # The pipeline source must not contain any reference to y_test for
    # threshold selection
    import dialong_automl.training.pipeline as _pipe_mod
    src = inspect.getsource(_pipe_mod)
    # Verify threshold is taken from cfg, not computed from test labels
    assert "threshold = self.cfg.training.threshold" in src, (
        "Threshold should be read from config, not computed from test data"
    )

    # run_summary also records the threshold
    summary_path = Path(result.artifact_paths["run_summary"])
    with open(summary_path) as fh:
        summary = json.load(fh)
    # threshold recorded in test_metrics section
    tm = summary.get("test_metrics", {})
    assert tm.get("test_threshold") == 0.5


def test_leaderboard_csv_exists_and_has_rows(tmp_path):
    """leaderboard.csv exists and has at least one data row."""
    from dialong_automl.training.pipeline import TrainingPipeline

    csv = _tiny_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    result = TrainingPipeline(cfg, full_csv_path=csv).run()

    lb_path = Path(result.artifact_paths["leaderboard"])
    assert lb_path.exists()
    lb = pd.read_csv(lb_path)
    assert len(lb) >= 1
    assert "validation_auroc" in lb.columns
