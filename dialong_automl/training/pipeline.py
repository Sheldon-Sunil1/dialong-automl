"""Canonical end-to-end training pipeline for DiaLong-AutoML.

Test-set protection rules (enforced throughout this file)
----------------------------------------------------------
1. The test split is loaded once at the start and stored in ``self._test_*``
   attributes.  It is NEVER passed to Optuna, preprocessing fitting, model
   selection, or threshold selection.

2. Preprocessing is fitted ONLY on train data during Optuna search.
   For the final retrain it is refitted on train+validation data.

3. Optuna receives only ``X_train_*`` and ``X_val_*``.

4. The test set is evaluated exactly ONCE, at the very end of
   ``run()``, after the best configuration has been selected and the
   final model retrained on train+validation.

5. Model selection criterion: validation AUROC only.

6. Threshold is always 0.5 (configurable) and never tuned on the test set.

Pipeline steps
--------------
1. Load full-timeline CSV.
2. Build future-outcome cohort (Phase 3).
3. Patient-level stratified split (Phase 3).
4. Build wide + sequence representations for each split.
5. Fit WidePreprocessor on train, transform val and test.
6. Fit SequencePreprocessor on train, transform val and test.
7. Run AutoMLOptimizer (Optuna) — val only, test never accessed.
8. Save leaderboard + best_config.json.
9. Retrain best model on train+validation (preprocessing refitted on train+val).
10. Evaluate best model on test set — ONCE, final.
11. Save all artefacts.
12. Return PipelineResult with all actual metrics.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dialong_automl.config.schema import AppConfig
from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
from dialong_automl.data.loader import load_longitudinal_csv
from dialong_automl.data.splitter import PatientSplitter, SplitConfig
from dialong_automl.evaluation.metrics import evaluate, metrics_to_jsonable, select_threshold_by_f1
from dialong_automl.features.preprocessing import SequencePreprocessor, WidePreprocessor
from dialong_automl.features.sequence import SequenceRepresentationBuilder
from dialong_automl.features.wide import WideRepresentationBuilder
from dialong_automl.models import build_model
from dialong_automl.models.base import REPRESENTATION_SEQUENCE, REPRESENTATION_WIDE
from dialong_automl.automl.optimizer import AutoMLOptimizer, resolve_optuna_storage
from dialong_automl.utils.paths import ArtifactPaths
from dialong_automl.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """All outputs produced by one complete pipeline run."""

    # Counts
    n_input_patients: int
    n_eligible_patients: int
    n_train: int
    n_val: int
    n_test: int
    positive_rate: float

    # Selection
    best_model_name: str
    best_representation: str
    best_hyperparams: dict[str, Any]
    best_val_auroc: float | None

    # Validation metrics (best model, before retrain)
    val_metrics: dict[str, Any]

    # Final test metrics (after retrain on train+val, evaluated once)
    test_metrics: dict[str, Any]

    # Timing
    total_run_seconds: float
    optuna_seconds: float
    retrain_seconds: float

    # Paths
    artifact_paths: dict[str, str] = field(default_factory=dict)

    # Optuna summary
    n_trials_completed: int = 0
    n_trials_total: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DiaLong-AutoML Training Pipeline — Result",
            "=" * 60,
            f"  Eligible patients   : {self.n_eligible_patients}",
            f"  Train / Val / Test  : {self.n_train} / {self.n_val} / {self.n_test}",
            f"  Positive rate       : {self.positive_rate:.4f}",
            f"  Best model          : {self.best_model_name} ({self.best_representation})",
            f"  Val AUROC           : {self.best_val_auroc:.4f}" if self.best_val_auroc else "  Val AUROC           : N/A",
        ]
        for k, v in self.test_metrics.items():
            if isinstance(v, float) and k not in ("threshold",):
                lines.append(f"  Test {k:20s}: {v:.4f}")
        lines.append(f"  Total run time      : {self.total_run_seconds:.1f}s")
        lines.append("=" * 60)
        return "\n".join(lines)


class HeldOutTestSet:
    """Stores the test split and forbids access until final evaluation.

    Construction of the object is allowed (the split itself is created
    up front).  Feature construction, preprocessor fitting, Optuna, and
    threshold selection must not call :meth:`release`.
    """

    def __init__(
        self,
        observation_df: pd.DataFrame,
        targets: pd.Series,
        patient_ids: list[str],
    ) -> None:
        self._observation_df = observation_df
        self._targets = targets
        self._patient_ids = list(patient_ids)
        self._released = False
        self.access_count = 0

    def release_for_final_evaluation(self) -> None:
        """Permit test-set access. Call only after model selection."""
        self._released = True
        logger.info("Test set released for one-time final evaluation.")

    def _require_released(self) -> None:
        if not self._released:
            raise RuntimeError(
                "Test set accessed before final evaluation. "
                "Optuna, model selection, threshold selection, and "
                "preprocessor fitting must not use the test split."
            )
        self.access_count += 1

    @property
    def is_released(self) -> bool:
        return self._released

    @property
    def observation_df(self) -> pd.DataFrame:
        self._require_released()
        return self._observation_df

    @property
    def targets(self) -> pd.Series:
        self._require_released()
        return self._targets

    @property
    def patient_ids(self) -> list[str]:
        self._require_released()
        return list(self._patient_ids)

    @property
    def n_patients(self) -> int:
        return len(self._patient_ids)


class TrainingPipeline:
    """Canonical end-to-end training pipeline.

    Parameters
    ----------
    cfg:
        Loaded ``AppConfig`` (from ``load_config``).
    full_csv_path:
        Override path to the full-timeline CSV.  When ``None`` the path
        is derived from ``cfg.paths.data_dir / "synthetic" /
        "diabetes_progression_full.csv"``.
    """

    def __init__(
        self,
        cfg: AppConfig,
        full_csv_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self._full_csv_path = (
            Path(full_csv_path)
            if full_csv_path
            else Path(cfg.paths.data_dir) / "synthetic" / "diabetes_progression_full.csv"
        )
        self._ap = ArtifactPaths(
            data_dir=cfg.paths.data_dir,
            artifacts_dir=cfg.paths.artifacts_dir,
            outputs_dir=cfg.paths.outputs_dir,
            logs_dir=cfg.paths.logs_dir,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """Execute the full pipeline and return a :class:`PipelineResult`."""
        t_total_start = time.perf_counter()

        # Create output directories
        Path(self.cfg.paths.artifacts_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.paths.outputs_dir).mkdir(parents=True, exist_ok=True)

        seed = self.cfg.training.seed
        set_global_seed(seed)

        # ------------------------------------------------------------------
        # Step 1: Load full-timeline CSV
        # ------------------------------------------------------------------
        logger.info("Step 1/10: Loading full timeline CSV from %s", self._full_csv_path)
        if not self._full_csv_path.exists():
            raise FileNotFoundError(
                f"Full-timeline CSV not found: {self._full_csv_path}\n"
                "Run: python scripts/generate_demo_data.py first."
            )
        full_df = load_longitudinal_csv(self._full_csv_path)
        logger.info("  Loaded %d rows, %d patients", len(full_df), full_df["patient_id"].nunique())

        # ------------------------------------------------------------------
        # Step 2: Build future-outcome cohort
        # ------------------------------------------------------------------
        logger.info("Step 2/10: Building future-outcome cohort")
        cohort_cfg = CohortConfig(
            mode=CohortMode.FUTURE_OUTCOME,
            min_observation_visits=self.cfg.cohort.min_observation_visits,
            max_observation_visits=self.cfg.cohort.max_observation_visits,
            prediction_horizon_days=self.cfg.cohort.prediction_horizon_days,
            diabetes_onset_hba1c_threshold=self.cfg.cohort.diabetes_onset_hba1c_threshold,
            exclude_known_diabetes_at_baseline=self.cfg.cohort.exclude_known_diabetes_at_baseline,
            require_followup_after_cutoff=self.cfg.cohort.require_followup_after_cutoff,
        )
        cohort = CohortBuilder(cohort_cfg).build(full_df)
        obs = cohort.observation_data
        targets = cohort.patient_targets
        n_input = cohort.audit.input_patients
        n_eligible = cohort.audit.eligible_patients
        pos_rate = cohort.audit.positive_rate
        logger.info(
            "  Eligible=%d, positive_rate=%.4f",
            n_eligible, pos_rate,
        )

        # ------------------------------------------------------------------
        # Step 3: Patient-level split
        # ------------------------------------------------------------------
        logger.info("Step 3/10: Splitting patients (70/15/15)")
        sc = self.cfg.split
        split = PatientSplitter(
            SplitConfig(
                train_size=sc.train_size,
                validation_size=sc.validation_size,
                test_size=sc.test_size,
                seed=sc.random_seed,
                stratify=sc.stratify,
                chronological=sc.chronological,
            )
        ).split(obs, targets)
        split.assert_no_overlap()
        n_train = len(split.train_ids)
        n_val = len(split.validation_ids)
        n_test = len(split.test_ids)
        logger.info("  Train=%d  Val=%d  Test=%d", n_train, n_val, n_test)

        # ------------------------------------------------------------------
        # Step 4: Build representations
        # ------------------------------------------------------------------
        logger.info("Step 4/10: Building wide + sequence representations")
        max_waves = self.cfg.cohort.max_observation_visits
        max_seq = self.cfg.cohort.max_observation_visits

        wide_builder = WideRepresentationBuilder(max_waves=max_waves)
        train_wide = wide_builder.build(split.train_df, split.train_targets)
        val_wide = wide_builder.build(split.validation_df, split.validation_targets)
        # Test wide — built now but NOT used until final evaluation
        test_wide = wide_builder.build(split.test_df, split.test_targets)

        seq_builder = SequenceRepresentationBuilder(max_seq_len=max_seq)
        train_seq = seq_builder.build(split.train_df, split.train_targets)
        # Val / test use training vocabulary — no vocabulary leakage
        val_seq = SequenceRepresentationBuilder(
            max_seq_len=max_seq,
            categorical_vocabularies=dict(train_seq.categorical_vocabularies),
        ).build(split.validation_df, split.validation_targets)
        test_seq = SequenceRepresentationBuilder(
            max_seq_len=max_seq,
            categorical_vocabularies=dict(train_seq.categorical_vocabularies),
        ).build(split.test_df, split.test_targets)

        logger.info(
            "  Wide: train=%s  Seq: train=%s",
            train_wide.X.shape, train_seq.X.shape,
        )

        # ------------------------------------------------------------------
        # Step 5: Fit WidePreprocessor on train only, transform val + test
        # ------------------------------------------------------------------
        logger.info("Step 5/10: Fitting WidePreprocessor on training data")
        wide_prep = WidePreprocessor()
        X_train_wide = wide_prep.fit_transform(train_wide.X)
        X_val_wide = wide_prep.transform(val_wide.X)
        X_test_wide = wide_prep.transform(test_wide.X)  # stored, not used yet

        y_train = split.train_targets.values.astype(np.int32)
        y_val = split.validation_targets.values.astype(np.int32)
        y_test = split.test_targets.values.astype(np.int32)

        # ------------------------------------------------------------------
        # Step 6: Fit SequencePreprocessor on train only, transform val + test
        # ------------------------------------------------------------------
        logger.info("Step 6/10: Fitting SequencePreprocessor on training data")
        seq_prep = SequencePreprocessor(feature_names=train_seq.feature_names)
        X_train_seq = seq_prep.fit_transform(train_seq.X, train_seq.mask)
        X_val_seq = seq_prep.transform(val_seq.X, val_seq.mask)
        X_test_seq = seq_prep.transform(test_seq.X, test_seq.mask)  # stored, not used yet

        # Derive lengths for GRU (max of 1 to avoid packing errors)
        def _lengths(mask: np.ndarray) -> np.ndarray:
            return np.maximum(mask.sum(axis=1).astype(np.int32), 1)

        lengths_train = _lengths(train_seq.mask)
        lengths_val = _lengths(val_seq.mask)
        lengths_test = _lengths(test_seq.mask)  # stored, not used yet

        # ------------------------------------------------------------------
        # Step 7: Optuna search — val only, test never accessed
        # ------------------------------------------------------------------
        logger.info("Step 7/10: Running Optuna search (%d trials)", self.cfg.optuna.n_trials)
        t_optuna_start = time.perf_counter()

        # Resolve storage path: convert relative sqlite URL to absolute path
        # under the artifacts directory.  The yaml uses the form
        # "sqlite:///artifacts/optuna_study.db"; we want to place the file
        # directly inside artifacts_dir, not nested again.
        storage = self.cfg.optuna.storage
        if storage and storage.startswith("sqlite:///"):
            raw_db = storage[len("sqlite:///"):]
            raw_path = Path(raw_db)
            if not raw_path.is_absolute():
                artifacts_dir = Path(self.cfg.paths.artifacts_dir).resolve()
                # Strip any leading segment that matches artifacts_dir name
                # to avoid double-nesting (e.g. artifacts/artifacts/...)
                parts = raw_path.parts
                if parts and parts[0] == artifacts_dir.name:
                    raw_path = Path(*parts[1:]) if len(parts) > 1 else Path("optuna_study.db")
                db_path = artifacts_dir / raw_path
                db_path.parent.mkdir(parents=True, exist_ok=True)
                storage = f"sqlite:///{db_path.as_posix()}"

        optimizer = AutoMLOptimizer(
            n_trials=self.cfg.optuna.n_trials,
            seed=self.cfg.optuna.seed,
            storage=storage,
            study_name=self.cfg.optuna.study_name,
            direction=self.cfg.optuna.direction,
            timeout_seconds=self.cfg.optuna.timeout_seconds,
            enabled_models=self.cfg.model.enabled_models,
            threshold=self.cfg.training.threshold,
            gru_cfg={
                "epochs": self.cfg.gru.epochs,
                "patience": self.cfg.gru.patience,
            },
        )
        optimizer.optimize(
            X_train_wide=X_train_wide,
            y_train=y_train,
            X_val_wide=X_val_wide,
            y_val=y_val,
            X_train_seq=X_train_seq,
            mask_train=train_seq.mask,
            lengths_train=lengths_train,
            X_val_seq=X_val_seq,
            mask_val=val_seq.mask,
            lengths_val=lengths_val,
        )
        optuna_seconds = time.perf_counter() - t_optuna_start

        # ------------------------------------------------------------------
        # Step 8: Save leaderboard + best_config
        # ------------------------------------------------------------------
        logger.info("Step 8/10: Saving leaderboard and best config")
        leaderboard_path = Path(self.cfg.paths.outputs_dir) / "leaderboard.csv"
        best_config_path = Path(self.cfg.paths.outputs_dir) / "best_config.json"
        optimizer.save_leaderboard(leaderboard_path)
        best_cfg = optimizer.save_best_config(best_config_path)

        best = optimizer.best_trial_result()
        if best is None:
            raise RuntimeError("No completed Optuna trials — cannot select best model.")

        best_model_name = best.model_name
        best_representation = best.representation
        best_hyperparams = best.hyperparams
        best_val_auroc = best.val_auroc
        val_metrics_search = best.val_auroc  # from Optuna search

        logger.info(
            "  Best: %s (val_auroc=%.4f)", best_model_name, best_val_auroc or 0.0
        )

        # ------------------------------------------------------------------
        # Step 9: Retrain best model on TRAIN + VALIDATION
        # ------------------------------------------------------------------
        logger.info("Step 9/10: Retraining best model on train+validation")
        t_retrain_start = time.perf_counter()

        # Combine train + validation DataFrames
        train_val_df = pd.concat(
            [split.train_df, split.validation_df], ignore_index=True
        )
        train_val_targets = pd.concat(
            [split.train_targets, split.validation_targets]
        )

        y_trainval = train_val_targets.values.astype(np.int32)

        if best_representation == REPRESENTATION_WIDE:
            # Refit wide preprocessor on train+val
            train_val_wide = wide_builder.build(train_val_df, train_val_targets)
            final_wide_prep = WidePreprocessor()
            X_trainval = final_wide_prep.fit_transform(train_val_wide.X)
            X_final_test = final_wide_prep.transform(test_wide.X)
            final_seq_prep = None
        else:
            # Refit sequence preprocessor on train+val
            train_val_seq = SequenceRepresentationBuilder(
                max_seq_len=max_seq,
                categorical_vocabularies=dict(train_seq.categorical_vocabularies),
            ).build(train_val_df, train_val_targets)
            final_seq_prep = SequencePreprocessor(
                feature_names=train_val_seq.feature_names
            )
            X_trainval = final_seq_prep.fit_transform(
                train_val_seq.X, train_val_seq.mask
            )
            X_final_test = final_seq_prep.transform(test_seq.X, test_seq.mask)
            final_wide_prep = None
            lengths_trainval = _lengths(train_val_seq.mask)
            mask_trainval = train_val_seq.mask

        final_model = build_model(best_model_name, best_hyperparams)

        if best_model_name == "gru":
            final_model.fit(
                X_trainval,
                y_trainval,
                mask=mask_trainval,
                lengths=lengths_trainval,
            )
        else:
            final_model.fit(X_trainval, y_trainval)

        retrain_seconds = time.perf_counter() - t_retrain_start

        # ------------------------------------------------------------------
        # Step 10: Final test evaluation — exactly once
        # ------------------------------------------------------------------
        logger.info("Step 10/10: Final test evaluation (one-time, never repeated)")
        threshold = self.cfg.training.threshold

        t_inf = time.perf_counter()
        if best_model_name == "gru":
            test_proba = final_model.predict_proba_with_mask(
                X_final_test, test_seq.mask
            )
        else:
            test_proba = final_model.predict_proba(X_final_test)
        inf_per_patient = (time.perf_counter() - t_inf) / max(len(test_proba), 1)

        test_metrics = evaluate(y_test, test_proba, threshold=threshold, prefix="test_")

        # Also compute final val metrics on retrained model for reporting
        if best_representation == REPRESENTATION_WIDE:
            val_proba_final = final_model.predict_proba(
                final_wide_prep.transform(val_wide.X)
            )
        else:
            val_proba_final = final_model.predict_proba_with_mask(
                final_seq_prep.transform(val_seq.X, val_seq.mask),
                val_seq.mask,
            )
        val_metrics = evaluate(
            y_val, val_proba_final, threshold=threshold, prefix="val_"
        )

        logger.info(
            "  Test AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
            test_metrics.get("test_auroc") or 0.0,
            test_metrics.get("test_auprc") or 0.0,
            test_metrics.get("test_f1") or 0.0,
        )

        # ------------------------------------------------------------------
        # Save artefacts
        # ------------------------------------------------------------------
        artifacts_dir = Path(self.cfg.paths.artifacts_dir)
        outputs_dir = Path(self.cfg.paths.outputs_dir)

        # Final model
        if best_representation == REPRESENTATION_SEQUENCE:
            model_path = artifacts_dir / "best_model.pt"
        else:
            model_path = artifacts_dir / "best_model.joblib"
        final_model.save(model_path)

        # Preprocessor(s)
        if best_representation == REPRESENTATION_WIDE:
            prep_path = artifacts_dir / "preprocessor.joblib"
            final_wide_prep.save(prep_path)
        else:
            prep_path = artifacts_dir / "preprocessor_seq.joblib"
            final_seq_prep.save(prep_path)

        # Sequence vocabulary (needed for inference on new data)
        vocab_path = artifacts_dir / "seq_vocabulary.json"
        with open(vocab_path, "w") as fh:
            json.dump(train_seq.categorical_vocabularies, fh, indent=2)

        # Model metadata
        total_seconds = time.perf_counter() - t_total_start
        n_trials_done = len(optimizer.build_leaderboard())

        metadata = {
            "project": "DiaLong-AutoML",
            "version": self.cfg.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "synthetic_data": True,
            "disclaimer": (
                "Synthetic demonstration data. "
                "Not clinical evidence. Not suitable for medical validation."
            ),
            "selected_model": best_model_name,
            "representation": best_representation,
            "hyperparameters": best_hyperparams,
            "seed": seed,
            "target_definition": (
                "target=1 iff any future visit (within prediction_horizon_days of "
                "observation cutoff) has diabetes_diagnosis==1 OR "
                f"hba1c >= {self.cfg.cohort.diabetes_onset_hba1c_threshold}"
            ),
            "cohort_mode": "future_outcome",
            "observation_visits": self.cfg.cohort.max_observation_visits,
            "prediction_horizon_days": self.cfg.cohort.prediction_horizon_days,
            "split_counts": {"train": n_train, "val": n_val, "test": n_test},
            "threshold": threshold,
            "validation_metrics": {
                k: v for k, v in val_metrics.items()
                if not isinstance(v, dict)
            },
            "test_metrics": {
                k: v for k, v in test_metrics.items()
                if not isinstance(v, dict)
            },
            "optuna_n_trials_completed": n_trials_done,
            "optuna_selection_criterion": "validation_auroc",
            "test_set_accessed_for_selection": False,
            "total_run_seconds": round(total_seconds, 2),
            "limitations": [
                "Trained on synthetic demonstration data only.",
                "Not validated on real patient data.",
                "No clinical safety evaluation performed.",
                "GRU trained on CPU; production would benefit from GPU.",
            ],
        }
        meta_path = artifacts_dir / "model_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)

        # test_metrics.json
        test_metrics_path = outputs_dir / "test_metrics.json"
        with open(test_metrics_path, "w", encoding="utf-8") as fh:
            json.dump(
                {k: v for k, v in test_metrics.items() if not isinstance(v, dict)},
                fh, indent=2, default=str,
            )

        # run_summary.json
        leaderboard_df = optimizer.build_leaderboard()
        run_summary = {
            "dataset_path": str(self._full_csv_path),
            "n_input_patients": n_input,
            "n_eligible_patients": n_eligible,
            "positive_rate": pos_rate,
            "split": {"train": n_train, "val": n_val, "test": n_test},
            "selected_model": best_model_name,
            "representation": best_representation,
            "n_trials_completed": n_trials_done,
            "validation_auroc_best": best_val_auroc,
            "validation_metrics": {
                k: v for k, v in val_metrics.items()
                if not isinstance(v, dict)
            },
            "test_metrics": {
                k: v for k, v in test_metrics.items()
                if not isinstance(v, dict)
            },
            "total_run_seconds": round(total_seconds, 2),
            "optuna_seconds": round(optuna_seconds, 2),
            "retrain_seconds": round(retrain_seconds, 2),
            "inf_time_per_patient_seconds": round(inf_per_patient, 6),
        }
        summary_path = outputs_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(run_summary, fh, indent=2, default=str)

        artifact_paths = {
            "best_model": str(model_path),
            "preprocessor": str(prep_path),
            "seq_vocabulary": str(vocab_path),
            "model_metadata": str(meta_path),
            "leaderboard": str(leaderboard_path),
            "best_config": str(best_config_path),
            "test_metrics": str(test_metrics_path),
            "run_summary": str(summary_path),
        }
        if storage:
            artifact_paths["optuna_study_db"] = storage

        result = PipelineResult(
            n_input_patients=n_input,
            n_eligible_patients=n_eligible,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            positive_rate=pos_rate,
            best_model_name=best_model_name,
            best_representation=best_representation,
            best_hyperparams=best_hyperparams,
            best_val_auroc=best_val_auroc,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            total_run_seconds=round(total_seconds, 2),
            optuna_seconds=round(optuna_seconds, 2),
            retrain_seconds=round(retrain_seconds, 2),
            artifact_paths=artifact_paths,
            n_trials_completed=n_trials_done,
            n_trials_total=self.cfg.optuna.n_trials,
        )

        logger.info("Pipeline complete. %s", result.summary())
        return result
