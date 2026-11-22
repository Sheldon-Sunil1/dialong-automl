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

6. The decision threshold follows ``training.threshold_strategy``:
   ``fixed`` uses ``training.threshold``; ``validation_f1`` maximises F1 on the
   VALIDATION split using the best search model's pre-retrain predictions. The
   threshold is fixed BEFORE the test set is evaluated and is never tuned on it.

7. Metric labelling (Phase 13B):
   ``validation_metrics`` / ``val_*``  = best search model (fitted on TRAIN only),
                                         evaluated on the validation split.
                                         NOT unbiased: that split also drives model
                                         selection, GRU early stopping and the
                                         validation_f1 threshold.
   ``retrain_resubstitution_metrics`` / ``resub_*`` = the FINAL retrained model
                                         evaluated on the validation patients,
                                         which were part of its training data.
                                         These are NOT held-out performance.
   ``test_metrics`` / ``test_*``       = final model on the untouched test split.

8. Optuna studies are isolated by an identity hash (dataset content, cohort,
   split, search configuration, seeds, optional run id).

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

import hashlib
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
from dialong_automl.evaluation.metrics import evaluate
from dialong_automl.evaluation.threshold import select_decision_threshold
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

    # Validation metrics: best search model (fitted on TRAIN only) on the validation
    # split, before the final retrain. Keys: val_*. NOT an unbiased estimate of
    # generalisation: the same split drives model/threshold selection and GRU
    # early stopping (see VALIDATION_METRICS_SOURCE).
    val_metrics: dict[str, Any]

    # Final test metrics (after retrain on train+val, evaluated once). Keys: test_*.
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

    # Metrics of the FINAL retrained model on the validation patients. These
    # patients were (wholly or partly) in its training data, so this is
    # resubstitution performance, NOT held-out validation. Keys: resub_*.
    resubstitution_metrics: dict[str, Any] = field(default_factory=dict)

    # Decision threshold used for threshold-dependent metrics, and how it was chosen.
    threshold: float = 0.5
    threshold_selection: dict[str, Any] = field(default_factory=dict)

    # Optuna study provenance.
    study_name: str | None = None
    study_identity_hash: str | None = None
    n_trials_failed: int = 0
    failed_trials: list[dict[str, Any]] = field(default_factory=list)

    # How the final model was retrained (see _retrain_policy_* helpers).
    retrain_policy: dict[str, Any] = field(default_factory=dict)

    # Phase 13C reference baselines (fixed comparators, never selectable).
    # {"enabled": False} when disabled; otherwise BaselineRunner.to_dict().
    baselines: dict[str, Any] = field(default_factory=dict)

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


def _aligned_labels(representation: Any, split_name: str) -> np.ndarray:
    """Return the builder's row-aligned labels as an int32 array.

    The representation's ``.y`` is a pandas Series whose order matches the
    rows of ``representation.X`` (both ordered by the builder's sorted
    patient_id).  Using it guarantees each label corresponds to the patient
    of the same feature row — unlike ``split.*_targets.values``, which
    follows the splitter's shuffled patient-ID order.

    Raises
    ------
    ValueError
        If the builder produced no labels (``representation.y is None``) or
        the label count does not match the number of feature rows.
    """
    y = getattr(representation, "y", None)
    if y is None:
        raise ValueError(
            f"Representation for the {split_name!r} split has no labels (.y is None); "
            "cannot align targets. Ensure patient_targets were passed to the builder."
        )
    n_rows = len(representation.patient_ids)
    if len(y) != n_rows:
        raise ValueError(
            f"Label/row count mismatch for the {split_name!r} split: "
            f"{len(y)} labels vs {n_rows} feature rows. "
            "This indicates patients were dropped between features and targets."
        )
    return y.values.astype(np.int32)


def _assert_same_patient_order(wide_rep: Any, seq_rep: Any, split_name: str) -> None:
    """Fail loudly if the wide and sequence builders disagree on patient order.

    The Optuna search shares a single label vector across wide and sequence
    trials, so both representations of a split must enumerate patients in the
    same order for that shared label vector to be correct for both.
    """
    if list(wide_rep.patient_ids) != list(seq_rep.patient_ids):
        raise RuntimeError(
            f"Wide and sequence builders produced different patient orders for "
            f"the {split_name!r} split; a shared label vector would mislabel one "
            "of them. This should not happen (both sort by patient_id)."
        )


def _internal_patient_split(
    patient_ids: list[str],
    targets: pd.Series,
    val_fraction: float,
    seed: int,
    stratify: bool,
) -> tuple[list[str], list[str]]:
    """Split *patient_ids* into (internal_train, internal_val) at patient level.

    Used only for the GRU final retrain so early stopping / best-epoch
    restoration stay active.  The split is reproducible (seeded) and
    label-preserving.  The test set is never involved here.

    Stratified splitting is attempted first (both classes in each subset);
    if that is not possible (too few patients of a class), it falls back to a
    seeded random split rather than failing the whole pipeline.

    Returns
    -------
    (internal_train_ids, internal_val_ids)
        Disjoint, non-empty patient-ID lists whose union is *patient_ids*.
    """
    pids = np.array(list(patient_ids))
    n = len(pids)
    if n < 2:
        raise ValueError(
            f"Need at least 2 patients for an internal train/val split, got {n}."
        )

    labels = targets.loc[list(pids)].astype(int).values
    # Ensure at least one patient in each internal subset.
    n_val = max(1, min(n - 1, round(n * val_fraction)))
    test_size = n_val / n

    both_classes = len(np.unique(labels)) >= 2
    if stratify and both_classes:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        try:
            train_idx, val_idx = next(splitter.split(pids, labels))
            return pids[train_idx].tolist(), pids[val_idx].tolist()
        except ValueError:
            logger.warning(
                "Internal stratified split not possible (%d patients); "
                "falling back to a seeded random split.", n,
            )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_sel = perm[:n_val]
    train_sel = perm[n_val:]
    return pids[train_sel].tolist(), pids[val_sel].tolist()


SEARCH_IDENTITY_VERSION = "13b1-1"

VALIDATION_METRICS_SOURCE = (
    "Best Optuna trial's model, fitted on the TRAIN split only, evaluated on the "
    "validation split before the final retrain. These patients were not in that "
    "model's training data, but the validation split was ALSO used for model and "
    "hyperparameter selection, GRU early stopping / checkpoint restoration, and "
    "(for threshold_strategy='validation_f1') threshold selection. These metrics "
    "are therefore optimistically biased and are NOT an unbiased estimate of "
    "generalisation performance; only test_metrics are untouched by selection."
)
LIMITATIONS_VALIDATION = (
    "Validation split used for model selection, GRU early stopping and (optionally) "
    "threshold selection; validation metrics are not unbiased estimates of "
    "generalisation. The threshold is derived from pre-retrain model probabilities "
    "and applied to the final retrained model."
)
RESUBSTITUTION_NOTE = (
    "Final retrained model evaluated on the validation patients, which were part "
    "of its training data. This is resubstitution performance, NOT held-out "
    "validation performance. Do not report it as validation or generalisation."
)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's CONTENT (independent of its path)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_search_identity(cfg: AppConfig, dataset_sha256: str) -> dict[str, Any]:
    """Describe everything that determines the outcome of the Optuna search.

    The result is hashed into the effective study name, so two runs share a
    persisted study only if the dataset content, cohort, split, search
    configuration, seeds and optional ``optuna.run_id`` are all identical.

    Deliberately excluded: ``n_trials`` and ``timeout_seconds`` (so an
    intentional resume may extend a study), ``storage`` (location only) and
    ``training.threshold_strategy`` (applied after the search).
    """
    return {
        "identity_version": SEARCH_IDENTITY_VERSION,
        "dataset_sha256": dataset_sha256,
        "cohort": cfg.cohort.model_dump(mode="json"),
        "split": cfg.split.model_dump(mode="json"),
        "training": {
            "seed": cfg.training.seed,
            "threshold": cfg.training.threshold,
            "max_iter_sklearn": cfg.training.max_iter_sklearn,
        },
        "gru": cfg.gru.model_dump(mode="json"),
        "model": cfg.model.model_dump(mode="json"),
        "optuna": {
            "base_study_name": cfg.optuna.study_name,
            "seed": cfg.optuna.seed,
            "direction": cfg.optuna.direction,
            "metric": cfg.optuna.metric,
            "sampler": cfg.optuna.sampler,
            "pruner": cfg.optuna.pruner,
            "run_id": cfg.optuna.run_id,
        },
    }


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
        # Records the (internal_train_ids, internal_val_ids) used for the GRU
        # final retrain; None for non-GRU runs. Exposed for testing/auditing.
        self._last_internal_split: tuple[list[str], list[str]] | None = None

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
        dataset_sha256 = file_sha256(self._full_csv_path)

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

        # ------------------------------------------------------------------
        # Labels MUST follow the representation builders' row order, NOT the
        # splitter's shuffled patient-ID order. Both builders order patients
        # by sorted patient_id and expose a `.y` Series aligned to their own
        # feature rows. Using split.*_targets.values here would pair labels
        # with the wrong patients because the splitter preserves
        # StratifiedShuffleSplit order while the builders sort.
        #
        # The optimizer shares a single y_train / y_val across both wide and
        # sequence trials, so the two representations must share patient order.
        # We derive labels from the builder `.y` and assert the wide/sequence
        # patient orders match, failing loudly rather than mislabelling.
        _assert_same_patient_order(train_wide, train_seq, "train")
        _assert_same_patient_order(val_wide, val_seq, "validation")
        _assert_same_patient_order(test_wide, test_seq, "test")

        y_train = _aligned_labels(train_wide, "train")
        y_val = _aligned_labels(val_wide, "validation")
        y_test = _aligned_labels(test_wide, "test")

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
            identity=build_search_identity(self.cfg, dataset_sha256),
            resume=self.cfg.optuna.resume,
            sampler=self.cfg.optuna.sampler,
            pruner=self.cfg.optuna.pruner,
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

        # Failed trials (not pruned!) are reported, never silently dropped.
        failed_trials = optimizer.failed_trials()
        failed_trials_path: Path | None = None
        if failed_trials:
            failed_trials_path = Path(self.cfg.paths.outputs_dir) / "failed_trials.json"
            with open(failed_trials_path, "w", encoding="utf-8") as fh:
                json.dump(failed_trials, fh, indent=2, default=str)
            logger.warning(
                "%d Optuna trial(s) FAILED; diagnostics in %s",
                len(failed_trials), failed_trials_path,
            )

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

        # --------------------------------------------------------------
        # Decision threshold: chosen NOW, from validation data only, before
        # the final retrain and long before the test set is touched.
        # --------------------------------------------------------------
        if best.val_proba is None or len(best.val_proba) != len(y_val):
            raise RuntimeError(
                "Best trial has no validation predictions aligned with the "
                "validation labels; cannot report genuine validation metrics."
            )
        best_val_proba = np.asarray(best.val_proba, dtype=np.float64)
        threshold_selection = select_decision_threshold(
            self.cfg.training.threshold_strategy,
            self.cfg.training.threshold,
            y_validation=y_val,
            proba_validation=best_val_proba,
        )
        threshold = float(threshold_selection["threshold"])
        logger.info(
            "  Threshold (%s) = %.3f, selected on: %s",
            threshold_selection["strategy"], threshold, threshold_selection["selected_on"],
        )

        # GENUINE validation metrics: the best trial's model was fitted on the
        # TRAIN split only and has never seen these validation patients.
        val_metrics = evaluate(y_val, best_val_proba, threshold=threshold, prefix="val_")

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

        if best_representation == REPRESENTATION_WIDE:
            # Refit wide preprocessor on train+val
            train_val_wide = wide_builder.build(train_val_df, train_val_targets)
            final_wide_prep = WidePreprocessor()
            X_trainval = final_wide_prep.fit_transform(train_val_wide.X)
            X_final_test = final_wide_prep.transform(test_wide.X)
            final_seq_prep = None
            # Labels aligned to the builder's row order (not the concat order
            # of the shuffled splitter targets).
            y_trainval = _aligned_labels(train_val_wide, "train+val")
            retrain_policy = {
                "model": best_model_name,
                "policy": "refit_on_train_plus_validation",
                "hyperparameters": "selected search hyperparameters, unchanged",
                "training_pool": "train + validation patients (all)",
                "preprocessor_fit_on": "train + validation patients",
                "early_stopping": "none (non-iterative / fixed-size model)",
                "test_set_used": False,
                "differences_from_search": [
                    "training patients: train+validation instead of the train split",
                    "preprocessor statistics re-estimated on train+validation",
                ],
            }
        else:
            # Sequence (GRU) retrain.
            #
            # The GRU needs a validation signal so early stopping and
            # best-epoch restoration stay active (see GRUModel.fit). To keep
            # the held-out TEST set completely untouched, we carve an INTERNAL
            # validation split out of the combined original train+val patients.
            #
            # Preprocessing is fitted on the INTERNAL TRAIN subset only and
            # applied unchanged to internal-val and test, so internal-val does
            # not leak into the preprocessor statistics.
            tv_patients = (
                train_val_df["patient_id"].drop_duplicates().tolist()
            )
            internal_train_ids, internal_val_ids = _internal_patient_split(
                tv_patients,
                train_val_targets,
                val_fraction=self.cfg.split.validation_size
                / (self.cfg.split.train_size + self.cfg.split.validation_size),
                seed=self.cfg.split.random_seed,
                # Always label-stratified (when both classes exist), regardless of
                # split.stratify: the internal holdout drives early stopping and
                # must contain both classes. This does not change the outer
                # train/validation/test partitions.
                stratify=True,
            )
            self._last_internal_split = (
                list(internal_train_ids),
                list(internal_val_ids),
            )
            # Early stopping scores validation AUROC, which is undefined for a
            # single class (it would silently restore the epoch-0 weights).
            if train_val_targets.nunique() >= 2:
                for part, ids in (
                    ("internal-train", internal_train_ids),
                    ("internal-validation", internal_val_ids),
                ):
                    if train_val_targets.loc[ids].nunique() < 2:
                        raise RuntimeError(
                            f"GRU retrain {part} partition contains a single class; "
                            "early stopping would be undefined."
                        )

            itrain_df = train_val_df[
                train_val_df["patient_id"].isin(internal_train_ids)
            ].copy()
            ival_df = train_val_df[
                train_val_df["patient_id"].isin(internal_val_ids)
            ].copy()
            itrain_targets = train_val_targets.loc[internal_train_ids]
            ival_targets = train_val_targets.loc[internal_val_ids]

            # Build sequence representations using the TRAINING vocabulary
            # (no vocabulary leakage) for each internal subset and the test set.
            itrain_seq = SequenceRepresentationBuilder(
                max_seq_len=max_seq,
                categorical_vocabularies=dict(train_seq.categorical_vocabularies),
            ).build(itrain_df, itrain_targets)
            ival_seq = SequenceRepresentationBuilder(
                max_seq_len=max_seq,
                categorical_vocabularies=dict(train_seq.categorical_vocabularies),
            ).build(ival_df, ival_targets)

            # Fit the sequence preprocessor on INTERNAL TRAIN only.
            final_seq_prep = SequencePreprocessor(
                feature_names=itrain_seq.feature_names
            )
            X_trainval = final_seq_prep.fit_transform(
                itrain_seq.X, itrain_seq.mask
            )
            X_internal_val = final_seq_prep.transform(ival_seq.X, ival_seq.mask)
            X_final_test = final_seq_prep.transform(test_seq.X, test_seq.mask)
            final_wide_prep = None

            # Use the builder's own target Series (result.y), which is aligned
            # to the sorted patient order of X — not the raw target index order.
            y_trainval = itrain_seq.y.values.astype(np.int32)
            y_internal_val = ival_seq.y.values.astype(np.int32)
            mask_trainval = itrain_seq.mask
            lengths_trainval = _lengths(itrain_seq.mask)
            mask_internal_val = ival_seq.mask
            lengths_internal_val = _lengths(ival_seq.mask)

            # The retrained network must see exactly the same input layout as the
            # architecture that was selected in the search.
            if (
                list(itrain_seq.feature_names) != list(train_seq.feature_names)
                or X_trainval.shape[1:] != X_train_seq.shape[1:]
            ):
                raise RuntimeError(
                    "GRU retrain input layout differs from the search layout "
                    f"({X_trainval.shape[1:]} vs {X_train_seq.shape[1:]}); the selected "
                    "architecture would not match."
                )

            retrain_policy = {
                "model": best_model_name,
                "policy": "internal_holdout_early_stopping",
                "hyperparameters": "selected search hyperparameters, unchanged "
                                   "(including epochs, patience and seed)",
                "training_pool": "train + validation patients",
                "internal_split": {
                    "stratified_by_label": True,
                    "seed": self.cfg.split.random_seed,
                    "holdout_fraction_of_pool": round(len(internal_val_ids) / len(tv_patients), 4),
                    "n_internal_train": len(internal_train_ids),
                    "n_internal_validation": len(internal_val_ids),
                },
                "preprocessor_fit_on": "internal-train patients only",
                "vocabulary_from": "original train split",
                "early_stopping": "validation AUROC on the internal holdout; "
                                  "best-epoch weights restored",
                "test_set_used": False,
                "differences_from_search": [
                    "training patients: a stratified share of train+validation instead of "
                    "the train split",
                    "early-stopping data: the internal holdout instead of the validation split",
                    "preprocessor statistics re-estimated on the internal-train patients",
                ],
            }

        final_model = build_model(best_model_name, best_hyperparams)

        if best_model_name == "gru":
            # Pass the internal validation set so GRUModel.fit runs early
            # stopping and restores the best-epoch weights before saving.
            final_model.fit(
                X_trainval,
                y_trainval,
                mask=mask_trainval,
                lengths=lengths_trainval,
                X_val=X_internal_val,
                y_val=y_internal_val,
                mask_val=mask_internal_val,
                lengths_val=lengths_internal_val,
            )
        else:
            final_model.fit(X_trainval, y_trainval)

        retrain_seconds = time.perf_counter() - t_retrain_start

        # ------------------------------------------------------------------
        # Step 10: Final test evaluation — exactly once
        # ------------------------------------------------------------------
        logger.info("Step 10/10: Final test evaluation (one-time, never repeated)")
        # `threshold` was fixed during model selection (validation data only)
        # and is applied to the test set unchanged.

        t_inf = time.perf_counter()
        if best_model_name == "gru":
            test_proba = final_model.predict_proba_with_mask(
                X_final_test, test_seq.mask
            )
        else:
            test_proba = final_model.predict_proba(X_final_test)
        inf_per_patient = (time.perf_counter() - t_inf) / max(len(test_proba), 1)

        test_metrics = evaluate(y_test, test_proba, threshold=threshold, prefix="test_")

        # Final retrained model on the validation patients. These patients were
        # in its training data (fully for wide models; partly for the GRU's
        # internal split), so this is RESUBSTITUTION performance and is reported
        # under its own "resub_" prefix, never as validation performance.
        if best_representation == REPRESENTATION_WIDE:
            val_proba_final = final_model.predict_proba(
                final_wide_prep.transform(val_wide.X)
            )
        else:
            val_proba_final = final_model.predict_proba_with_mask(
                final_seq_prep.transform(val_seq.X, val_seq.mask),
                val_seq.mask,
            )
        resub_metrics = evaluate(
            y_val, val_proba_final, threshold=threshold, prefix="resub_"
        )

        logger.info(
            "  Test AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
            test_metrics.get("test_auroc") or 0.0,
            test_metrics.get("test_auprc") or 0.0,
            test_metrics.get("test_f1") or 0.0,
        )

        # ------------------------------------------------------------------
        # Reference baselines B0-B3 (Phase 13C) — same patients, same policy
        # ------------------------------------------------------------------
        if self.cfg.baselines.enabled:
            baselines_record = self._run_baselines(
                wide_builder=wide_builder,
                split=split,
                train_wide=train_wide,
                val_wide=val_wide,
                test_wide=test_wide,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                train_val_df=train_val_df,
                train_val_targets=train_val_targets,
            )
        else:
            baselines_record = {"enabled": False}

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
        study_hash = best_cfg.get("study_identity_hash")
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
            "threshold_strategy": threshold_selection["strategy"],
            "threshold_selection": threshold_selection,
            "validation_metrics": {
                k: v for k, v in val_metrics.items()
                if not isinstance(v, dict)
            },
            "validation_metrics_source": VALIDATION_METRICS_SOURCE,
            "retrain_resubstitution_metrics": {
                k: v for k, v in resub_metrics.items()
                if not isinstance(v, dict)
            },
            "retrain_resubstitution_metrics_note": RESUBSTITUTION_NOTE,
            "test_metrics": {
                k: v for k, v in test_metrics.items()
                if not isinstance(v, dict)
            },
            "retrain_policy": retrain_policy,
            "optuna_study_name": optimizer.study_name,
            "optuna_study_identity_hash": study_hash,
            "optuna_sampler": optimizer.sampler_name,
            "optuna_pruner": optimizer.pruner_name,
            "optuna_n_trials_failed": len(failed_trials),
            "optuna_trial_states": optimizer.trial_state_counts(),
            "optuna_n_trials_completed": n_trials_done,
            "optuna_selection_criterion": "validation_auroc",
            "test_set_accessed_for_selection": False,
            "total_run_seconds": round(total_seconds, 2),
            "limitations": [
                "Trained on synthetic demonstration data only.",
                "Not validated on real patient data.",
                "No clinical safety evaluation performed.",
                "GRU trained on CPU; production would benefit from GPU.",
                LIMITATIONS_VALIDATION,
            ],
        }
        metadata["baselines"] = baselines_record
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
            "threshold": threshold,
            "threshold_selection": threshold_selection,
            "validation_metrics": {
                k: v for k, v in val_metrics.items()
                if not isinstance(v, dict)
            },
            "validation_metrics_source": VALIDATION_METRICS_SOURCE,
            "retrain_resubstitution_metrics": {
                k: v for k, v in resub_metrics.items()
                if not isinstance(v, dict)
            },
            "retrain_resubstitution_metrics_note": RESUBSTITUTION_NOTE,
            "retrain_policy": retrain_policy,
            "optuna_study_name": optimizer.study_name,
            "optuna_study_identity_hash": study_hash,
            "optuna_sampler": optimizer.sampler_name,
            "optuna_pruner": optimizer.pruner_name,
            "optuna_n_trials_failed": len(failed_trials),
            "optuna_trial_states": optimizer.trial_state_counts(),
            "limitations": [LIMITATIONS_VALIDATION],
            "test_metrics": {
                k: v for k, v in test_metrics.items()
                if not isinstance(v, dict)
            },
            "total_run_seconds": round(total_seconds, 2),
            "optuna_seconds": round(optuna_seconds, 2),
            "retrain_seconds": round(retrain_seconds, 2),
            "inf_time_per_patient_seconds": round(inf_per_patient, 6),
        }
        run_summary["baselines"] = baselines_record
        summary_path = outputs_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(run_summary, fh, indent=2, default=str)

        # Side-by-side table: selected model first, then each baseline.
        comparison_path: Path | None = None
        if baselines_record.get("enabled"):
            from dialong_automl.baselines import comparison_rows

            comparison_path = outputs_dir / "baselines_comparison.csv"
            pd.DataFrame(comparison_rows(
                selected_model=best_model_name,
                model_threshold=threshold,
                model_val_metrics=val_metrics,
                model_test_metrics=test_metrics,
                baseline_results=baselines_record["results"],
            )).to_csv(comparison_path, index=False)

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
        if failed_trials_path is not None:
            artifact_paths["failed_trials"] = str(failed_trials_path)
        if comparison_path is not None:
            artifact_paths["baselines_comparison"] = str(comparison_path)
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
            resubstitution_metrics=resub_metrics,
            threshold=threshold,
            threshold_selection=threshold_selection,
            study_name=optimizer.study_name,
            study_identity_hash=study_hash,
            n_trials_failed=len(failed_trials),
            failed_trials=failed_trials,
            retrain_policy=retrain_policy,
            baselines=baselines_record,
        )

        logger.info("Pipeline complete. %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Reference baselines (Phase 13C)
    # ------------------------------------------------------------------

    def _run_baselines(
        self,
        *,
        wide_builder: WideRepresentationBuilder,
        split: Any,
        train_wide: Any,
        val_wide: Any,
        test_wide: Any,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        train_val_df: pd.DataFrame,
        train_val_targets: pd.Series,
    ) -> dict[str, Any]:
        """Fit B0-B3 with the selected model's policy and return the record.

        Fit on train, threshold on validation, refit on train + validation,
        then score the test split once. Uses the pipeline's own patient
        partitions and builder-aligned labels, so baselines and the selected
        model are scored on exactly the same patients. Test inputs are only
        constructed for the final ``evaluate_test`` call.
        """
        from dialong_automl.baselines import BaselineInputs, BaselineRunner

        logger.info("Reference baselines B0-B3: fit on train, threshold on validation")
        runner = BaselineRunner.from_config(self.cfg)
        runner.fit_and_validate(
            BaselineInputs(train_wide.X, list(train_wide.patient_ids), split.train_df, y_train),
            BaselineInputs(val_wide.X, list(val_wide.patient_ids), split.validation_df, y_val),
        )

        tv_wide = wide_builder.build(train_val_df, train_val_targets)
        runner.refit(BaselineInputs(
            tv_wide.X, list(tv_wide.patient_ids), train_val_df,
            _aligned_labels(tv_wide, "train+val (baselines)"),
        ))

        runner.evaluate_test(
            BaselineInputs(test_wide.X, list(test_wide.patient_ids), split.test_df, y_test)
        )
        record = runner.to_dict()
        for name, r in record["results"].items():
            logger.info(
                "  %-24s val_auroc=%s test_auroc=%s threshold=%.4f",
                name, r["validation_metrics"].get("val_auroc"),
                r["test_metrics"].get("test_auroc"), r["threshold"],
            )
        return record
