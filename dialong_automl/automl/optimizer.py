"""Optuna-based mixed-model AutoML optimizer for DiaLong-AutoML.

Scientific rules enforced here
-------------------------------
1. Optuna objective uses ONLY validation AUROC — never test AUROC.
2. The test set DataFrames / arrays are never passed into or accessed by
   this module.  The optimizer API accepts only train + validation data.
3. Preprocessing is fitted on training data only, then applied to
   validation data using the training-fitted preprocessor.
4. Threshold selection is deferred to the training pipeline and uses the
   validation set only (via each trial's stored validation probabilities).
6. Study isolation: the effective study name embeds a hash of the experiment
   identity; a same-identity study is only reused on explicit ``resume``.
5. TCN is explicitly excluded from the search space.

Search space
------------
model_name in {logistic_regression, random_forest, xgboost, lightgbm, gru}
Each model family has its own hyperparameter search sub-space (see
``_suggest_hyperparams``).

Leaderboard
-----------
After the study completes, a leaderboard CSV is written to
``outputs/leaderboard.csv``.  Only validation metrics appear here;
test metrics are stored separately after the final retrain.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from dialong_automl.config.schema import (
    SUPPORTED_OPTUNA_DIRECTIONS,
    SUPPORTED_OPTUNA_PRUNERS,
    SUPPORTED_OPTUNA_SAMPLERS,
)
from dialong_automl.evaluation.metrics import evaluate
from dialong_automl.models import build_model, REGISTRY
from dialong_automl.models.base import REPRESENTATION_WIDE, REPRESENTATION_SEQUENCE

logger = logging.getLogger(__name__)

# Models excluded from Optuna search — must include TCN
_EXCLUDED_FROM_SEARCH: frozenset[str] = frozenset({"tcn"})


def resolve_optuna_storage(storage: str | None, artifacts_dir: str | Path | None = None) -> str | None:
    """Resolve a relative sqlite URL to an absolute path under CWD.

    ``sqlite:///artifacts/optuna_study.db`` is written to
    ``<cwd>/artifacts/optuna_study.db`` and is not nested a second time
    under *artifacts_dir*.
    """
    if not storage:
        return None
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return storage
    raw = storage[len(prefix) :]
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve().as_posix()}"

class StudyExistsError(RuntimeError):
    """A persisted study with the same identity already exists and resume is off."""


class StudyIdentityMismatchError(ValueError):
    """A persisted study does not match the requested experiment identity."""


class NoCompletedTrialsError(RuntimeError):
    """No trial completed and at least one trial FAILED (not merely pruned)."""


def make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    """Instantiate the configured, seeded sampler (unsupported names are rejected)."""
    if name == "TPESampler":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "RandomSampler":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported sampler {name!r}; supported: {SUPPORTED_OPTUNA_SAMPLERS}.")


def make_pruner(name: str) -> optuna.pruners.BasePruner:
    """Instantiate the configured pruner (unsupported names are rejected)."""
    if name == "MedianPruner":
        return optuna.pruners.MedianPruner()
    if name == "NopPruner":
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unsupported pruner {name!r}; supported: {SUPPORTED_OPTUNA_PRUNERS}.")


def canonical_identity_json(identity: dict[str, Any]) -> str:
    """Deterministic JSON for an identity dict (sorted keys, str fallback)."""
    return json.dumps(identity, sort_keys=True, default=str, separators=(",", ":"))


def study_identity_hash(identity: dict[str, Any]) -> str:
    """Short deterministic hash (sha256, 12 hex chars) of an identity dict."""
    return hashlib.sha256(canonical_identity_json(identity).encode("utf-8")).hexdigest()[:12]


def build_study_name(base_name: str, identity: dict[str, Any]) -> str:
    """Effective study name: ``<base>__<identity hash>``."""
    return f"{base_name}__{study_identity_hash(identity)}"


# Map each model to its required representation
_MODEL_REPRESENTATION: dict[str, str] = {
    "logistic_regression": REPRESENTATION_WIDE,
    "random_forest": REPRESENTATION_WIDE,
    "xgboost": REPRESENTATION_WIDE,
    "lightgbm": REPRESENTATION_WIDE,
    "gru": REPRESENTATION_SEQUENCE,
}


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Stores per-trial information for the leaderboard."""

    trial_number: int
    model_name: str
    representation: str
    state: str
    val_auroc: float | None
    val_auprc: float | None
    val_f1: float | None
    val_accuracy: float | None
    val_precision: float | None
    val_recall: float | None
    val_specificity: float | None
    val_brier: float | None
    train_time_seconds: float
    inference_time_per_patient_seconds: float
    hyperparams: dict[str, Any] = field(default_factory=dict)
    # Predicted probabilities of this trial's model (fitted on TRAIN only) on the
    # VALIDATION split. Used for genuine pre-retrain validation metrics and for
    # validation-based threshold selection. Not written to the leaderboard.
    val_proba: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "trial_number": self.trial_number,
            "model_name": self.model_name,
            "representation": self.representation,
            "state": self.state,
            "validation_auroc": self.val_auroc,
            "validation_auprc": self.val_auprc,
            "validation_f1": self.val_f1,
            "validation_accuracy": self.val_accuracy,
            "validation_precision": self.val_precision,
            "validation_recall": self.val_recall,
            "validation_specificity": self.val_specificity,
            "validation_brier": self.val_brier,
            "train_time_seconds": self.train_time_seconds,
            "inference_time_per_patient_seconds": self.inference_time_per_patient_seconds,
            "hyperparameters": json.dumps(self.hyperparams),
        }
        return d


# ---------------------------------------------------------------------------
# AutoML optimizer
# ---------------------------------------------------------------------------

class AutoMLOptimizer:
    """Mixed-model Optuna search optimizing validation AUROC.

    Parameters
    ----------
    n_trials:
        Maximum number of trials.
    seed:
        Seed for the TPE sampler.
    storage:
        SQLAlchemy connection string for study persistence.
        E.g. ``"sqlite:///artifacts/optuna_study.db"``.
    study_name:
        BASE study name. The effective name is ``<study_name>__<hash of identity>``.
    identity:
        JSON-serialisable description of the experiment (dataset identity,
        configuration, seeds, fold/run label ...). Required when ``storage`` is
        set, so that unrelated experiments can never share a persisted study.
    resume:
        If ``False`` (default) a persisted study with the same effective name must
        not exist (``StudyExistsError``). If ``True`` the existing study is
        resumed, earlier completed trials are re-hydrated into the leaderboard,
        and ``n_trials`` is the TOTAL number of trials wanted. Requires storage.
    sampler / pruner:
        Names from ``SUPPORTED_OPTUNA_SAMPLERS`` / ``SUPPORTED_OPTUNA_PRUNERS``;
        instantiated by ``make_sampler`` / ``make_pruner`` (seeded sampler).
        Unsupported names raise ``ValueError``.
    direction:
        ``"maximize"`` only (best-trial selection maximises validation AUROC).
    timeout_seconds:
        Wall-clock timeout (None = unlimited).
    enabled_models:
        Subset of model names to include in search.
        TCN is always excluded regardless of this list.
    threshold:
        Classification threshold for binary metrics (not AUROC selection).
    gru_cfg:
        Dict of GRU training settings (epochs, patience, etc.).
    """

    def __init__(
        self,
        n_trials: int = 5,
        seed: int = 42,
        storage: str | None = None,
        study_name: str = "dialong_automl_study",
        direction: str = "maximize",
        timeout_seconds: int | None = None,
        enabled_models: list[str] | None = None,
        threshold: float = 0.5,
        gru_cfg: dict[str, Any] | None = None,
        identity: dict[str, Any] | None = None,
        resume: bool = False,
        sampler: str = "TPESampler",
        pruner: str = "MedianPruner",
    ) -> None:
        if sampler not in SUPPORTED_OPTUNA_SAMPLERS:
            raise ValueError(
                f"Unsupported sampler {sampler!r}; supported: {SUPPORTED_OPTUNA_SAMPLERS}."
            )
        if pruner not in SUPPORTED_OPTUNA_PRUNERS:
            raise ValueError(
                f"Unsupported pruner {pruner!r}; supported: {SUPPORTED_OPTUNA_PRUNERS}."
            )
        if direction not in SUPPORTED_OPTUNA_DIRECTIONS:
            raise ValueError(
                f"Unsupported direction {direction!r}; supported: {SUPPORTED_OPTUNA_DIRECTIONS} "
                "(best-trial selection maximises validation AUROC)."
            )
        self.sampler_name = sampler
        self.pruner_name = pruner
        if storage and identity is None:
            raise ValueError(
                "A persistent Optuna storage requires an experiment 'identity' so "
                "that unrelated experiments cannot share a study."
            )
        if resume and not storage:
            raise ValueError("resume=True requires a persistent Optuna storage.")
        self.n_trials = n_trials
        self.seed = seed
        self.storage = storage
        self.base_study_name = study_name
        self.identity = dict(identity) if identity is not None else None
        self.resume = resume
        self.study_name = (
            build_study_name(study_name, self.identity)
            if self.identity is not None else study_name
        )
        self.direction = direction
        self.timeout_seconds = timeout_seconds
        self.threshold = threshold
        self.gru_cfg = gru_cfg or {}

        # Resolve enabled models, always excluding TCN
        all_available = [
            n for n in REGISTRY.list_all()
            if n not in _EXCLUDED_FROM_SEARCH and REGISTRY.get(n).is_available()
        ]
        if enabled_models is not None:
            self.enabled_models = [
                m for m in enabled_models
                if m not in _EXCLUDED_FROM_SEARCH and m in all_available
            ]
        else:
            self.enabled_models = all_available

        self._trial_results: list[TrialResult] = []
        self._study: optuna.Study | None = None
        self.n_interrupted_trials_ignored = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def optimize(
        self,
        # Wide representation data (for LR, RF, XGB, LGBM)
        X_train_wide: np.ndarray,
        y_train: np.ndarray,
        X_val_wide: np.ndarray,
        y_val: np.ndarray,
        # Sequence representation data (for GRU)
        X_train_seq: np.ndarray | None = None,
        mask_train: np.ndarray | None = None,
        lengths_train: np.ndarray | None = None,
        X_val_seq: np.ndarray | None = None,
        mask_val: np.ndarray | None = None,
        lengths_val: np.ndarray | None = None,
    ) -> optuna.Study:
        """Run Optuna search and return the completed study.

        Parameters
        ----------
        X_train_wide / X_val_wide:
            Preprocessed wide feature arrays (training and validation).
        y_train / y_val:
            Binary targets for training and validation.
        X_train_seq / X_val_seq:
            3-D padded sequence arrays (only required when GRU is enabled).
        mask_train / mask_val:
            Boolean mask arrays for sequence data.
        lengths_train / lengths_val:
            Actual sequence lengths.

        Returns
        -------
        optuna.Study
            The completed study with all trials.

        Critical
        --------
        Only y_val and X_val_* are used for the objective.
        The test set must NEVER be passed to or accessed by this method.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = make_sampler(self.sampler_name, self.seed)
        pruner = make_pruner(self.pruner_name)

        # --- Study isolation (Phase 13B) -----------------------------------
        # The effective study name embeds a hash of the experiment identity, so
        # a different dataset/seed/config maps to a different study. A study with
        # the SAME identity is only reused when the caller explicitly asks to
        # resume; otherwise we refuse instead of silently mixing trials.
        exists = False
        if self.storage:
            exists = self.study_name in optuna.study.get_all_study_names(self.storage)
            if exists and not self.resume:
                raise StudyExistsError(
                    f"An Optuna study named {self.study_name!r} already exists in the "
                    "configured storage. Refusing to silently reuse its trials. "
                    "Set optuna.resume=true to resume it intentionally, or change "
                    "optuna.run_id / the configuration to start an independent study."
                )

        self._study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=exists,  # True only for an intentional resume
        )

        identity_json = (
            canonical_identity_json(self.identity) if self.identity is not None else None
        )
        n_already_done = 0
        if exists:
            stored = self._study.user_attrs.get("identity_json")
            if stored != identity_json:
                raise StudyIdentityMismatchError(
                    f"Study {self.study_name!r} was created for a different experiment "
                    "identity; refusing to resume it."
                )
            self._rehydrate_trials(self._study)
            # "Finished" = COMPLETE + PRUNED + FAIL. n_trials is the TOTAL target
            # of finished trials; interrupted (RUNNING/WAITING) trials are not
            # counted and never enter the leaderboard.
            n_already_done = len([
                t for t in self._study.trials
                if t.state in (
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.PRUNED,
                    optuna.trial.TrialState.FAIL,
                )
            ])
            self.n_interrupted_trials_ignored = len([
                t for t in self._study.trials
                if t.state in (
                    optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING,
                )
            ])
            if self.n_interrupted_trials_ignored:
                logger.warning(
                    "Study %s contains %d interrupted (RUNNING/WAITING) trial(s) from an "
                    "earlier session; they are ignored and left untouched.",
                    self.study_name, self.n_interrupted_trials_ignored,
                )
            logger.info(
                "Resuming study %s with %d finished trials.", self.study_name, n_already_done
            )
        elif identity_json is not None:
            self._study.set_user_attr("identity_json", identity_json)
            self._study.set_user_attr("identity_hash", study_identity_hash(self.identity))

        # Store data on self for use inside the closure
        self._X_train_wide = X_train_wide
        self._y_train = np.asarray(y_train)
        self._X_val_wide = X_val_wide
        self._y_val = np.asarray(y_val)
        self._X_train_seq = X_train_seq
        self._mask_train = mask_train
        self._lengths_train = lengths_train
        self._X_val_seq = X_val_seq
        self._mask_val = mask_val
        self._lengths_val = lengths_val

        logger.info(
            "Optuna search: %d trials, models=%s, storage=%s",
            self.n_trials, self.enabled_models, self.storage,
        )

        # On resume, n_trials is the TOTAL wanted; only run the remainder.
        n_to_run = max(self.n_trials - n_already_done, 0) if exists else self.n_trials
        if n_to_run > 0:
            # catch=(Exception,): a FAILED trial is recorded as FAIL (with
            # diagnostics, see _record_failure) and the search continues, so one
            # bad hyperparameter combination cannot abort the study. It is never
            # turned into a PRUNED trial. KeyboardInterrupt/SystemExit still propagate.
            self._study.optimize(
                self._objective,
                n_trials=n_to_run,
                timeout=self.timeout_seconds,
                show_progress_bar=False,
                catch=(Exception,),
            )

        completed = [
            t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        # Reconcile: the leaderboard may only contain trials Optuna recorded as COMPLETE.
        complete_numbers = {t.number for t in completed}
        self._trial_results = [
            r for r in self._trial_results if r.trial_number in complete_numbers
        ]
        failed = self.failed_trials()
        best_value = max((t.value for t in completed), default=float("nan"))
        logger.info(
            "Optuna finished: %d completed, %d failed trials, best_value=%.4f",
            len(completed), len(failed), best_value,
        )
        if failed and not completed:
            first = failed[0]
            raise NoCompletedTrialsError(
                f"No Optuna trial completed; {len(failed)} trial(s) FAILED (this is not "
                f"pruning). First failure: trial {first['trial_number']} "
                f"[{first['model_name']}, stage={first['stage']}] "
                f"{first['exception_type']}: {first['message']}"
            )
        return self._study

    # ------------------------------------------------------------------
    # Failure diagnostics
    # ------------------------------------------------------------------

    def _record_failure(self, trial: optuna.Trial, stage: str, exc: BaseException) -> None:
        """Persist diagnostics on the trial; the caller re-raises so Optuna marks it FAIL."""
        trial.set_user_attr("failure_stage", stage)
        trial.set_user_attr("failure_type", type(exc).__name__)
        trial.set_user_attr("failure_message", str(exc)[:500])
        trial.set_user_attr("failure_traceback", traceback.format_exc()[-3000:])
        logger.warning(
            "Trial %d FAILED during %s (%s: %s). Recorded as failed, not pruned.",
            trial.number, stage, type(exc).__name__, exc,
        )

    def failed_trials(self) -> list[dict[str, Any]]:
        """Diagnostics for every FAILED trial of the study (including resumed ones).

        Failed trials never appear in the leaderboard or in best-trial selection.
        """
        if self._study is None:
            return []
        out: list[dict[str, Any]] = []
        for t in self._study.trials:
            if t.state != optuna.trial.TrialState.FAIL:
                continue
            a = t.user_attrs
            out.append({
                "trial_number": t.number,
                "model_name": a.get("model_name", t.params.get("model_name", "unknown")),
                "stage": a.get("failure_stage", "unknown"),
                "exception_type": a.get("failure_type", "unknown"),
                "message": a.get("failure_message", "no diagnostics recorded"),
                "traceback": a.get("failure_traceback"),
                "params": dict(t.params),
            })
        return out

    def trial_state_counts(self) -> dict[str, int]:
        """Number of trials per Optuna state (COMPLETE, PRUNED, FAIL, RUNNING, ...)."""
        counts: dict[str, int] = {}
        if self._study is not None:
            for t in self._study.trials:
                counts[t.state.name] = counts.get(t.state.name, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective — returns validation AUROC.

        The test set is NEVER accessed here.
        """
        # 1. Choose model
        model_name = trial.suggest_categorical("model_name", self.enabled_models)
        representation = _MODEL_REPRESENTATION.get(model_name, REPRESENTATION_WIDE)

        # 2. Suggest hyperparameters
        hp = self._suggest_hyperparams(trial, model_name)
        # Stored up front so that a FAILED trial is still diagnosable.
        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("representation", representation)
        trial.set_user_attr("hyperparams_json", json.dumps(hp, default=str))

        # Exception policy (Phase 13B.1):
        #   * optuna.TrialPruned  -> re-raised untouched (genuine pruning).
        #   * any other Exception -> diagnostics recorded, then re-raised, so
        #     Optuna marks the trial FAIL. Errors are NEVER turned into pruning.
        stage = "setup"
        try:
            # 3. Select appropriate training data
            if representation == REPRESENTATION_SEQUENCE:
                if self._X_train_seq is None or self._X_val_seq is None:
                    raise ValueError(
                        f"Model {model_name!r} needs sequence data but none was provided."
                    )
                X_train = self._X_train_seq
                X_val = self._X_val_seq
            else:
                X_train = self._X_train_wide
                X_val = self._X_val_wide

            # 4. Train
            stage = "training"
            t_start = time.perf_counter()
            model = build_model(model_name, hp)
            if model_name == "gru":
                model.fit(
                    X_train, self._y_train,
                    mask=self._mask_train,
                    lengths=self._lengths_train,
                    X_val=X_val,
                    y_val=self._y_val,
                    mask_val=self._mask_val,
                    lengths_val=self._lengths_val,
                    trial=trial,
                )
            else:
                model.fit(X_train, self._y_train)
            train_time = time.perf_counter() - t_start

            # 5. Predict on VALIDATION only — never test
            stage = "inference"
            t_inf_start = time.perf_counter()
            if model_name == "gru":
                val_proba = model.predict_proba_with_mask(X_val, self._mask_val)
            else:
                val_proba = model.predict_proba(X_val)
            inf_time = (time.perf_counter() - t_inf_start) / max(len(val_proba), 1)

            # 6. Evaluate on VALIDATION only
            stage = "evaluation"
            metrics = evaluate(self._y_val, val_proba, threshold=self.threshold)
            val_auroc = metrics.get("auroc")
            if val_auroc is None:
                raise ValueError(
                    "Validation AUROC is undefined (single-class validation labels or a "
                    "metric error); the trial is recorded as FAILED, not pruned."
                )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            self._record_failure(trial, stage, exc)
            raise

        # 7. Store trial result
        val_proba_list = [float(p) for p in np.asarray(val_proba).ravel()]
        result = TrialResult(
            trial_number=trial.number,
            model_name=model_name,
            representation=representation,
            state="complete",
            val_auroc=val_auroc,
            val_auprc=metrics.get("auprc"),
            val_f1=metrics.get("f1"),
            val_accuracy=metrics.get("accuracy"),
            val_precision=metrics.get("precision"),
            val_recall=metrics.get("recall"),
            val_specificity=metrics.get("specificity"),
            val_brier=metrics.get("brier"),
            train_time_seconds=round(train_time, 4),
            inference_time_per_patient_seconds=round(inf_time, 6),
            hyperparams=hp,
            val_proba=val_proba_list,
        )

        # Store metrics as trial user attributes for later inspection
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                trial.set_user_attr(k, v)
        trial.set_user_attr("train_time_seconds", round(train_time, 4))
        trial.set_user_attr("inf_time_per_patient", round(inf_time, 6))
        # Everything needed to rebuild this trial's TrialResult on resume.
        trial.set_user_attr("val_proba", val_proba_list)

        # Only now (all persistence succeeded) does the trial enter the
        # leaderboard, so the leaderboard can never contain a trial Optuna
        # did not record as COMPLETE.
        self._trial_results.append(result)

        logger.info(
            "Trial %d | %-22s | val_auroc=%.4f | train=%.2fs",
            trial.number, model_name, val_auroc, train_time,
        )
        return float(val_auroc)

    # ------------------------------------------------------------------
    # Hyperparameter suggestion
    # ------------------------------------------------------------------

    def _suggest_hyperparams(
        self, trial: optuna.Trial, model_name: str
    ) -> dict[str, Any]:
        """Suggest hyperparameters for *model_name*."""

        if model_name == "logistic_regression":
            return {
                "C": trial.suggest_float("lr_C", 1e-4, 100.0, log=True),
                "class_weight": trial.suggest_categorical(
                    "lr_class_weight", [None, "balanced"]
                ),
                "solver": trial.suggest_categorical("lr_solver", ["lbfgs", "saga"]),
                "max_iter": 2000,
                "random_state": self.seed,
            }

        if model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("rf_n_estimators", 100, 800),
                "max_depth": trial.suggest_categorical(
                    "rf_max_depth", [None, 5, 10, 15, 20, 30]
                ),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "rf_max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
                ),
                "class_weight": trial.suggest_categorical(
                    "rf_class_weight", [None, "balanced", "balanced_subsample"]
                ),
                "random_state": self.seed,
                "n_jobs": -1,
            }

        if model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 800),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
                "learning_rate": trial.suggest_float(
                    "xgb_learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "xgb_colsample_bytree", 0.6, 1.0
                ),
                "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float(
                    "xgb_reg_lambda", 1e-6, 20.0, log=True
                ),
                "random_state": self.seed,
                "n_jobs": -1,
            }

        if model_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 800),
                "learning_rate": trial.suggest_float(
                    "lgbm_learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("lgbm_num_leaves", 8, 128),
                "max_depth": trial.suggest_int("lgbm_max_depth", -1, 20),
                "min_child_samples": trial.suggest_int(
                    "lgbm_min_child_samples", 5, 100
                ),
                "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "lgbm_colsample_bytree", 0.6, 1.0
                ),
                "lambda_l1": trial.suggest_float(
                    "lgbm_lambda_l1", 1e-8, 10.0, log=True
                ),
                "lambda_l2": trial.suggest_float(
                    "lgbm_lambda_l2", 1e-8, 10.0, log=True
                ),
                "subsample_freq": 1,
                "random_state": self.seed,
                "n_jobs": -1,
                "verbose": -1,
            }

        if model_name == "gru":
            gru = self.gru_cfg
            num_layers = trial.suggest_int("gru_num_layers", 1, 3)
            dropout = trial.suggest_float("gru_dropout", 0.0, 0.5) if num_layers > 1 else 0.0
            return {
                "hidden_size": trial.suggest_categorical(
                    "gru_hidden_size", [32, 64, 128, 256]
                ),
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": trial.suggest_float(
                    "gru_lr", 1e-4, 3e-3, log=True
                ),
                "weight_decay": trial.suggest_float(
                    "gru_wd", 1e-7, 1e-3, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "gru_batch_size", [16, 32, 64]
                ),
                "epochs": int(gru.get("epochs", 20)),
                "patience": int(gru.get("patience", 4)),
                "seed": self.seed,
            }

        raise ValueError(f"No hyperparameter search space defined for {model_name!r}")

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def _rehydrate_trials(self, study: optuna.Study) -> None:
        """Rebuild ``TrialResult`` objects for COMPLETE trials of a resumed study.

        Ensures the leaderboard and best-trial selection reflect the whole
        resumed study (not only the trials run in this process).
        """
        self._trial_results = []
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            a = t.user_attrs
            required = (
                "model_name", "representation", "hyperparams_json", "val_proba",
            )
            missing = [k for k in required if k not in a]
            if missing or t.value is None:
                raise StudyIdentityMismatchError(
                    f"Completed trial {t.number} of study {study.study_name!r} lacks the "
                    f"stored metadata required to resume (missing: {missing or ['value']}). "
                    "The study is incompatible with this version; refusing to mix it into "
                    "this run. Start a new study (change optuna.run_id) instead."
                )
            try:
                hp = json.loads(a["hyperparams_json"])
            except (TypeError, ValueError) as exc:
                raise StudyIdentityMismatchError(
                    f"Trial {t.number} of study {study.study_name!r} has unreadable "
                    "hyperparameter metadata; refusing to resume."
                ) from exc
            self._trial_results.append(TrialResult(
                trial_number=t.number,
                model_name=a["model_name"],
                representation=a["representation"],
                state="complete",
                val_auroc=float(t.value),
                val_auprc=a.get("auprc"),
                val_f1=a.get("f1"),
                val_accuracy=a.get("accuracy"),
                val_precision=a.get("precision"),
                val_recall=a.get("recall"),
                val_specificity=a.get("specificity"),
                val_brier=a.get("brier"),
                train_time_seconds=float(a.get("train_time_seconds", 0.0)),
                inference_time_per_patient_seconds=float(a.get("inf_time_per_patient", 0.0)),
                hyperparams=hp,
                val_proba=a["val_proba"],
            ))

    # ------------------------------------------------------------------
    # Leaderboard + best-config
    # ------------------------------------------------------------------

    def build_leaderboard(self) -> pd.DataFrame:
        """Return a DataFrame of all completed trial results, sorted by val AUROC."""
        if not self._trial_results:
            return pd.DataFrame()
        rows = [r.to_dict() for r in self._trial_results]
        df = pd.DataFrame(rows)
        df = df.sort_values("validation_auroc", ascending=False).reset_index(drop=True)
        return df

    def best_trial_result(self) -> TrialResult | None:
        """Return the TrialResult with the highest validation AUROC."""
        completed = [r for r in self._trial_results if r.val_auroc is not None]
        if not completed:
            return None
        return max(completed, key=lambda r: r.val_auroc)

    def best_validation_metrics(self) -> dict[str, Any]:
        """Validation metrics for the selected trial (no test metrics)."""
        best = self.best_trial_result()
        if best is None:
            return {}
        return {
            "val_auroc": best.val_auroc,
            "val_auprc": best.val_auprc,
            "val_f1": best.val_f1,
            "val_accuracy": best.val_accuracy,
            "val_precision": best.val_precision,
            "val_recall": best.val_recall,
            "val_specificity": best.val_specificity,
            "val_brier": best.val_brier,
            "val_threshold": self.threshold,
        }

    def save_leaderboard(self, path: str | Path) -> None:
        """Write the leaderboard CSV to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.build_leaderboard()
        df.to_csv(path, index=False)
        logger.info("Leaderboard saved → %s (%d rows)", path, len(df))

    def save_best_config(self, path: str | Path) -> dict[str, Any]:
        """Write best_config.json to *path* and return it."""
        best = self.best_trial_result()
        if best is None:
            raise RuntimeError("No completed trials to select best config from.")
        cfg = {
            "trial_number": best.trial_number,
            "model_name": best.model_name,
            "representation": best.representation,
            "hyperparameters": best.hyperparams,
            "validation_auroc": best.val_auroc,
            "validation_auprc": best.val_auprc,
            "validation_f1": best.val_f1,
            "selection_criterion": "validation_auroc",
            "note": "Selected using validation AUROC only. Test set not accessed.",
            "study_name": self.study_name,
            "study_identity_hash": (
                study_identity_hash(self.identity) if self.identity is not None else None
            ),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        logger.info("Best config saved → %s (model=%s, val_auroc=%.4f)",
                    path, best.model_name, best.val_auroc)
        return cfg
