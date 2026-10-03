"""Optuna-based mixed-model AutoML optimizer for DiaLong-AutoML.

Scientific rules enforced here
-------------------------------
1. Optuna objective uses ONLY validation AUROC — never test AUROC.
2. The test set DataFrames / arrays are never passed into or accessed by
   this module.  The optimizer API accepts only train + validation data.
3. Preprocessing is fitted on training data only, then applied to
   validation data using the training-fitted preprocessor.
4. Threshold selection is deferred to the training pipeline and uses the
   validation set only.
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

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

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
        Stable name so the study survives process restarts.
    direction:
        ``"maximize"`` (default) or ``"minimize"``.
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
    ) -> None:
        self.n_trials = n_trials
        self.seed = seed
        self.storage = storage
        self.study_name = study_name
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

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner()

        self._study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=True,
        )

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

        self._study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            show_progress_bar=False,
        )

        logger.info(
            "Optuna finished: %d completed trials, best_value=%.4f",
            len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            self._study.best_value if self._study.best_trial else float("nan"),
        )
        return self._study

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

        # 3. Select appropriate training data
        if representation == REPRESENTATION_SEQUENCE:
            if self._X_train_seq is None or self._X_val_seq is None:
                # GRU selected but no sequence data provided — skip
                raise optuna.exceptions.TrialPruned("Sequence data not available for GRU.")
            X_train = self._X_train_seq
            X_val = self._X_val_seq
        else:
            X_train = self._X_train_wide
            X_val = self._X_val_wide

        # 4. Train
        t_start = time.perf_counter()
        model = build_model(model_name, hp)
        try:
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
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            logger.warning("Trial %d training failed: %s", trial.number, exc)
            raise optuna.exceptions.TrialPruned(f"Training failed: {exc}") from exc

        train_time = time.perf_counter() - t_start

        # 5. Predict on VALIDATION only — never test
        t_inf_start = time.perf_counter()
        try:
            if model_name == "gru":
                val_proba = model.predict_proba_with_mask(X_val, self._mask_val)
            else:
                val_proba = model.predict_proba(X_val)
        except Exception as exc:
            logger.warning("Trial %d inference failed: %s", trial.number, exc)
            raise optuna.exceptions.TrialPruned(f"Inference failed: {exc}") from exc
        inf_time = (time.perf_counter() - t_inf_start) / max(len(val_proba), 1)

        # 6. Evaluate on VALIDATION only
        metrics = evaluate(self._y_val, val_proba, threshold=self.threshold)
        val_auroc = metrics.get("auroc")
        if val_auroc is None:
            logger.warning(
                "Trial %d: validation AUROC is undefined (single class or error). "
                "Pruning rather than fabricating a score.",
                trial.number,
            )
            raise optuna.exceptions.TrialPruned("Validation AUROC undefined.")

        # 7. Store trial result
        self._trial_results.append(TrialResult(
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
        ))

        # Store metrics as trial user attributes for later inspection
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                trial.set_user_attr(k, v)
        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("representation", representation)
        trial.set_user_attr("train_time_seconds", round(train_time, 4))
        trial.set_user_attr("inf_time_per_patient", round(inf_time, 6))

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
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        logger.info("Best config saved → %s (model=%s, val_auroc=%.4f)",
                    path, best.model_name, best.val_auroc)
        return cfg
