"""Unified evaluation runner: repeated nested patient-level CV (Phase 13E).

Protocol (per repeat r, per outer fold k)
-----------------------------------------
1. Outer split: stratified K-fold over eligible PATIENTS (seeded by r). The
   outer test fold is wrapped in :class:`HeldOutFold` and is read exactly once.
2. Inner split of the outer-training patients: stratified K-fold. For every
   inner fold the representations, categorical vocabulary and preprocessors
   are built/fitted on the inner-training patients only.
3. Model selection: Optuna (third-party library) over the existing search
   space; the objective is the MEAN inner-fold validation AUROC
   (:class:`CVAutoMLOptimizer`). Each trial also yields inner out-of-fold
   (OOF) probabilities for every outer-training patient.
4. Threshold: ``training.threshold_strategy`` applied to the best trial's
   inner OOF probabilities (never the outer test fold).
5. Final fit on all outer-training patients with the selected configuration
   (wide: refit; GRU: internal stratified holdout for early stopping, as in
   ``TrainingPipeline``), then a single prediction of the outer test fold.
6. Baselines B0-B3 follow the same policy: inner-CV OOF scores choose their
   thresholds, refit on outer-train, test once.

Pooled outer-test predictions (every eligible patient exactly once per
repeat) are summarised with patient-level bootstrap CIs and paired
comparisons (``dialong_automl.experiments.statistics``). Every run writes a
``RunResult`` with provenance to a NEW directory and never overwrites.

Optuna studies are in-memory and unique per (repeat, fold). Pruning is
inactive here: intermediate values are not reported across inner folds.

DISCLAIMER: synthetic demonstration data only. Not clinical evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from dialong_automl.automl.optimizer import _MODEL_REPRESENTATION, AutoMLOptimizer, TrialResult
from dialong_automl.baselines.predictors import BaselineInputs, default_baselines
from dialong_automl.baselines.runner import select_raw_score_threshold
from dialong_automl.config.schema import AppConfig
from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
from dialong_automl.data.synthetic_v2 import dataframe_sha256, generate_scenario
from dialong_automl.evaluation.metrics import auroc, evaluate
from dialong_automl.evaluation.result_schema import (
    ExperimentCondition,
    RunResult,
    config_hash,
    new_run_id,
)
from dialong_automl.evaluation.threshold import select_decision_threshold
from dialong_automl.experiments.statistics import ALL_METRICS, bootstrap_summary
from dialong_automl.features.preprocessing import SequencePreprocessor, WidePreprocessor
from dialong_automl.features.sequence import SequenceRepresentationBuilder
from dialong_automl.features.wide import WideRepresentationBuilder
from dialong_automl.models import build_model
from dialong_automl.models.base import REPRESENTATION_SEQUENCE, REPRESENTATION_WIDE
from dialong_automl.reliability.calibration import make_calibrator
from dialong_automl.reliability.conformal import MondrianConformal, coverage_summary, set_labels
from dialong_automl.reliability.ensemble import (
    RANGE_NOTE,
    member_subsamples,
    variability_range,
    width_summary,
)
from dialong_automl.reliability.ood import MahalanobisOOD
from dialong_automl.reliability.quality import quality_flags_by_patient
from dialong_automl.reliability.report import build_report, width_cutoffs
from dialong_automl.training.pipeline import (
    _aligned_labels,
    _assert_same_patient_order,
    _internal_patient_split,
)

logger = logging.getLogger(__name__)

MODEL_PREDICTOR = "automl_selected"
CALIBRATED_PREDICTOR = "automl_calibrated"     # Phase 14B: same model, calibrated output
SENSITIVITY_ALPHA = 0.20                       # Phase 14F: conformal sensitivity analysis
ORACLE_PREDICTOR = "oracle_latent_truth"
PROTOCOL_VERSION = "13e-1"
DISCLAIMER = ("Synthetic demonstration data only. Not clinical evidence. Does not "
              "establish clinical validity or real-world performance.")


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

@dataclass
class Partition:
    """One set of patients in every representation the models/baselines need."""

    ids: list[str]
    y: np.ndarray
    obs: pd.DataFrame
    wide_raw: pd.DataFrame
    X_wide: np.ndarray | None = None
    X_seq: np.ndarray | None = None
    mask: np.ndarray | None = None

    @property
    def lengths(self) -> np.ndarray:
        return np.maximum(self.mask.sum(axis=1).astype(np.int32), 1)

    def baseline_inputs(self) -> BaselineInputs:
        return BaselineInputs(self.wide_raw, list(self.ids), self.obs, self.y)


def _subset(obs: pd.DataFrame, targets: pd.Series,
            ids: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    ids = sorted(ids)
    sub = obs[obs["patient_id"].isin(set(ids))].sort_values(["patient_id", "visit_date"])
    return sub.reset_index(drop=True), targets.loc[ids]


def _represent(obs: pd.DataFrame, targets: pd.Series, ids: list[str], max_visits: int,
               vocab: dict | None, name: str) -> tuple[Partition, Any, Any]:
    sub, t = _subset(obs, targets, ids)
    wide = WideRepresentationBuilder(max_waves=max_visits).build(sub, t)
    seq = SequenceRepresentationBuilder(
        max_seq_len=max_visits, categorical_vocabularies=dict(vocab) if vocab else None,
    ).build(sub, t)
    _assert_same_patient_order(wide, seq, name)
    part = Partition(ids=list(wide.patient_ids), y=_aligned_labels(wide, name), obs=sub,
                     wide_raw=wide.X)
    return part, wide, seq


def prepare_pair(obs: pd.DataFrame, targets: pd.Series, fit_ids: list[str],
                 eval_ids: list[str], max_visits: int, name: str = "fold"
                 ) -> tuple[Partition, Partition, dict[str, Any]]:
    """Representations + preprocessors fitted on ``fit_ids`` only, applied to ``eval_ids``."""
    if set(fit_ids) & set(eval_ids):
        raise RuntimeError(f"{name}: fit and evaluation patients overlap.")
    fit, fit_wide, fit_seq = _represent(obs, targets, fit_ids, max_visits, None, f"{name}-fit")
    vocab = dict(fit_seq.categorical_vocabularies)
    ev, ev_wide, ev_seq = _represent(obs, targets, eval_ids, max_visits, vocab, f"{name}-eval")
    wide_prep = WidePreprocessor()
    fit.X_wide = wide_prep.fit_transform(fit_wide.X)
    ev.X_wide = wide_prep.transform(ev_wide.X)
    seq_prep = SequencePreprocessor(feature_names=fit_seq.feature_names)
    fit.X_seq, fit.mask = seq_prep.fit_transform(fit_seq.X, fit_seq.mask), fit_seq.mask
    ev.X_seq, ev.mask = seq_prep.transform(ev_seq.X, ev_seq.mask), ev_seq.mask
    return fit, ev, {"vocab": vocab, "wide_prep": wide_prep, "seq_prep": seq_prep}


class HeldOutFold:
    """Outer test patients (visits AND labels). Every read is counted.

    The selection code (:meth:`NestedCVRunner.run_fold`) receives only the
    outer-training visits and labels; the test fold exists only inside this
    object until :meth:`release` is called once for the final prediction.
    """

    def __init__(self, obs: pd.DataFrame, targets: pd.Series, ids: list[str]) -> None:
        self._ids = sorted(ids)
        self._obs, self._targets = _subset(obs, targets, self._ids)
        self.access_count = 0

    @property
    def n(self) -> int:
        return len(self._ids)

    def release(self) -> tuple[list[str], pd.DataFrame, pd.Series]:
        self.access_count += 1
        return list(self._ids), self._obs, self._targets


# ---------------------------------------------------------------------------
# Inner-CV Optuna objective
# ---------------------------------------------------------------------------

class CVAutoMLOptimizer(AutoMLOptimizer):
    """``AutoMLOptimizer`` whose objective is the mean inner-fold validation AUROC.

    Reuses the parent's search space, sampler/pruner handling, study creation,
    failure recording and leaderboard. Each completed trial stores inner OOF
    probabilities for all outer-training patients (``TrialResult.val_proba``,
    ordered like ``outer_ids``), so the threshold can be chosen without the
    outer test fold.
    """

    def optimize_cv(self, folds: list[tuple[Partition, Partition]],
                    outer_ids: list[str]) -> optuna.Study:
        self._cv_folds = folds
        self._cv_pos = {pid: i for i, pid in enumerate(outer_ids)}
        self._cv_y = np.full(len(outer_ids), -1, dtype=np.int32)
        for _, va in folds:
            for pid, yy in zip(va.ids, va.y):
                self._cv_y[self._cv_pos[pid]] = yy
        if (self._cv_y < 0).any():
            raise RuntimeError("Inner validation folds do not cover every outer-training patient.")
        tr, va = folds[0]
        # The parent stores these arrays; the overridden objective uses the folds.
        return self.optimize(
            X_train_wide=tr.X_wide, y_train=tr.y, X_val_wide=va.X_wide, y_val=va.y,
            X_train_seq=tr.X_seq, mask_train=tr.mask, lengths_train=tr.lengths,
            X_val_seq=va.X_seq, mask_val=va.mask, lengths_val=va.lengths,
        )

    @property
    def oof_labels(self) -> np.ndarray:
        return self._cv_y

    def _objective(self, trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model_name", self.enabled_models)
        representation = _MODEL_REPRESENTATION.get(model_name, REPRESENTATION_WIDE)
        hp = self._suggest_hyperparams(trial, model_name)
        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("representation", representation)
        trial.set_user_attr("hyperparams_json", json.dumps(hp, default=str))

        stage = "setup"
        t0 = time.perf_counter()
        try:
            oof = np.full(len(self._cv_pos), np.nan)
            fold_aurocs = []
            for tr, va in self._cv_folds:
                stage = "training"
                model = build_model(model_name, hp)
                if representation == REPRESENTATION_SEQUENCE:
                    # No trial reporting across folds (steps would collide), so
                    # pruning is inactive in nested CV.
                    model.fit(tr.X_seq, tr.y, mask=tr.mask, lengths=tr.lengths,
                              X_val=va.X_seq, y_val=va.y, mask_val=va.mask,
                              lengths_val=va.lengths)
                    stage = "inference"
                    proba = model.predict_proba_with_mask(va.X_seq, va.mask)
                else:
                    model.fit(tr.X_wide, tr.y)
                    stage = "inference"
                    proba = model.predict_proba(va.X_wide)
                stage = "evaluation"
                a = auroc(va.y, proba)
                if a is None:
                    raise ValueError("Inner-fold AUROC undefined (single-class inner fold).")
                fold_aurocs.append(a)
                for pid, pr in zip(va.ids, np.asarray(proba).ravel()):
                    oof[self._cv_pos[pid]] = float(pr)
            stage = "evaluation"
            if np.isnan(oof).any():
                raise RuntimeError("Out-of-fold predictions incomplete.")
            objective = float(np.mean(fold_aurocs))
            metrics = evaluate(self._cv_y, oof, threshold=self.threshold)
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            self._record_failure(trial, stage, exc)
            raise

        oof_list = [float(v) for v in oof]
        result = TrialResult(
            trial_number=trial.number, model_name=model_name, representation=representation,
            state="complete", val_auroc=objective, val_auprc=metrics.get("auprc"),
            val_f1=metrics.get("f1"), val_accuracy=metrics.get("accuracy"),
            val_precision=metrics.get("precision"), val_recall=metrics.get("recall"),
            val_specificity=metrics.get("specificity"), val_brier=metrics.get("brier"),
            train_time_seconds=round(time.perf_counter() - t0, 4),
            inference_time_per_patient_seconds=0.0, hyperparams=hp, val_proba=oof_list,
        )
        trial.set_user_attr("inner_fold_aurocs", [float(a) for a in fold_aurocs])
        trial.set_user_attr("auroc", objective)
        trial.set_user_attr("val_proba", oof_list)
        self._trial_results.append(result)
        return objective


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class NestedCVResult:
    run_id: str
    run_dir: Path
    summary: list[dict[str, Any]]
    comparisons: list[dict[str, Any]]
    folds: list[dict[str, Any]]
    predictions: pd.DataFrame
    run_result: RunResult
    cohort_audit: dict[str, Any] = field(default_factory=dict)


def _git_provenance(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                                 timeout=20, check=True)
            return out.stdout.strip()
        except Exception:  # noqa: BLE001 — provenance is best-effort
            return None
    status = git("status", "--porcelain")
    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
            "git_dirty": None if status is None else bool(status)}


def _package_versions() -> dict[str, str | None]:
    import importlib.metadata as md
    out: dict[str, str | None] = {}
    for p in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "torch", "optuna",
              "shap", "pydantic"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


class NestedCVRunner:
    """Runs the Phase 13E protocol on a synthetic scenario."""

    def __init__(self, cfg: AppConfig, repo_root: str | Path | None = None) -> None:
        self.cfg = cfg
        self.ncv = cfg.nested_cv
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        self.max_visits = cfg.cohort.max_observation_visits

    # ----------------------------------------------------------- data
    def generate(self):
        n = self.ncv
        return generate_scenario(n.scenario, n.n_patients, n.dataset_seed,
                                 compute_oracle=n.scenario == "v2_standard",
                                 oracle_draws=n.oracle_draws)

    def build_cohort(self, full_df: pd.DataFrame):
        c = self.cfg.cohort
        return CohortBuilder(CohortConfig(
            mode=CohortMode.FUTURE_OUTCOME,
            min_observation_visits=c.min_observation_visits,
            max_observation_visits=c.max_observation_visits,
            prediction_horizon_days=c.prediction_horizon_days,
            diabetes_onset_hba1c_threshold=c.diabetes_onset_hba1c_threshold,
            exclude_known_diabetes_at_baseline=c.exclude_known_diabetes_at_baseline,
            require_followup_after_cutoff=c.require_followup_after_cutoff,
            require_outcome_measurement=c.require_outcome_measurement,
        )).build(full_df)

    def run(self, run_name: str | None = None, write: bool = True) -> NestedCVResult:
        t0 = time.perf_counter()
        gen = self.generate()
        cohort = self.build_cohort(gen.full_df)
        return self.run_on_cohort(cohort.observation_data, cohort.patient_targets,
                                  generation_metadata=gen.metadata, truth=gen.truth_patients,
                                  cohort_audit=cohort.audit.to_dict(), run_name=run_name,
                                  write=write, started=t0)

    # ----------------------------------------------------------- protocol
    def outer_splits(self, ids: list[str], targets: pd.Series, repeat: int):
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=self.ncv.outer_folds, shuffle=True,
                              random_state=self.cfg.split.random_seed + repeat)
        ids_arr = np.array(sorted(ids))
        y = targets.loc[ids_arr].astype(int).to_numpy()
        for tr, te in skf.split(ids_arr, y):
            yield ids_arr[tr].tolist(), ids_arr[te].tolist()

    def _inner_folds(self, obs, targets, outer_ids, seed):
        from sklearn.model_selection import StratifiedKFold

        ids_arr = np.array(sorted(outer_ids))
        y = targets.loc[ids_arr].astype(int).to_numpy()
        skf = StratifiedKFold(n_splits=self.ncv.inner_folds, shuffle=True, random_state=seed)
        return [prepare_pair(obs, targets, ids_arr[a].tolist(), ids_arr[b].tolist(),
                             self.max_visits, name=f"inner{j}")
                for j, (a, b) in enumerate(skf.split(ids_arr, y))]

    def run_fold(self, repeat: int, fold: int, obs: pd.DataFrame, targets: pd.Series,
                 outer_ids: list[str], test: HeldOutFold) -> tuple[dict[str, Any], list[dict]]:
        """Select, threshold and fit on ``outer_ids`` only; then read ``test`` once.

        ``obs`` and ``targets`` must contain ONLY outer-training patients.
        """
        cfg = self.cfg
        outer_ids = sorted(outer_ids)
        if set(targets.index) != set(outer_ids) or set(test._ids) & set(obs["patient_id"]):
            raise RuntimeError("run_fold must receive outer-training patients only.")
        seed = cfg.optuna.seed + 100 * repeat + fold
        inner_seed = cfg.split.random_seed + 100 * repeat + fold
        inner = self._inner_folds(obs, targets, outer_ids, inner_seed)
        inner_pairs = [(f, e) for f, e, _ in inner]

        # --- model selection (inner CV) ---
        opt = CVAutoMLOptimizer(
            n_trials=self.ncv.n_trials, seed=seed, storage=None,
            study_name=f"{cfg.optuna.study_name}_ncv_r{repeat}_f{fold}",
            direction=cfg.optuna.direction, timeout_seconds=cfg.optuna.timeout_seconds,
            enabled_models=cfg.model.enabled_models, threshold=cfg.training.threshold,
            gru_cfg={"epochs": cfg.gru.epochs, "patience": cfg.gru.patience},
            sampler=cfg.optuna.sampler, pruner=cfg.optuna.pruner,
        )
        opt.optimize_cv(inner_pairs, outer_ids)
        best = opt.best_trial_result()
        y_oof = opt.oof_labels
        sel = select_decision_threshold(cfg.training.threshold_strategy, cfg.training.threshold,
                                        y_validation=y_oof,
                                        proba_validation=np.asarray(best.val_proba))
        if sel["strategy"] == "validation_f1":
            sel["selected_on"] = "inner-CV out-of-fold probabilities (outer-training patients)"
        sel["applied_to"] = "final model fitted on all outer-training patients"

        # --- Phase 14B: calibrator fitted on the same inner OOF probabilities ---
        calibrator = None
        if cfg.reliability.enabled:
            calibrator = make_calibrator(cfg.reliability.calibration_method).fit(
                np.asarray(best.val_proba), y_oof)
        # --- Phase 14C: conformal thresholds on the calibrated inner OOF ---
        conformal = None
        if calibrator is not None:
            conformal = MondrianConformal(cfg.reliability.conformal_alpha).fit(
                calibrator.transform(np.asarray(best.val_proba)), y_oof)
            # Sensitivity analysis only (design decision 3); never used by R6.
            conformal_sens = MondrianConformal(SENSITIVITY_ALPHA).fit(
                calibrator.transform(np.asarray(best.val_proba)), y_oof)

        # --- baselines: inner OOF -> threshold ---
        horizon = cfg.baselines.b2_horizon_days or cfg.cohort.prediction_horizon_days

        def make_baselines():
            return default_baselines(horizon_days=int(horizon), C=cfg.baselines.b3_c,
                                     max_iter=cfg.training.max_iter_sklearn,
                                     seed=cfg.training.seed)

        pos = {pid: i for i, pid in enumerate(outer_ids)}
        names = [b.name for b in make_baselines()]
        oof_b = {nm: np.full(len(outer_ids), np.nan) for nm in names}
        for tr, va in inner_pairs:
            for b in make_baselines():
                b.fit(tr.baseline_inputs())
                for pid, s in zip(va.ids, b.predict_scores(va.baseline_inputs())):
                    oof_b[b.name][pos[pid]] = s
        b_thresholds: dict[str, dict[str, Any]] = {}
        for b in make_baselines():
            s = oof_b[b.name]
            if b.is_probabilistic:
                info = select_decision_threshold(cfg.training.threshold_strategy,
                                                 cfg.training.threshold, y_oof, s)
            else:
                info = select_raw_score_threshold(
                    y_oof, s, cfg.cohort.diabetes_onset_hba1c_threshold)
            b_thresholds[b.name] = info

        # --- final fit on ALL outer-training patients (still no test data) ---
        train_part, _, _ = _represent(obs, targets, outer_ids, self.max_visits, None, "outer-train")
        final = self._fit_final(best.model_name, best.representation, best.hyperparams,
                                obs, targets, outer_ids)
        final_baselines = make_baselines()
        for b in final_baselines:
            b.fit(train_part.baseline_inputs())
        # --- Phase 14C: ensemble members on subsamples of outer-train ---
        members = []
        n_members = cfg.reliability.ensemble_members if calibrator is not None else 0
        if n_members:
            subs = member_subsamples(outer_ids, targets.loc[outer_ids].to_numpy(), n_members,
                                     cfg.reliability.ensemble_subsample_fraction,
                                     seed=cfg.training.seed + 1000 * repeat + fold)
            for sub_ids in subs:
                s_obs, s_t = _subset(obs, targets, sub_ids)
                members.append(self._fit_final(best.model_name, best.representation,
                                               best.hyperparams, s_obs, s_t, sub_ids))
        # --- Phase 14D: OOD reference and width cut-offs, outer-train only ---
        rel = cfg.reliability
        ood = MahalanobisOOD(percentiles=rel.ood_percentiles).fit(train_part.wide_raw) \
            if calibrator is not None else None
        cutoffs = None
        if members:
            # Label-free: members' range widths on the outer-training patients.
            tr_mp = [calibrator.transform(np.asarray(m(outer_ids, obs, targets)[1]))
                     for m in members]
            cutoffs = width_cutoffs(variability_range(np.vstack(tr_mp))["width"],
                                    rel.width_percentiles)

        # --- the ONLY read of the outer test fold ---
        test_ids, test_obs, test_targets = test.release()
        test_part, model_scores = final(test_ids, test_obs, test_targets)
        rows = []
        for pid, yy, s in zip(test_part.ids, test_part.y, model_scores):
            rows.append({"repeat": repeat, "fold": fold, "patient_id": pid, "y": int(yy),
                         "predictor": MODEL_PREDICTOR, "score": float(s),
                         "threshold": float(sel["threshold"]), "is_probabilistic": True})
        calibrated_threshold = None
        if calibrator is not None:
            # The decision stays defined on the raw probability; the threshold is
            # mapped through the (monotone) calibrator for display and metrics.
            calibrated_threshold = float(calibrator.transform(np.array([sel["threshold"]]))[0])
            cal_scores = calibrator.transform(np.asarray(model_scores))
            has0, has1 = conformal.predict_sets(cal_scores)
            s0, s1 = conformal_sens.predict_sets(cal_scores)
            rng_ = None
            if members:
                # Same single read of the test fold; every member scores it.
                mp = []
                for m in members:
                    m_part, m_scores = m(test_ids, test_obs, test_targets)
                    if list(m_part.ids) != list(test_part.ids):
                        raise RuntimeError("Ensemble member scored patients in a different order.")
                    mp.append(calibrator.transform(np.asarray(m_scores)))
                rng_ = variability_range(np.vstack(mp))
            qflags = quality_flags_by_patient(test_part.obs, self.max_visits, rel.hba1c_age_days)
            ood_pct = ood.percentile(test_part.wide_raw)
            ood_cat = ood.category(test_part.wide_raw)
            labels = set_labels(has0, has1)
            for i, (pid, yy, s) in enumerate(zip(test_part.ids, test_part.y, cal_scores)):
                row = {"repeat": repeat, "fold": fold, "patient_id": pid, "y": int(yy),
                       "predictor": CALIBRATED_PREDICTOR, "score": float(s),
                       "threshold": calibrated_threshold, "is_probabilistic": True,
                       "set_no_onset": bool(has0[i]), "set_onset": bool(has1[i]),
                       "sens_set_no_onset": bool(s0[i]), "sens_set_onset": bool(s1[i])}
                if rng_ is not None:
                    row.update(range_low=float(rng_["low"][i]), range_high=float(rng_["high"][i]),
                               range_width=float(rng_["width"][i]))
                flags = qflags.get(str(pid), [])
                rep = build_report(flags, labels[i],
                                   row.get("range_low"), row.get("range_high"), cutoffs,
                                   ood_cat[i], float(ood_pct[i]))
                row.update(ood_percentile=float(ood_pct[i]), ood_category=ood_cat[i],
                           flags=";".join(f.code for f in rep.flags),
                           reliability_level=rep.level)
                rows.append(row)
        for b in final_baselines:
            scores = b.predict_scores(test_part.baseline_inputs())
            thr = float(b_thresholds[b.name]["threshold"])
            for pid, yy, s in zip(test_part.ids, test_part.y, scores):
                rows.append({"repeat": repeat, "fold": fold, "patient_id": pid, "y": int(yy),
                             "predictor": b.name, "score": float(s), "threshold": thr,
                             "is_probabilistic": b.is_probabilistic})
        record = {
            "repeat": repeat, "fold": fold, "n_train": len(outer_ids), "n_test": test.n,
            "n_train_positive": int(targets.loc[outer_ids].sum()),
            "n_test_positive": int(test_part.y.sum()),
            "test_patient_ids_sha256": hashlib.sha256(",".join(test_part.ids).encode()).hexdigest(),
            "selected_model": best.model_name, "representation": best.representation,
            "hyperparameters": best.hyperparams, "inner_cv_auroc_best": best.val_auroc,
            "threshold": float(sel["threshold"]), "threshold_selection": sel,
            "baseline_thresholds": {k: float(v["threshold"]) for k, v in b_thresholds.items()},
            "optuna_study_name": opt.study_name, "optuna_trial_states": opt.trial_state_counts(),
            "optuna_failed_trials": opt.failed_trials(),
            "test_access_count": test.access_count,
            "calibration": (None if calibrator is None else {
                **calibrator.to_dict(),
                "fit_partition": "inner-CV out-of-fold probabilities (outer-training patients)",
                "calibrated_threshold": calibrated_threshold,
            }),
            "conformal": (None if conformal is None else {
                **conformal.to_dict(),
                "fit_partition": "calibrated inner-CV out-of-fold probabilities "
                                 "(same patients as the calibrator; approximate guarantee)",
            }),
            "ensemble": {"members": len(members),
                         "subsample_fraction": cfg.reliability.ensemble_subsample_fraction,
                         "width_cutoffs": cutoffs,
                         "width_cutoffs_from": "member range widths on outer-training patients "
                                               "(label-free; members saw ~80 % of them)",
                         "note": RANGE_NOTE} if members else None,
            "ood": None if ood is None else {"n_fit": ood.n_fit, "cutoffs": ood.cutoffs,
                                             "percentiles": list(ood.percentiles),
                                             "fit_partition": "outer-training patients"},
        }
        return record, rows

    def _fit_final(self, model_name, representation, hp, obs, targets, outer_ids):
        """Fit the selected configuration on outer-train; return a test scorer."""
        cfg = self.cfg
        if representation == REPRESENTATION_WIDE:
            part, wide, seq = _represent(obs, targets, outer_ids, self.max_visits, None, "final")
            vocab = dict(seq.categorical_vocabularies)
            wide_prep = WidePreprocessor()
            X = wide_prep.fit_transform(wide.X)
            model = build_model(model_name, hp).fit(X, part.y)

            def score(test_ids, test_obs, test_targets):
                tp, tw, _ = _represent(test_obs, test_targets, test_ids, self.max_visits, vocab,
                                       "test")
                return tp, np.asarray(model.predict_proba(wide_prep.transform(tw.X))).ravel()
            return score

        # GRU: same policy as TrainingPipeline (internal stratified holdout).
        it_ids, iv_ids = _internal_patient_split(
            outer_ids, targets.loc[outer_ids],
            val_fraction=(cfg.split.validation_size
                          / (cfg.split.train_size + cfg.split.validation_size)),
            seed=cfg.split.random_seed, stratify=True,
        )
        fit, ival, preps = prepare_pair(obs, targets, it_ids, iv_ids, self.max_visits, "final-gru")
        model = build_model(model_name, hp)
        model.fit(fit.X_seq, fit.y, mask=fit.mask, lengths=fit.lengths, X_val=ival.X_seq,
                  y_val=ival.y, mask_val=ival.mask, lengths_val=ival.lengths)
        vocab, seq_prep = preps["vocab"], preps["seq_prep"]

        def score_seq(test_ids, test_obs, test_targets):
            tp, _, ts = _represent(test_obs, test_targets, test_ids, self.max_visits, vocab, "test")
            X = seq_prep.transform(ts.X, ts.mask)
            return tp, np.asarray(model.predict_proba_with_mask(X, ts.mask)).ravel()
        return score_seq

    # ----------------------------------------------------------- driver
    def run_on_cohort(self, obs: pd.DataFrame, targets: pd.Series,
                      generation_metadata: dict[str, Any] | None = None,
                      truth: pd.DataFrame | None = None, cohort_audit: dict | None = None,
                      run_name: str | None = None, write: bool = True,
                      started: float | None = None) -> NestedCVResult:
        started = started or time.perf_counter()
        targets = targets.astype(int)
        ids = sorted(targets.index.astype(str))
        targets.index = targets.index.astype(str)
        if targets.nunique() < 2:
            raise ValueError("Cohort must contain both classes.")
        folds, rows, assignment = [], [], {}
        for r in range(self.ncv.repeats):
            for k, (tr_ids, te_ids) in enumerate(self.outer_splits(ids, targets, r)):
                logger.info("Nested CV repeat %d fold %d: train=%d test=%d",
                            r, k, len(tr_ids), len(te_ids))
                held = HeldOutFold(obs, targets, te_ids)
                # Selection code only ever sees the outer-training patients.
                tr_obs, tr_targets = _subset(obs, targets, tr_ids)
                rec, fold_rows = self.run_fold(r, k, tr_obs, tr_targets, tr_ids, held)
                if held.access_count != 1:
                    raise RuntimeError(f"Outer test fold r{r}f{k} read {held.access_count} times.")
                folds.append(rec)
                rows.extend(fold_rows)
                assignment[f"r{r}f{k}"] = rec["test_patient_ids_sha256"]

        pred = pd.DataFrame(rows)
        if truth is not None and "oracle_prob" in truth.columns:
            tp = truth.set_index("patient_id")
            o = pred[pred.predictor == MODEL_PREDICTOR].copy()
            o["predictor"] = ORACLE_PREDICTOR
            o["score"] = tp.loc[o["patient_id"], "oracle_prob"].to_numpy()
            o["threshold"] = 0.5
            pred = pd.concat([pred, o], ignore_index=True)

        summary, comparisons = self.summarise(pred)
        split_hash = hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()
        result = self._record(summary, comparisons, folds, generation_metadata, cohort_audit,
                              split_hash, ids, targets, started)
        result.metadata["reliability"] = reliability_summary(pred, self.cfg.reliability)
        run_dir = (Path(self.cfg.paths.outputs_dir) / self.ncv.output_subdir
                   / (run_name or result.run_id))
        out = NestedCVResult(run_id=result.run_id, run_dir=run_dir, summary=summary,
                             comparisons=comparisons, folds=folds, predictions=pred,
                             run_result=result, cohort_audit=cohort_audit or {})
        if write:
            self._write(out, generation_metadata)
        return out

    def summarise(self, pred: pd.DataFrame):
        reps = sorted(pred["repeat"].unique())
        names = list(dict.fromkeys(pred["predictor"]))
        ids = sorted(pred.loc[pred.predictor == MODEL_PREDICTOR, "patient_id"].unique())
        scores, thresholds, prob = {}, {}, {}
        y = None
        fold_ids = None
        for nm in names:
            sub = pred[pred.predictor == nm]
            mats_s, mats_t, mats_f = [], [], []
            for r in reps:
                s = sub[sub.repeat == r].set_index("patient_id")
                if len(s) != len(ids) or s.index.duplicated().any():
                    raise RuntimeError(f"{nm}: patients not covered exactly once in repeat {r}.")
                s = s.loc[ids]
                mats_s.append(s["score"].to_numpy(float))
                mats_t.append(s["threshold"].to_numpy(float))
                mats_f.append(s["fold"].to_numpy(int))
                y = s["y"].to_numpy(int)
            scores[nm], thresholds[nm] = np.vstack(mats_s), np.vstack(mats_t)
            prob[nm] = bool(sub["is_probabilistic"].iloc[0])
            f = np.vstack(mats_f)
            if fold_ids is None:
                fold_ids = f
            elif not np.array_equal(fold_ids, f):
                raise RuntimeError(f"{nm}: outer-fold assignment differs between predictors.")
        return bootstrap_summary(y, scores, thresholds, prob, self.ncv.bootstrap_resamples,
                                 self.ncv.bootstrap_seed, reference=MODEL_PREDICTOR,
                                 primary=self.ncv.primary_comparator,
                                 not_comparators={ORACLE_PREDICTOR: "ceiling",
                                                  CALIBRATED_PREDICTOR: "calibrated_variant"},
                                 fold_ids=fold_ids)

    def _record(self, summary, comparisons, folds, gen_meta, cohort_audit, split_hash, ids,
                targets, started) -> RunResult:
        gen_meta = gen_meta or {}
        metrics: dict[str, float | None] = {}
        for row in summary:
            for m in ALL_METRICS:
                for suffix in ("", "_ci_low", "_ci_high"):
                    metrics[f"{row['predictor']}.{m}{suffix}"] = row.get(f"{m}{suffix}")
        selected = pd.Series([f["selected_model"] for f in folds]).value_counts().to_dict()
        meta = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol": self.ncv.model_dump(mode="json"),
            "scenario": gen_meta.get("scenario", self.ncv.scenario),
            "generator_version": gen_meta.get("generator_version"),
            "generator_params_hash": gen_meta.get("params_hash"),
            "dataset_sha256": gen_meta.get("dataset_sha256"),
            "split_hash": split_hash,
            "n_eligible_patients": len(ids), "n_positive": int(targets.loc[ids].sum()),
            "cohort_audit": cohort_audit,
            "selected_model_counts": selected,
            "test_access_counts": {f"r{f['repeat']}f{f['fold']}": f["test_access_count"]
                                   for f in folds},
            "threshold_strategy": self.cfg.training.threshold_strategy,
            "optuna": {"sampler": self.cfg.optuna.sampler, "pruner": self.cfg.optuna.pruner,
                       "storage": "in-memory, one study per (repeat, fold)",
                       "pruning_active": False, "library": "Optuna (third-party)"},
            "comparisons": comparisons,
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": _package_versions(), **_git_provenance(self.repo_root),
            "deterministic_torch": self.cfg.reproducibility.deterministic_torch,
            "runtime_seconds": round(time.perf_counter() - started, 1),
            "synthetic_data": True, "disclaimer": DISCLAIMER,
            "limitations": [
                "Synthetic data only; not clinical evidence.",
                "Bootstrap CIs resample patients of one fixed synthetic dataset; they do not "
                "cover variation across datasets or populations.",
                "The oracle uses latent ground truth and is an approximate upper bound, "
                "not a model.",
            ],
        }
        return RunResult(
            run_id=new_run_id(), config_hash=config_hash(self.cfg),
            seeds={"dataset": self.ncv.dataset_seed, "split": self.cfg.split.random_seed,
                   "optuna": self.cfg.optuna.seed, "training": self.cfg.training.seed,
                   "bootstrap": self.ncv.bootstrap_seed},
            dataset_id=f"{meta['scenario']}:{(meta['dataset_sha256'] or '')[:16]}",
            split_id=split_hash[:16], condition=ExperimentCondition(kind="baseline"),
            metrics=metrics, metadata=meta,
        )

    def _write(self, out: NestedCVResult, gen_meta) -> None:
        out.run_dir.mkdir(parents=True, exist_ok=False)   # never overwrite a run
        out.run_result.save(out.run_dir / "run_result.json")
        out.predictions.to_csv(out.run_dir / "predictions.csv", index=False)
        pd.DataFrame(out.summary).to_csv(out.run_dir / "summary.csv", index=False)
        pd.DataFrame(out.comparisons).to_csv(out.run_dir / "comparisons.csv", index=False)
        with (out.run_dir / "folds.json").open("w", encoding="utf-8") as fh:
            json.dump(out.folds, fh, indent=2, default=str)
        rel = out.run_result.metadata.get("reliability")
        if rel is not None:
            with (out.run_dir / "reliability_summary.json").open("w", encoding="utf-8") as fh:
                json.dump(rel, fh, indent=2)
        with (out.run_dir / "resolved_config.json").open("w", encoding="utf-8") as fh:
            json.dump(self.cfg.model_dump(mode="json"), fh, indent=2, default=str)
        if gen_meta:
            with (out.run_dir / "generation_metadata.json").open("w", encoding="utf-8") as fh:
                json.dump(gen_meta, fh, indent=2, default=str)
        logger.info("Nested CV run written to %s", out.run_dir)


def reliability_summary(pred: pd.DataFrame, rel_cfg) -> dict[str, Any] | None:
    """Conformal coverage and variability-range widths on the outer test folds.

    Coverage is computed per repeat on all patients (each scored once) and
    averaged over repeats; the minimum per-fold coverage is also reported.
    """
    cal = pred[pred.predictor == CALIBRATED_PREDICTOR]
    if cal.empty or "set_onset" not in cal.columns:
        return None
    per_rep = [coverage_summary(g["y"], g["set_no_onset"].astype(bool), g["set_onset"].astype(bool))
               for _, g in cal.groupby("repeat")]
    keys = [k for k in per_rep[0] if k.startswith(("coverage", "share_", "mean_set"))]
    out: dict[str, Any] = {
        "alpha": rel_cfg.conformal_alpha, "target_coverage": 1 - rel_cfg.conformal_alpha,
        "guarantee": "approximate in nested CV (calibration scores from inner-fold models)",
        "conformal": {k: float(np.mean([r[k] for r in per_rep])) for k in keys},
        "n_patients": per_rep[0]["n"],
    }
    fold_cov = []
    for _, g in cal.groupby(["repeat", "fold"]):
        s = coverage_summary(g["y"], g["set_no_onset"].astype(bool), g["set_onset"].astype(bool))
        fold_cov.append(s)
    for c in ("no_onset", "onset"):
        vals = [f[f"coverage_{c}"] for f in fold_cov if f[f"coverage_{c}"] is not None]
        out["conformal"][f"min_fold_coverage_{c}"] = float(min(vals)) if vals else None
    if "reliability_level" in cal.columns:
        # Descriptive only; the within-risk-stratum evaluation is Phase 14F.
        lv = {}
        for level, g in cal.groupby("reliability_level"):
            err = (g["score"] >= g["threshold"]).astype(int) != g["y"]
            lv[level] = {"share": float(len(g) / len(cal)),
                         "brier": float(np.mean((g["score"] - g["y"]) ** 2)),
                         "misclassification": float(err.mean()),
                         "onset_rate": float(g["y"].mean())}
        out["levels"] = lv
        codes = cal["flags"].fillna("").str.split(";").explode()
        out["flag_shares"] = {k: float(v / len(cal))
                              for k, v in codes[codes != ""].value_counts().items()}
    if "range_width" in cal.columns and cal["range_width"].notna().all():
        out["ensemble"] = {"members": rel_cfg.ensemble_members,
                           "width": width_summary(cal["range_width"].to_numpy()),
                           "share_point_inside_range": float(
                               ((cal["score"] >= cal["range_low"])
                                & (cal["score"] <= cal["range_high"])).mean()),
                           "note": RANGE_NOTE}
    return out


def timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


__all__ = ["CVAutoMLOptimizer", "HeldOutFold", "NestedCVResult", "NestedCVRunner",
           "Partition", "prepare_pair", "dataframe_sha256", "MODEL_PREDICTOR",
           "CALIBRATED_PREDICTOR", "ORACLE_PREDICTOR"]
