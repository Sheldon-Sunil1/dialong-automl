"""Phase 13B.1 regression tests.

1. GRU search vs final-retrain consistency and the documented retrain policy.
2. Completed / pruned / FAILED Optuna trials are kept apart.
3. Configured sampler and pruner are applied (or explicitly rejected).
4. Study resume: interrupted studies, state handling, metadata errors.
5. Validation / threshold limitations are documented in the artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pytest

from dialong_automl.automl import optimizer as opt_mod
from dialong_automl.automl.optimizer import (
    AutoMLOptimizer,
    NoCompletedTrialsError,
    StudyIdentityMismatchError,
    make_pruner,
    make_sampler,
)
from dialong_automl.config.schema import AppConfig, OptunaConfig
from dialong_automl.training import pipeline as pipe_mod
from dialong_automl.training.pipeline import TrainingPipeline, build_search_identity
from tests.test_gru_final_retrain import _cfg as _gru_cfg
from tests.test_gru_final_retrain import _tiny_full_csv
from tests.test_phase13b_correctness import (
    SKIP_TORCH,
    _gru_arrays,
    _search,
    _sqlite,
    _wide,
)
from tests.test_training_pipeline import _tiny_cfg, _tiny_csv

TS = optuna.trial.TrialState
_GRU_SPLIT_CALLS: list[dict] = []   # filled by the gru_run fixture


# ===========================================================================
# 1. GRU search / final-retrain consistency
# ===========================================================================

@pytest.fixture(scope="module")
def gru_run(tmp_path_factory):
    from dialong_automl.models import gru as gru_mod

    tmp_path = tmp_path_factory.mktemp("gru13b1")
    csv = _tiny_full_csv(tmp_path)
    cfg = _gru_cfg(tmp_path, "gru")          # note: split.stratify is False here
    cfg.reliability.ensemble_members = 0     # members (Phase 14E) would add GRU builds/fits

    fits: list[dict] = []
    builds: list[tuple] = []
    real_fit, real_build = gru_mod.GRUModel.fit, pipe_mod.build_model
    real_split = pipe_mod._internal_patient_split
    split_calls: list[dict] = []

    def spy_split(patient_ids, targets, val_fraction, seed, stratify):
        split_calls.append({"stratify": stratify, "cfg_stratify": cfg.split.stratify})
        return real_split(patient_ids, targets, val_fraction, seed, stratify)

    pipe_mod._internal_patient_split = spy_split

    def spy_fit(self, X, y, **kw):
        fits.append({"hp": dict(self.hyperparams), "X_shape": X.shape, "y": np.asarray(y),
                     "mask": kw.get("mask"), "lengths": kw.get("lengths"),
                     "y_val": kw.get("y_val"), "X_val": kw.get("X_val"),
                     "mask_val": kw.get("mask_val")})
        return real_fit(self, X, y, **kw)

    def spy_build(name, hp=None):
        builds.append((name, dict(hp or {})))
        return real_build(name, hp)

    gru_mod.GRUModel.fit = spy_fit
    pipe_mod.build_model = spy_build
    try:
        result = TrainingPipeline(cfg, full_csv_path=csv).run()
    finally:
        gru_mod.GRUModel.fit = real_fit
        pipe_mod.build_model = real_build
        pipe_mod._internal_patient_split = real_split
    meta = json.loads(Path(result.artifact_paths["model_metadata"]).read_text())
    _GRU_SPLIT_CALLS[:] = split_calls
    return result, meta, fits, builds


@SKIP_TORCH
def test_gru_retrain_uses_selected_hyperparameters_unchanged(gru_run):
    result, _, fits, builds = gru_run
    assert len(builds) == 1 and builds[0][0] == "gru"
    assert builds[0][1] == result.best_hyperparams            # exactly the selected set
    search_fit, final_fit = fits[0], fits[-1]
    assert search_fit["hp"] == final_fit["hp"]                # arch, lr, epochs, patience, seed


@SKIP_TORCH
def test_gru_retrain_has_same_input_layout_and_masks_as_search(gru_run):
    _, _, fits, _ = gru_run
    search_fit, final_fit = fits[0], fits[-1]
    assert search_fit["X_shape"][1:] == final_fit["X_shape"][1:]     # (T, F) identical
    for f in (search_fit, final_fit):
        assert f["mask"] is not None and f["lengths"] is not None
        np.testing.assert_array_equal(
            f["lengths"], np.maximum(f["mask"].sum(axis=1), 1)       # lengths follow the mask
        )
        assert f["mask_val"] is not None and f["X_val"] is not None


@SKIP_TORCH
def test_gru_retrain_early_stopping_holdout_has_both_classes_even_without_stratify(gru_run):
    """cfg.split.stratify=False must not allow a single-class early-stopping holdout."""
    _, _, fits, _ = gru_run
    final_fit = fits[-1]
    assert len(np.unique(final_fit["y_val"])) == 2
    assert len(np.unique(final_fit["y"])) == 2
    # The data-independent guarantee: the internal split is label-stratified even
    # though the configuration disables stratification for the outer split.
    (call,) = _GRU_SPLIT_CALLS
    assert call["cfg_stratify"] is False and call["stratify"] is True


@SKIP_TORCH
def test_gru_retrain_policy_is_documented_in_metadata(gru_run):
    result, meta, _, _ = gru_run
    pol = meta["retrain_policy"]
    assert pol == result.retrain_policy
    assert pol["policy"] == "internal_holdout_early_stopping"
    assert pol["test_set_used"] is False
    assert pol["internal_split"]["stratified_by_label"] is True
    assert (pol["internal_split"]["n_internal_train"]
            + pol["internal_split"]["n_internal_validation"]) == result.n_train + result.n_val
    assert pol["preprocessor_fit_on"].startswith("internal-train")
    assert "unchanged" in pol["hyperparameters"]
    assert len(pol["differences_from_search"]) >= 2           # differences are stated, not hidden


@SKIP_TORCH
def test_gru_fit_rejects_single_class_validation_labels():
    from dialong_automl.models.gru import GRUModel

    X, y, mask = _gru_arrays()
    model = GRUModel({"epochs": 2, "hidden_size": 8})
    with pytest.raises(ValueError, match="both classes"):
        model.fit(X, y, mask=mask, X_val=X, y_val=np.zeros(len(y), dtype=np.int32), mask_val=mask)


@SKIP_TORCH
def test_pipeline_rejects_single_class_internal_holdout(tmp_path, monkeypatch):
    csv = _tiny_full_csv(tmp_path)
    cfg = _gru_cfg(tmp_path, "gru")

    def bad_split(patient_ids, targets, val_fraction, seed, stratify):
        neg = [p for p in patient_ids if int(targets.loc[p]) == 0][:4]
        return [p for p in patient_ids if p not in neg], neg

    monkeypatch.setattr(pipe_mod, "_internal_patient_split", bad_split)
    with pytest.raises(RuntimeError, match="single class"):
        TrainingPipeline(cfg, full_csv_path=csv).run()


def test_wide_retrain_policy_is_documented(tmp_path):
    csv = _tiny_csv(tmp_path, n_patients=120)
    result = TrainingPipeline(_tiny_cfg(tmp_path), full_csv_path=csv).run()
    pol = result.retrain_policy
    assert pol["policy"] == "refit_on_train_plus_validation"
    assert pol["early_stopping"].startswith("none")
    assert pol["differences_from_search"]


# ===========================================================================
# 2. Completed / pruned / failed trials
# ===========================================================================

def _lgbm_opt(fail_first: int = 0, fail_exc=RuntimeError("boom"), n_trials=2,
              y_val_override=None, stage="training", **kw):
    """Optimizer whose model factory raises on the first `fail_first` calls."""
    Xtr, ytr, Xv, yv = _wide()
    if y_val_override is not None:
        yv = y_val_override(yv)
    real = opt_mod.build_model
    calls = {"n": 0}

    def build(name, hp=None):
        calls["n"] += 1
        model = real(name, hp)
        if calls["n"] <= fail_first:
            if stage == "training":
                raise fail_exc
            def bad_predict(_X):
                raise fail_exc
            model.predict_proba = bad_predict
        return model

    opt = AutoMLOptimizer(n_trials=n_trials, seed=1, enabled_models=["lightgbm"], **kw)
    return opt, build, (Xtr, ytr, Xv, yv)


def _run(opt, build, data, monkeypatch):
    Xtr, ytr, Xv, yv = data
    monkeypatch.setattr(opt_mod, "build_model", build)
    return opt.optimize(X_train_wide=Xtr, y_train=ytr, X_val_wide=Xv, y_val=yv)


def test_completed_trials_are_complete(monkeypatch):
    opt, build, data = _lgbm_opt(fail_first=0, n_trials=2)
    study = _run(opt, build, data, monkeypatch)
    assert [t.state for t in study.trials] == [TS.COMPLETE, TS.COMPLETE]
    assert opt.failed_trials() == [] and len(opt.build_leaderboard()) == 2
    assert opt.trial_state_counts() == {"COMPLETE": 2}


def test_training_error_is_failed_not_pruned_and_leaderboard_stays_consistent(monkeypatch):
    opt, build, data = _lgbm_opt(fail_first=1, n_trials=2)
    study = _run(opt, build, data, monkeypatch)
    assert [t.state for t in study.trials] == [TS.FAIL, TS.COMPLETE]   # not PRUNED
    (fail,) = opt.failed_trials()
    assert fail["trial_number"] == 0 and fail["stage"] == "training"
    assert fail["exception_type"] == "RuntimeError" and fail["message"] == "boom"
    assert "boom" in fail["traceback"] and fail["model_name"] == "lightgbm"
    assert "model_name" in study.trials[0].user_attrs                  # diagnosable
    board = opt.build_leaderboard()
    assert list(board["trial_number"]) == [1]                          # failed trial absent
    assert opt.best_trial_result().trial_number == 1
    assert opt.trial_state_counts() == {"FAIL": 1, "COMPLETE": 1}


def test_programming_error_in_inference_is_failed_with_stage(monkeypatch):
    opt, build, data = _lgbm_opt(fail_first=1, fail_exc=TypeError("bad arg"), stage="inference")
    study = _run(opt, build, data, monkeypatch)
    assert study.trials[0].state == TS.FAIL
    assert opt.failed_trials()[0]["stage"] == "inference"
    assert opt.failed_trials()[0]["exception_type"] == "TypeError"


def test_all_trials_failing_raises_with_diagnostics(monkeypatch):
    opt, build, data = _lgbm_opt(fail_first=99, n_trials=2)
    with pytest.raises(NoCompletedTrialsError, match="FAILED.*boom"):
        _run(opt, build, data, monkeypatch)
    assert [t.state for t in opt._study.trials] == [TS.FAIL, TS.FAIL]
    assert opt.build_leaderboard().empty


def test_undefined_validation_auroc_is_a_failure_not_pruning(monkeypatch):
    opt, build, data = _lgbm_opt(
        n_trials=1, y_val_override=lambda y: np.zeros_like(y)
    )
    with pytest.raises(NoCompletedTrialsError, match="undefined"):
        _run(opt, build, data, monkeypatch)
    assert opt._study.trials[0].state == TS.FAIL
    assert opt.failed_trials()[0]["stage"] == "evaluation"


def test_keyboard_interrupt_is_not_swallowed_or_recorded_as_failure(monkeypatch):
    opt, build, data = _lgbm_opt(fail_first=1, fail_exc=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _run(opt, build, data, monkeypatch)
    assert opt.build_leaderboard().empty


@SKIP_TORCH
def test_genuine_pruning_stays_pruned_and_is_not_a_failure(monkeypatch):
    from tests.test_phase13b_correctness import _gru_search

    opt, study = _gru_search(monkeypatch, always_prune=True)
    assert [t.state for t in study.trials] == [TS.PRUNED]
    assert opt.failed_trials() == []
    assert opt.trial_state_counts() == {"PRUNED": 1}


def test_pipeline_reports_failed_trials(tmp_path, monkeypatch):
    csv = _tiny_csv(tmp_path, n_patients=120)
    cfg = _tiny_cfg(tmp_path, n_trials=2)
    real, calls = opt_mod.build_model, {"n": 0}

    def flaky(name, hp=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("flaky model")
        return real(name, hp)

    monkeypatch.setattr(opt_mod, "build_model", flaky)
    result = TrainingPipeline(cfg, full_csv_path=csv).run()
    assert result.n_trials_failed == 1 and result.n_trials_completed == 1
    meta = json.loads(Path(result.artifact_paths["model_metadata"]).read_text())
    assert meta["optuna_n_trials_failed"] == 1
    assert meta["optuna_trial_states"] == {"FAIL": 1, "COMPLETE": 1}
    failed = json.loads(Path(result.artifact_paths["failed_trials"]).read_text())
    assert failed[0]["message"] == "flaky model"


# ===========================================================================
# 3. Sampler / pruner configuration
# ===========================================================================

@pytest.mark.parametrize("field,bad", [
    ("sampler", "CmaEsSampler"), ("pruner", "HyperbandPruner"), ("metric", "pr_auc"),
    ("direction", "minimize"), ("n_jobs", 4),
])
def test_unsupported_optuna_config_is_rejected(field, bad):
    with pytest.raises(ValueError):
        OptunaConfig(**{field: bad})


def test_supported_optuna_config_is_accepted():
    cfg = OptunaConfig(sampler="RandomSampler", pruner="NopPruner")
    assert (cfg.sampler, cfg.pruner) == ("RandomSampler", "NopPruner")


def test_optimizer_rejects_unsupported_values():
    for kw in ({"sampler": "Nope"}, {"pruner": "Nope"}, {"direction": "minimize"}):
        with pytest.raises(ValueError):
            AutoMLOptimizer(**kw)
    with pytest.raises(ValueError):
        make_sampler("Nope", 1)
    with pytest.raises(ValueError):
        make_pruner("Nope")


@pytest.mark.parametrize("sampler,pruner,s_cls,p_cls", [
    ("TPESampler", "MedianPruner", optuna.samplers.TPESampler, optuna.pruners.MedianPruner),
    ("RandomSampler", "NopPruner", optuna.samplers.RandomSampler, optuna.pruners.NopPruner),
])
def test_configured_sampler_and_pruner_are_applied_to_the_study(sampler, pruner, s_cls, p_cls):
    Xtr, ytr, Xv, yv = _wide()
    opt = AutoMLOptimizer(n_trials=1, seed=5, enabled_models=["lightgbm"],
                          sampler=sampler, pruner=pruner)
    study = opt.optimize(X_train_wide=Xtr, y_train=ytr, X_val_wide=Xv, y_val=yv)
    assert type(study.sampler) is s_cls and type(study.pruner) is p_cls


def test_pruner_choice_changes_pruning_behaviour():
    """MedianPruner prunes a clearly-worse trial; NopPruner never does."""
    def n_pruned(name):
        study = optuna.create_study(direction="maximize", pruner=make_pruner(name),
                                    sampler=make_sampler("RandomSampler", 0))

        def obj(trial):
            good = trial.number < 6
            trial.report(1.0 if good else 0.0, 0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return 1.0 if good else 0.0

        study.optimize(obj, n_trials=7)
        return sum(t.state == TS.PRUNED for t in study.trials)

    assert n_pruned("MedianPruner") == 1
    assert n_pruned("NopPruner") == 0


def test_sampler_is_seeded_and_reproducible():
    def draws(seed):
        s = optuna.create_study(sampler=make_sampler("RandomSampler", seed))
        s.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=4)
        return [t.params["x"] for t in s.trials]

    assert draws(3) == draws(3) and draws(3) != draws(4)


def test_pipeline_applies_configured_sampler_and_pruner(tmp_path, monkeypatch):
    csv = _tiny_csv(tmp_path, n_patients=120)
    cfg = _tiny_cfg(tmp_path)
    cfg.optuna.sampler, cfg.optuna.pruner = "RandomSampler", "NopPruner"
    seen: dict = {}
    real_init = AutoMLOptimizer.__init__

    def spy_init(self, *a, **k):
        seen.update(sampler=k.get("sampler"), pruner=k.get("pruner"))
        real_init(self, *a, **k)

    monkeypatch.setattr(AutoMLOptimizer, "__init__", spy_init)
    result = TrainingPipeline(cfg, full_csv_path=csv).run()
    assert seen == {"sampler": "RandomSampler", "pruner": "NopPruner"}
    meta = json.loads(Path(result.artifact_paths["model_metadata"]).read_text())
    assert (meta["optuna_sampler"], meta["optuna_pruner"]) == ("RandomSampler", "NopPruner")


def test_study_identity_reflects_effective_search_configuration():
    base = AppConfig()
    a = build_search_identity(base, "h")
    b = base.model_copy(deep=True)
    b.optuna.sampler = "RandomSampler"
    c = base.model_copy(deep=True)
    c.optuna.pruner = "NopPruner"
    assert a != build_search_identity(b, "h") and a != build_search_identity(c, "h")
    assert a["optuna"]["sampler"] == "TPESampler" and a["optuna"]["pruner"] == "MedianPruner"


# ===========================================================================
# 4. Resume behaviour
# ===========================================================================

def _snapshot(study: optuna.Study, numbers: list[int]) -> list[tuple]:
    out = []
    for t in study.trials:
        if t.number in numbers:
            out.append((t.number, t.state, t.value, dict(t.params), dict(t.user_attrs)))
    return out


def _load(storage, name):
    return optuna.load_study(study_name=name, storage=storage)


def test_resume_restores_completed_trials_faithfully(tmp_path):
    storage = _sqlite(tmp_path)
    first = _search(storage, {"k": 1}, n_trials=2)
    orig = {r.trial_number: r for r in first._trial_results}

    resumed = _search(storage, {"k": 1}, n_trials=2, resume=True)
    assert len(resumed._trial_results) == 2
    for r in resumed._trial_results:
        o = orig[r.trial_number]
        assert (r.model_name, r.representation, r.state) == (
            o.model_name, o.representation, "complete",
        )
        assert r.val_auroc == o.val_auroc and r.hyperparams == o.hyperparams
        assert r.val_proba == o.val_proba and r.val_f1 == o.val_f1


def test_resume_with_pruned_failed_and_interrupted_trials(tmp_path):
    storage = _sqlite(tmp_path)
    first = _search(storage, {"k": 2}, n_trials=1)
    study = _load(storage, first.study_name)
    pruned = study.ask()
    study.tell(pruned, state=TS.PRUNED)
    failed = study.ask()
    study.tell(failed, state=TS.FAIL)
    zombie = study.ask()                                   # never told: process "died"
    before = _snapshot(_load(storage, first.study_name), [0, pruned.number, failed.number])

    # finished = 1 complete + 1 pruned + 1 failed = 3; target 4 -> exactly one new trial.
    resumed = _search(storage, {"k": 2}, n_trials=4, resume=True)
    s = _load(storage, first.study_name)
    states = {t.number: t.state for t in s.trials}
    assert states[0] == TS.COMPLETE and states[pruned.number] == TS.PRUNED
    assert states[failed.number] == TS.FAIL
    assert states[zombie.number] == TS.RUNNING             # left untouched
    assert len(s.trials) == 5 and sum(v == TS.COMPLETE for v in states.values()) == 2
    assert resumed.n_interrupted_trials_ignored == 1
    # Pruned / failed / interrupted trials never reach the leaderboard.
    assert sorted(resumed.build_leaderboard()["trial_number"]) == sorted(
        n for n, st in states.items() if st == TS.COMPLETE
    )
    # Historical trials are byte-for-byte unchanged.
    assert _snapshot(s, [0, pruned.number, failed.number]) == before


def test_repeated_resume_never_duplicates_trials(tmp_path):
    storage = _sqlite(tmp_path)
    first = _search(storage, {"k": 3}, n_trials=1)
    for _ in range(3):
        again = _search(storage, {"k": 3}, n_trials=2, resume=True)
    s = _load(storage, first.study_name)
    assert len(s.trials) == 2                                # target is the TOTAL
    nums = list(again.build_leaderboard()["trial_number"])
    assert sorted(nums) == [0, 1] and len(set(nums)) == 2


def test_resume_target_counts_all_finished_trials(tmp_path):
    storage = _sqlite(tmp_path)
    first = _search(storage, {"k": 4}, n_trials=1)
    s = _load(storage, first.study_name)
    t = s.ask()
    s.tell(t, state=TS.FAIL)                                 # 2 finished already
    again = _search(storage, {"k": 4}, n_trials=2, resume=True)
    assert len(_load(storage, first.study_name).trials) == 2  # nothing new was run
    assert len(again.build_leaderboard()) == 1


def test_resume_refuses_completed_trial_without_stored_metadata(tmp_path):
    storage = _sqlite(tmp_path)
    first = _search(storage, {"k": 5}, n_trials=1)
    s = _load(storage, first.study_name)
    t = s.ask()
    s.tell(t, 0.9)                                           # COMPLETE, no user attrs
    with pytest.raises(StudyIdentityMismatchError, match="lacks the stored metadata"):
        _search(storage, {"k": 5}, n_trials=3, resume=True)


def test_resume_refuses_completed_trial_missing_validation_predictions(tmp_path):
    storage = _sqlite(tmp_path)
    first = _search(storage, {"k": 6}, n_trials=1)
    s = _load(storage, first.study_name)
    s.add_trial(optuna.trial.create_trial(
        value=0.9, params={}, distributions={},
        user_attrs={"model_name": "lightgbm", "representation": "wide",
                    "hyperparams_json": "{}"},               # no val_proba
    ))
    with pytest.raises(StudyIdentityMismatchError, match="val_proba"):
        _search(storage, {"k": 6}, n_trials=3, resume=True)


def test_resume_refuses_unreadable_hyperparameter_metadata(tmp_path):
    storage = _sqlite(tmp_path)
    first = _search(storage, {"k": 7}, n_trials=1)
    s = _load(storage, first.study_name)
    s.add_trial(optuna.trial.create_trial(
        value=0.9, params={}, distributions={},
        user_attrs={"model_name": "lightgbm", "representation": "wide",
                    "hyperparams_json": "{not json", "val_proba": [0.1]},
    ))
    with pytest.raises(StudyIdentityMismatchError, match="unreadable"):
        _search(storage, {"k": 7}, n_trials=3, resume=True)


def test_resume_does_not_touch_other_studies_in_the_same_database(tmp_path):
    storage = _sqlite(tmp_path)
    other = _search(storage, {"k": "other"}, n_trials=1)
    snap = _snapshot(_load(storage, other.study_name), [0])
    _search(storage, {"k": 8}, n_trials=1)
    _search(storage, {"k": 8}, n_trials=2, resume=True)
    assert _snapshot(_load(storage, other.study_name), [0]) == snap


# ===========================================================================
# 5. Validation / threshold limitations are documented in the artefacts
# ===========================================================================

def test_validation_limitations_are_stated_in_artifacts(tmp_path):
    csv = _tiny_csv(tmp_path, n_patients=120)
    cfg = _tiny_cfg(tmp_path)
    cfg.training.threshold_strategy = "validation_f1"
    result = TrainingPipeline(cfg, full_csv_path=csv).run()
    meta = json.loads(Path(result.artifact_paths["model_metadata"]).read_text())
    summary = json.loads(Path(result.artifact_paths["run_summary"]).read_text())

    src = meta["validation_metrics_source"]
    assert "NOT an unbiased estimate" in src
    for needle in ("model and hyperparameter selection", "early stopping", "threshold selection"):
        assert needle in src
    assert "held-out" not in src.lower()                      # not called held-out performance
    assert any("not unbiased" in x for x in meta["limitations"])
    assert any("not unbiased" in x for x in summary["limitations"])

    sel = meta["threshold_selection"]
    assert "pre-retrain" in sel["applied_to"] and "retrained model" in sel["applied_to"]
    assert sel["strategy"] == "validation_f1" and sel["test_set_used"] is False
