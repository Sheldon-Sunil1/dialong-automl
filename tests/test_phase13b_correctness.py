"""Phase 13B regression tests: experimental correctness fixes.

Covers
------
1. Validation-metric labelling after the final retrain.
2. Optuna study isolation and intentional resume.
3. GRU pruning raising ``optuna.TrialPruned`` (and normal early stopping).
4. Threshold strategy handling without test-set leakage.

All datasets are tiny; no expensive Optuna search is run.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import optuna
import pytest

from dialong_automl.automl.optimizer import (
    AutoMLOptimizer,
    StudyExistsError,
    StudyIdentityMismatchError,
    build_study_name,
)
from dialong_automl.config.schema import AppConfig
from dialong_automl.data.splitter import PatientSplitter
from dialong_automl.evaluation.metrics import evaluate, select_threshold_by_f1
from dialong_automl.evaluation.threshold import select_decision_threshold
from dialong_automl.training import pipeline as pipe_mod
from dialong_automl.training.pipeline import TrainingPipeline, build_search_identity
from tests.test_training_pipeline import _tiny_cfg, _tiny_csv


def _torch_ok() -> bool:
    try:
        import torch
        torch.tensor([1.0])
        return True
    except (ImportError, OSError):
        return False


SKIP_TORCH = pytest.mark.skipif(not _torch_ok(), reason="torch unavailable")


# ===========================================================================
# 1. Validation metric labelling
# ===========================================================================

@pytest.fixture(scope="module")
def labelled_run(tmp_path_factory):
    """One tiny pipeline run with spies recording evaluate()/build_model() order."""
    tmp_path = tmp_path_factory.mktemp("labelled")
    csv = _tiny_csv(tmp_path, n_patients=120)
    cfg = _tiny_cfg(tmp_path)

    events: list[tuple] = []
    orig_eval, orig_build = pipe_mod.evaluate, pipe_mod.build_model

    def spy_eval(y_true, y_proba, threshold=0.5, prefix=""):
        events.append(("evaluate", prefix, len(y_true)))
        return orig_eval(y_true, y_proba, threshold=threshold, prefix=prefix)

    def spy_build(*a, **k):
        events.append(("build_model",))
        return orig_build(*a, **k)

    pipe_mod.evaluate, pipe_mod.build_model = spy_eval, spy_build
    try:
        result = TrainingPipeline(cfg, full_csv_path=csv).run()
    finally:
        pipe_mod.evaluate, pipe_mod.build_model = orig_eval, orig_build

    meta = json.loads(Path(result.artifact_paths["model_metadata"]).read_text())
    summary = json.loads(Path(result.artifact_paths["run_summary"]).read_text())
    return result, meta, summary, events


def test_validation_metrics_are_the_pre_retrain_search_metrics(labelled_run):
    result, meta, summary, _ = labelled_run
    # The search score IS the validation AUROC of the train-only model, so the
    # reported validation AUROC must equal it exactly.
    best = result.best_val_auroc
    assert result.val_metrics["val_auroc"] == pytest.approx(best, abs=1e-12)
    assert meta["validation_metrics"]["val_auroc"] == pytest.approx(best, abs=1e-12)
    assert summary["validation_metrics"]["val_auroc"] == pytest.approx(best, abs=1e-12)
    assert "train-only" in meta["validation_metrics_source"].lower() or "TRAIN" in meta["validation_metrics_source"]


def test_validation_metrics_computed_before_final_retrain(labelled_run):
    """The 'val_' evaluation must happen before the final model is built/fitted."""
    _, _, _, events = labelled_run
    kinds = [e for e in events if e[0] == "build_model" or e[1] in ("val_", "resub_", "test_")]
    names = [e[0] if e[0] == "build_model" else e[1] for e in kinds]
    assert names.index("val_") < names.index("build_model")
    # Resubstitution is computed after the retrain, never labelled val_.
    assert names.index("resub_") > names.index("build_model")
    assert names.count("val_") == 1 and names.count("resub_") == 1 and names.count("test_") == 1


def test_metric_names_match_evaluation_data(labelled_run):
    result, meta, summary, _ = labelled_run
    val_pos = int(result.val_metrics["val_n_positive"])
    for blob in (meta, summary):
        assert all(k.startswith("val_") for k in blob["validation_metrics"])
        assert all(k.startswith("resub_") for k in blob["retrain_resubstitution_metrics"])
        assert all(k.startswith("test_") for k in blob["test_metrics"])
        assert not any(k.startswith("val_") for k in blob["retrain_resubstitution_metrics"])
        # Same validation patients, but a different model => different keys.
        assert blob["validation_metrics"]["val_n_samples"] == result.n_val
        assert blob["retrain_resubstitution_metrics"]["resub_n_samples"] == result.n_val
        assert blob["retrain_resubstitution_metrics"]["resub_n_positive"] == val_pos
        assert blob["test_metrics"]["test_n_samples"] == result.n_test
    assert "NOT held-out" in meta["retrain_resubstitution_metrics_note"]


def test_pipeline_result_exposes_labelled_metrics(labelled_run):
    result, *_ = labelled_run
    assert all(k.startswith("resub_") for k in result.resubstitution_metrics)
    assert all(k.startswith("val_") for k in result.val_metrics)
    assert all(k.startswith("test_") for k in result.test_metrics)


# ===========================================================================
# 2. Optuna study isolation and resume
# ===========================================================================

def _wide(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 8)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(np.int32)
    k = int(n * 0.7)
    return X[:k], y[:k], X[k:], y[k:]


def _search(storage, identity, n_trials=1, resume=False, seed=42):
    Xtr, ytr, Xv, yv = _wide()
    opt = AutoMLOptimizer(
        n_trials=n_trials, seed=seed, storage=storage, study_name="iso",
        identity=identity, resume=resume, enabled_models=["lightgbm"],
    )
    opt.optimize(X_train_wide=Xtr, y_train=ytr, X_val_wide=Xv, y_val=yv)
    return opt


def _sqlite(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}"


def test_study_name_depends_on_identity_and_is_deterministic():
    a = {"seed": 1, "dataset": "x"}
    assert build_study_name("s", a) == build_study_name("s", dict(a))
    assert build_study_name("s", a) != build_study_name("s", {**a, "seed": 2})
    assert build_study_name("s", a) != build_study_name("s", {**a, "dataset": "y"})
    assert build_study_name("s", a) != build_study_name("t", a)


def test_persistent_storage_requires_identity(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        AutoMLOptimizer(storage=_sqlite(tmp_path))
    with pytest.raises(ValueError, match="resume"):
        AutoMLOptimizer(resume=True)


def test_different_identities_get_isolated_studies(tmp_path):
    storage = _sqlite(tmp_path)
    o1 = _search(storage, {"seed": 1})
    o2 = _search(storage, {"seed": 2})
    assert o1.study_name != o2.study_name
    names = set(optuna.study.get_all_study_names(storage))
    assert {o1.study_name, o2.study_name} <= names
    # Each study holds only its own single trial; leaderboards do not mix.
    for o in (o1, o2):
        assert len(optuna.load_study(study_name=o.study_name, storage=storage).trials) == 1
        assert len(o.build_leaderboard()) == 1


def test_same_identity_is_not_silently_reused(tmp_path):
    storage = _sqlite(tmp_path)
    first = _search(storage, {"seed": 1})
    with pytest.raises(StudyExistsError):
        _search(storage, {"seed": 1})
    # The refused run must not have touched the stored study.
    assert len(optuna.load_study(study_name=first.study_name, storage=storage).trials) == 1


def test_intentional_resume_continues_the_same_study(tmp_path):
    storage = _sqlite(tmp_path)
    ident = {"seed": 1}
    first = _search(storage, ident, n_trials=1)
    hp0 = first.build_leaderboard().iloc[0]["hyperparameters"]

    resumed = _search(storage, ident, n_trials=2, resume=True)  # n_trials = TOTAL
    study = optuna.load_study(study_name=resumed.study_name, storage=storage)
    assert len(study.trials) == 2
    board = resumed.build_leaderboard()
    assert len(board) == 2                              # leaderboard == whole study
    assert sorted(board["trial_number"]) == [0, 1]
    assert hp0 in set(board["hyperparameters"])         # re-hydrated trial 0
    assert resumed.best_trial_result().val_proba is not None

    again = _search(storage, ident, n_trials=2, resume=True)   # nothing left to run
    assert len(optuna.load_study(study_name=again.study_name, storage=storage).trials) == 2
    assert len(again.build_leaderboard()) == 2


def test_resume_refuses_tampered_identity(tmp_path):
    storage = _sqlite(tmp_path)
    first = _search(storage, {"seed": 1})
    optuna.load_study(study_name=first.study_name, storage=storage).set_user_attr(
        "identity_json", '{"seed": 999}'
    )
    with pytest.raises(StudyIdentityMismatchError):
        _search(storage, {"seed": 1}, n_trials=2, resume=True)


def test_search_identity_covers_dataset_config_and_seeds_but_not_trial_budget():
    base = AppConfig()
    ident = lambda c, h="h1": build_search_identity(c, h)  # noqa: E731
    ref = build_study_name("s", ident(base))

    def changed(mutate, h="h1"):
        c = base.model_copy(deep=True)
        mutate(c)
        return build_study_name("s", ident(c, h))

    assert changed(lambda c: None, h="h2") != ref                       # dataset content
    assert changed(lambda c: setattr(c.optuna, "seed", 7)) != ref       # sampler seed
    assert changed(lambda c: setattr(c.training, "seed", 7)) != ref     # training seed
    assert changed(lambda c: setattr(c.split, "random_seed", 7)) != ref # split seed
    assert changed(lambda c: setattr(c.cohort, "prediction_horizon_days", 90)) != ref
    assert changed(lambda c: setattr(c.model, "enabled_models", ["lightgbm"])) != ref
    assert changed(lambda c: setattr(c.gru, "epochs", 3)) != ref
    assert changed(lambda c: setattr(c.optuna, "run_id", "fold-3")) != ref  # fold/run id
    # Budget / location / post-search settings must NOT change the identity.
    assert changed(lambda c: setattr(c.optuna, "n_trials", 99)) == ref
    assert changed(lambda c: setattr(c.optuna, "storage", "sqlite:///x.db")) == ref
    assert changed(lambda c: setattr(c.training, "threshold_strategy", "validation_f1")) == ref


def test_pipeline_isolates_and_resumes_with_persistent_storage(tmp_path):
    csv = _tiny_csv(tmp_path, n_patients=120)
    db = tmp_path / "shared.db"

    def cfg_for(sub: str, **kw) -> AppConfig:
        cfg = _tiny_cfg(tmp_path / sub)
        cfg.optuna.storage = f"sqlite:///{db.as_posix()}"
        for k, v in kw.items():
            setattr(cfg.optuna, k, v)
        return cfg

    r1 = TrainingPipeline(cfg_for("a"), full_csv_path=csv).run()
    meta1 = json.loads(Path(r1.artifact_paths["model_metadata"]).read_text())
    assert meta1["optuna_study_name"] == r1.study_name

    # Same experiment again without resume -> refused (no silent trial reuse).
    with pytest.raises(StudyExistsError):
        TrainingPipeline(cfg_for("b"), full_csv_path=csv).run()

    # A different seed is a different experiment and gets its own study.
    r2 = TrainingPipeline(cfg_for("c", seed=7), full_csv_path=csv).run()
    assert r2.study_name != r1.study_name

    # Intentional resume of the first experiment reproduces its selection.
    r3 = TrainingPipeline(cfg_for("d", resume=True), full_csv_path=csv).run()
    assert r3.study_name == r1.study_name
    assert r3.best_val_auroc == pytest.approx(r1.best_val_auroc)
    names = optuna.study.get_all_study_names(f"sqlite:///{db.as_posix()}")
    assert r1.study_name in names and r2.study_name in names


# ===========================================================================
# 3. GRU pruning
# ===========================================================================

def _gru_arrays(n=40, T=4, F=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, F)).astype(np.float32)
    y = (X[:, -1, 0] + 0.3 * rng.standard_normal(n) > 0).astype(np.int32)
    mask = np.ones((n, T), dtype=np.uint8)
    return X, y, mask


class _FakeTrial:
    def __init__(self, prune_at: int | None):
        self.prune_at = prune_at
        self.reports: list[tuple[float, int]] = []

    def report(self, value, step):
        self.reports.append((float(value), int(step)))

    def should_prune(self):
        return self.prune_at is not None and len(self.reports) - 1 >= self.prune_at


@SKIP_TORCH
def test_gru_raises_trial_pruned_when_pruner_requests():
    from dialong_automl.models.gru import GRUModel

    X, y, mask = _gru_arrays()
    model = GRUModel({"epochs": 10, "patience": 10, "hidden_size": 8, "batch_size": 16})
    trial = _FakeTrial(prune_at=1)
    with pytest.raises(optuna.exceptions.TrialPruned):
        model.fit(X, y, mask=mask, X_val=X, y_val=y, mask_val=mask, trial=trial)
    assert [s for _, s in trial.reports] == [0, 1]       # reported each epoch, stopped at 1
    assert not getattr(model, "_fitted", False)          # no usable partial model


@SKIP_TORCH
def test_gru_completes_when_not_pruned_and_reports_each_epoch():
    from dialong_automl.models.gru import GRUModel

    X, y, mask = _gru_arrays()
    model = GRUModel({"epochs": 3, "patience": 10, "hidden_size": 8, "batch_size": 16})
    trial = _FakeTrial(prune_at=None)
    model.fit(X, y, mask=mask, X_val=X, y_val=y, mask_val=mask, trial=trial)
    assert [s for _, s in trial.reports] == [0, 1, 2]
    assert model.predict_proba_with_mask(X, mask).shape == (len(y),)


@SKIP_TORCH
def test_gru_normal_early_stopping_still_works():
    from dialong_automl.models.gru import GRUModel

    X, y, mask = _gru_arrays()
    rng = np.random.default_rng(1)
    y_noise = rng.integers(0, 2, len(y)).astype(np.int32)   # val labels unrelated to X
    model = GRUModel({"epochs": 40, "patience": 1, "hidden_size": 8, "batch_size": 16})
    trial = _FakeTrial(prune_at=None)
    model.fit(X, y, mask=mask, X_val=X, y_val=y_noise, mask_val=mask, trial=trial)
    assert 2 <= len(trial.reports) < 40                      # stopped by patience, not epochs
    # And without a trial (final retrain path) nothing is raised.
    GRUModel({"epochs": 2, "hidden_size": 8}).fit(
        X, y, mask=mask, X_val=X, y_val=y, mask_val=mask
    )


def _gru_search(monkeypatch=None, always_prune=False):
    if always_prune:
        class AlwaysPrune(optuna.pruners.BasePruner):
            def prune(self, study, trial):
                return True

        monkeypatch.setattr(optuna.pruners, "MedianPruner", AlwaysPrune)
    Xs, ys, mask = _gru_arrays(n=48)
    Xw, yw, _, _ = _wide(n=48)
    ntr = 32
    lengths = mask.sum(axis=1).astype(np.int32)
    opt = AutoMLOptimizer(
        n_trials=1, seed=3, enabled_models=["gru"], gru_cfg={"epochs": 3, "patience": 3},
    )
    study = opt.optimize(
        X_train_wide=Xw[:ntr], y_train=ys[:ntr], X_val_wide=Xw[ntr:], y_val=ys[ntr:],
        X_train_seq=Xs[:ntr], mask_train=mask[:ntr], lengths_train=lengths[:ntr],
        X_val_seq=Xs[ntr:], mask_val=mask[ntr:], lengths_val=lengths[ntr:],
    )
    return opt, study


@SKIP_TORCH
def test_pruned_gru_trial_is_pruned_not_completed(monkeypatch):
    opt, study = _gru_search(monkeypatch, always_prune=True)
    assert [t.state for t in study.trials] == [optuna.trial.TrialState.PRUNED]
    assert opt._trial_results == []            # never recorded as a completed result
    assert opt.best_trial_result() is None
    assert opt.build_leaderboard().empty


@SKIP_TORCH
def test_unpruned_gru_trial_completes_normally():
    opt, study = _gru_search()
    assert [t.state for t in study.trials] == [optuna.trial.TrialState.COMPLETE]
    assert len(opt._trial_results) == 1
    assert opt.best_trial_result().val_auroc == pytest.approx(study.trials[0].value)


# ===========================================================================
# 4. Threshold strategy
# ===========================================================================

def test_fixed_strategy_uses_configured_value_and_no_data():
    info = select_decision_threshold("fixed", 0.37)
    assert info["threshold"] == 0.37 and info["strategy"] == "fixed"
    assert info["test_set_used"] is False and info["n_samples"] is None
    assert "no data" in info["selected_on"]


def test_validation_f1_matches_f1_maximiser_on_validation_data():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([.05, .1, .2, .3, .6, .7, .9, .4, .55, .15])
    info = select_decision_threshold("validation_f1", 0.5, y, p)
    assert info["threshold"] == select_threshold_by_f1(y, p)
    assert evaluate(y, p, threshold=info["threshold"])["f1"] == 1.0
    assert info["n_samples"] == 10 and info["n_positive"] == 4
    assert "validation" in info["selected_on"] and info["test_set_used"] is False


def test_validation_f1_falls_back_explicitly_when_f1_undefined():
    y = np.zeros(8, dtype=int)                       # no positives -> F1 undefined
    info = select_decision_threshold("validation_f1", 0.42, y, np.linspace(0.1, 0.9, 8))
    assert info["threshold"] == 0.42 and info["fallback_reason"]


def test_unsupported_or_incomplete_threshold_requests_are_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        select_decision_threshold("test_f1", 0.5)
    with pytest.raises(ValueError):
        select_decision_threshold("validation_f1", 0.5)          # no validation data
    with pytest.raises(ValueError):
        select_decision_threshold("validation_f1", 0.5, np.array([0, 1]), np.array([0.2]))
    with pytest.raises(ValueError):
        select_decision_threshold("fixed", 1.5)


def test_threshold_selector_cannot_receive_test_data():
    params = set(inspect.signature(select_decision_threshold).parameters)
    assert params == {"strategy", "fixed_threshold", "y_validation", "proba_validation"}


def test_pipeline_rejects_unsupported_strategy(tmp_path):
    csv = _tiny_csv(tmp_path, n_patients=120)
    cfg = _tiny_cfg(tmp_path)
    with pytest.raises(ValueError):
        cfg.training.threshold_strategy = "bogus"   # rejected by schema or by the pipeline
        TrainingPipeline(cfg, full_csv_path=csv).run()


def test_validation_f1_pipeline_ignores_test_labels(tmp_path, monkeypatch):
    csv = _tiny_csv(tmp_path, n_patients=150)

    def run(sub: str, flip_test: bool):
        cfg = _tiny_cfg(tmp_path / sub)
        cfg.training.threshold_strategy = "validation_f1"
        calls: list[dict] = []
        orig_sel = pipe_mod.select_decision_threshold

        def spy_sel(*a, **k):
            calls.append({"args": a, "kwargs": k})
            return orig_sel(*a, **k)

        monkeypatch.setattr(pipe_mod, "select_decision_threshold", spy_sel)
        orig_split = PatientSplitter.split

        def maybe_flipped(self, *a, **k):
            r = orig_split(self, *a, **k)
            if flip_test:
                r = dataclasses.replace(r, test_targets=1 - r.test_targets)
            return r

        monkeypatch.setattr(PatientSplitter, "split", maybe_flipped)
        result = TrainingPipeline(cfg, full_csv_path=csv).run()
        monkeypatch.undo()
        return result, calls

    base, base_calls = run("base", flip_test=False)
    flipped, flip_calls = run("flip", flip_test=True)

    # The selector saw exactly the validation partition, nothing else.
    for res, calls in ((base, base_calls), (flipped, flip_calls)):
        assert len(calls) == 1
        kw = calls[0]["kwargs"]
        assert len(kw["y_validation"]) == len(kw["proba_validation"]) == res.n_val

    # Flipping every TEST label changed the test results ...
    assert base.test_metrics["test_auroc"] != flipped.test_metrics["test_auroc"]
    # ... but not the threshold, its provenance, or the validation metrics.
    assert base.threshold == flipped.threshold
    assert base.threshold_selection == flipped.threshold_selection
    assert base.val_metrics == flipped.val_metrics

    # The test evaluation used the already-selected threshold unchanged.
    for res in (base, flipped):
        assert res.test_metrics["test_threshold"] == res.threshold
        assert res.val_metrics["val_threshold"] == res.threshold
        assert res.threshold_selection["strategy"] == "validation_f1"
        assert res.threshold_selection["test_set_used"] is False
        meta = json.loads(Path(res.artifact_paths["model_metadata"]).read_text())
        assert meta["threshold"] == res.threshold
        assert meta["threshold_selection"]["selected_on"] == res.threshold_selection["selected_on"]
