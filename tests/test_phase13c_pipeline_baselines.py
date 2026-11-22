"""Phase 13C step 3: baselines integrated into TrainingPipeline.run().

Checks that baselines are reported on exactly the model's patients, follow the
mirrored policy (fit train -> threshold validation -> refit train+val -> test
once), never touch the test set before the end, never enter the Optuna search,
and do not change the Optuna study identity.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from dialong_automl.baselines import BaselineRunner
from dialong_automl.data.splitter import PatientSplitter
from dialong_automl.training import pipeline as pipe_mod
from dialong_automl.training.pipeline import TrainingPipeline, build_search_identity
from tests.test_training_pipeline import _tiny_cfg, _tiny_csv

NAMES = [
    "b0_prevalence", "b1_last_hba1c", "b2_trend_extrapolation",
    "b3_logreg_all", "b3_logreg_hba1c_only", "b3_logreg_no_hba1c",
]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p13c")
    csv = _tiny_csv(tmp, n_patients=150)
    cfg = _tiny_cfg(tmp, n_trials=2)
    seen: dict = {}
    real_refit, real_test = BaselineRunner.refit, BaselineRunner.evaluate_test

    def spy_refit(self, train_val):
        seen["refit_ids"] = list(train_val.patient_ids)
        return real_refit(self, train_val)

    def spy_test(self, test):
        seen["test_ids"] = list(test.patient_ids)
        seen["stage_before_test"] = self.stage
        return real_test(self, test)

    BaselineRunner.refit, BaselineRunner.evaluate_test = spy_refit, spy_test
    try:
        result = TrainingPipeline(cfg, full_csv_path=csv).run()
    finally:
        BaselineRunner.refit, BaselineRunner.evaluate_test = real_refit, real_test
    meta = json.loads(Path(result.artifact_paths["model_metadata"]).read_text())
    summary = json.loads(Path(result.artifact_paths["run_summary"]).read_text())
    return result, meta, summary, seen, cfg


def test_baselines_enabled_by_default_in_config():
    from dialong_automl.config.schema import AppConfig

    assert AppConfig().baselines.enabled is True


def test_baselines_reported_in_metadata_summary_and_result(run):
    result, meta, summary, _, _ = run
    for blob in (meta["baselines"], summary["baselines"], result.baselines):
        assert blob["enabled"] is True and blob["stage"] == "tested"
        assert list(blob["results"]) == NAMES
        assert blob["policy"]["selectable_as_model"] is False
        assert blob["policy"]["optuna_candidate"] is False
    assert meta["baselines"] == summary["baselines"]


def test_baselines_scored_on_exactly_the_models_patients(run):
    result, meta, _, seen, _ = run
    model_val = meta["validation_metrics"]
    for name, r in meta["baselines"]["results"].items():
        assert r["validation_metrics"]["val_n_samples"] == model_val["val_n_samples"]
        assert r["validation_metrics"]["val_n_positive"] == model_val["val_n_positive"]
        assert r["test_metrics"]["test_n_samples"] == meta["test_metrics"]["test_n_samples"]
        assert r["test_metrics"]["test_n_positive"] == meta["test_metrics"]["test_n_positive"]
        assert r["retrain_resubstitution_metrics"]["resub_n_samples"] == result.n_val


def test_mirrored_policy_refit_on_train_plus_validation_then_test_once(run):
    result, _, _, seen, _ = run
    assert len(seen["refit_ids"]) == result.n_train + result.n_val
    assert not set(seen["refit_ids"]) & set(seen["test_ids"])     # no test patient in refit
    assert len(seen["test_ids"]) == result.n_test
    assert seen["stage_before_test"] == "refitted"


def test_selected_model_and_leaderboard_never_contain_baselines(run):
    result, meta, _, _, _ = run
    assert meta["selected_model"] not in NAMES
    board = pd.read_csv(result.artifact_paths["leaderboard"])
    assert not set(board["model_name"]) & set(NAMES)


def test_comparison_table_written(run):
    result, meta, _, _, _ = run
    table = pd.read_csv(result.artifact_paths["baselines_comparison"])
    assert list(table["predictor"]) == [meta["selected_model"], *NAMES]
    assert table.loc[0, "kind"] == "selected_model"
    assert table.loc[0, "test_auroc"] == pytest.approx(meta["test_metrics"]["test_auroc"])
    b0 = table[table.predictor == "b0_prevalence"].iloc[0]
    assert b0["test_auroc"] == 0.5 and b0["val_auroc"] == 0.5
    assert pd.isna(table[table.predictor == "b1_last_hba1c"].iloc[0]["test_brier"])


def test_baselines_do_not_change_the_optuna_study_identity(run):
    *_, cfg = run
    ref = build_search_identity(cfg, "h")
    off = cfg.model_copy(deep=True)
    off.baselines.enabled = False
    other = cfg.model_copy(deep=True)
    other.baselines.b3_c = 0.1
    assert build_search_identity(off, "h") == ref == build_search_identity(other, "h")


def test_disabling_baselines(tmp_path):
    csv = _tiny_csv(tmp_path, n_patients=120)
    cfg = _tiny_cfg(tmp_path)
    cfg.baselines.enabled = False
    result = TrainingPipeline(cfg, full_csv_path=csv).run()
    meta = json.loads(Path(result.artifact_paths["model_metadata"]).read_text())
    assert meta["baselines"] == {"enabled": False} == result.baselines
    assert "baselines_comparison" not in result.artifact_paths
    assert not (Path(cfg.paths.outputs_dir) / "baselines_comparison.csv").exists()


def test_flipping_test_labels_changes_only_baseline_test_metrics(tmp_path, monkeypatch):
    csv = _tiny_csv(tmp_path, n_patients=150)

    def run_once(sub: str, flip: bool) -> dict:
        cfg = _tiny_cfg(tmp_path / sub)
        orig = PatientSplitter.split

        def maybe_flip(self, *a, **k):
            r = orig(self, *a, **k)
            return dataclasses.replace(r, test_targets=1 - r.test_targets) if flip else r

        monkeypatch.setattr(PatientSplitter, "split", maybe_flip)
        res = TrainingPipeline(cfg, full_csv_path=csv).run()
        monkeypatch.undo()
        return res.baselines["results"]

    base, flipped = run_once("base", False), run_once("flip", True)
    for name in NAMES:
        for key in ("threshold", "threshold_selection", "params_train_fit", "params_final_fit",
                    "validation_metrics", "retrain_resubstitution_metrics"):
            assert base[name][key] == flipped[name][key], (name, key)
    assert base["b1_last_hba1c"]["test_metrics"] != flipped["b1_last_hba1c"]["test_metrics"]


def test_baselines_helper_never_called_before_model_test_evaluation(monkeypatch, tmp_path):
    """The model's test metrics are computed before the baseline block runs."""
    csv = _tiny_csv(tmp_path, n_patients=120)
    order: list[str] = []
    real_eval, real_run = pipe_mod.evaluate, TrainingPipeline._run_baselines

    def spy_eval(y, p, threshold=0.5, prefix=""):
        order.append(prefix)
        return real_eval(y, p, threshold=threshold, prefix=prefix)

    def spy_run(self, **kw):
        order.append("BASELINES")
        return real_run(self, **kw)

    monkeypatch.setattr(pipe_mod, "evaluate", spy_eval)
    monkeypatch.setattr(TrainingPipeline, "_run_baselines", spy_run)
    TrainingPipeline(_tiny_cfg(tmp_path), full_csv_path=csv).run()
    assert order.index("test_") < order.index("BASELINES")
    assert order.count("BASELINES") == 1
