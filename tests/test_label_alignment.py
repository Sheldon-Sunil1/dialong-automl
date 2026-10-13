"""Regression tests — label/feature-row alignment in the training pipeline.

Confirmed bug (now fixed): the data splitter preserves StratifiedShuffleSplit
patient-ID order in ``split.*_targets``, while the representation builders emit
feature rows in sorted-patient order. The pipeline previously paired
``split.*_targets.values`` with the builders' feature matrices, assigning
labels to the wrong patients.

These tests assert PATIENT-LEVEL correspondence (not just equal lengths):
each label must equal the true target of the patient represented by the
feature row at the same index. They fail against the pre-fix code (which used
``split.*_targets.values``) and pass against the fix (which uses the builder
``.y`` and asserts wide/sequence patient-order agreement).
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from dialong_automl.config.schema import AppConfig
from dialong_automl.data.splitter import PatientSplitter, SplitConfig
from dialong_automl.features.sequence import SequenceRepresentationBuilder
from dialong_automl.features.wide import WideRepresentationBuilder
from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator
from dialong_automl.training.pipeline import (
    _aligned_labels,
    _assert_same_patient_order,
)


def _lgbm_ok() -> bool:
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


SKIP_LGBM = pytest.mark.skipif(not _lgbm_ok(), reason="lightgbm unavailable")


def _torch_ok() -> bool:
    try:
        import torch
        torch.tensor([1.0])
        return True
    except (ImportError, OSError):
        return False


SKIP_TORCH = pytest.mark.skipif(not _torch_ok(), reason="torch unavailable")


# ---------------------------------------------------------------------------
# Deterministic cohort where TRUE LABEL is a function of patient_id.
# Even patient index -> 1, odd -> 0. Lets us detect per-patient mismatches.
# ---------------------------------------------------------------------------

def _labelled_cohort(n: int = 20):
    rows = []
    pids = [f"SYN_{i:03d}" for i in range(n)]
    tmap: dict[str, int] = {}
    for i, pid in enumerate(pids):
        tmap[pid] = int(i % 2 == 0)
        for v in range(3):
            rows.append({
                "patient_id": pid,
                "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(days=90 * v),
                "hba1c": 5.0 + 0.1 * i + 0.01 * v,
                "sex": "F" if i % 2 == 0 else "M",
            })
    obs = pd.DataFrame(rows)
    targets = pd.Series(tmap, name="target")
    targets.index.name = "patient_id"
    return obs, targets, tmap


def _shuffled_split(obs, targets):
    """A real stratified split whose ID order differs from sorted order."""
    split = PatientSplitter(
        SplitConfig(train_size=0.6, validation_size=0.2, test_size=0.2,
                    seed=42, stratify=True)
    ).split(obs, targets)
    # Precondition for the regression: the splitter's order must differ from
    # sorted order, otherwise the test would not exercise the bug.
    assert split.train_ids != sorted(split.train_ids), (
        "test precondition: splitter must produce a non-sorted order"
    )
    return split


# ---------------------------------------------------------------------------
# 1. Wide builder: .y aligns by patient; split targets would not.
# ---------------------------------------------------------------------------

def test_wide_builder_labels_align_by_patient():
    obs, targets, tmap = _labelled_cohort()
    split = _shuffled_split(obs, targets)

    wide = WideRepresentationBuilder(max_waves=4).build(
        split.train_df, split.train_targets
    )
    y_aligned = _aligned_labels(wide, "train")

    # Builder .y matches true label for each feature row's patient.
    for row_idx, pid in enumerate(wide.patient_ids):
        assert y_aligned[row_idx] == tmap[pid], (
            f"row {row_idx} ({pid}): aligned label {y_aligned[row_idx]} "
            f"!= true {tmap[pid]}"
        )

    # And the OLD (buggy) approach would mismatch at least one patient.
    old = split.train_targets.values.astype(int)
    mismatches = sum(
        int(old[i]) != tmap[pid] for i, pid in enumerate(wide.patient_ids)
    )
    assert mismatches > 0, (
        "expected the pre-fix split.train_targets.values pairing to mislabel; "
        "if this is 0 the test no longer guards the bug"
    )


# ---------------------------------------------------------------------------
# 2. Sequence builder: .y aligns by patient; split targets would not.
# ---------------------------------------------------------------------------

def test_sequence_builder_labels_align_by_patient():
    obs, targets, tmap = _labelled_cohort()
    split = _shuffled_split(obs, targets)

    seq = SequenceRepresentationBuilder(max_seq_len=4).build(
        split.train_df, split.train_targets
    )
    y_aligned = _aligned_labels(seq, "train")

    for row_idx, pid in enumerate(seq.patient_ids):
        assert y_aligned[row_idx] == tmap[pid]

    old = split.train_targets.values.astype(int)
    mismatches = sum(
        int(old[i]) != tmap[pid] for i, pid in enumerate(seq.patient_ids)
    )
    assert mismatches > 0


# ---------------------------------------------------------------------------
# 3. Wide and sequence builders agree on patient order (shared label vector).
# ---------------------------------------------------------------------------

def test_wide_and_sequence_patient_order_match():
    obs, targets, _ = _labelled_cohort()
    split = _shuffled_split(obs, targets)
    wide = WideRepresentationBuilder(max_waves=4).build(split.train_df, split.train_targets)
    seq = SequenceRepresentationBuilder(max_seq_len=4).build(split.train_df, split.train_targets)
    assert wide.patient_ids == seq.patient_ids
    # The guard must not raise for matching orders.
    _assert_same_patient_order(wide, seq, "train")


def test_assert_same_patient_order_raises_on_mismatch():
    obs, targets, _ = _labelled_cohort()
    split = _shuffled_split(obs, targets)
    wide = WideRepresentationBuilder(max_waves=4).build(split.train_df, split.train_targets)
    seq = SequenceRepresentationBuilder(max_seq_len=4).build(split.train_df, split.train_targets)

    class _FakeRep:
        patient_ids = list(reversed(seq.patient_ids))

    with pytest.raises(RuntimeError, match="different patient orders"):
        _assert_same_patient_order(wide, _FakeRep(), "train")


# ---------------------------------------------------------------------------
# 4. _aligned_labels error handling.
# ---------------------------------------------------------------------------

def test_aligned_labels_raises_when_y_is_none():
    obs, _targets, _ = _labelled_cohort()
    # Build WITHOUT targets -> result.y is None.
    wide = WideRepresentationBuilder(max_waves=4).build(obs)
    assert wide.y is None
    with pytest.raises(ValueError, match="no labels"):
        _aligned_labels(wide, "train")


def test_aligned_labels_raises_on_count_mismatch():
    obs, targets, _ = _labelled_cohort(n=8)
    wide = WideRepresentationBuilder(max_waves=4).build(obs, targets)

    class _BadRep:
        patient_ids = list(wide.patient_ids) + ["EXTRA"]  # one more row than labels
        y = wide.y

    with pytest.raises(ValueError, match="Label/row count mismatch"):
        _aligned_labels(_BadRep(), "train")


# ---------------------------------------------------------------------------
# Pipeline-level: labels passed to the optimizer / final eval are aligned.
# ---------------------------------------------------------------------------

def _tiny_full_csv(tmp_path, n_patients: int = 90) -> str:
    gen_cfg = GeneratorConfig(
        n_patients=n_patients,
        min_obs_visits=2, max_obs_visits=3,
        min_future_visits=2, max_future_visits=3,
        seed=42, progression_prevalence=0.40, missingness_rate=0.0,
    )
    res = SyntheticGenerator(config=gen_cfg).generate()
    paths = res.save(output_dir=tmp_path / "data" / "synthetic")
    return str(paths["full_csv"])


def _tiny_cfg(tmp_path, model: str) -> AppConfig:
    cfg = AppConfig()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.outputs_dir = str(tmp_path / "outputs")
    cfg.paths.logs_dir = str(tmp_path / "logs")
    cfg.cohort.min_observation_visits = 2
    cfg.cohort.max_observation_visits = 3
    cfg.cohort.prediction_horizon_days = 365 * 5
    cfg.cohort.exclude_known_diabetes_at_baseline = False
    cfg.cohort.require_followup_after_cutoff = True
    cfg.split.train_size = 0.60
    cfg.split.validation_size = 0.20
    cfg.split.test_size = 0.20
    cfg.split.stratify = False
    cfg.training.seed = 42
    cfg.training.threshold = 0.5
    cfg.model.enabled_models = [model]
    cfg.optuna.n_trials = 1
    cfg.optuna.storage = None
    cfg.optuna.study_name = f"align_test_{model}"
    cfg.optuna.seed = 42
    return cfg


def _rebuild_expected_labels(cfg, csv):
    """Independently reconstruct the builder-aligned labels for each split."""
    from dialong_automl.data.loader import load_longitudinal_csv
    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode

    full_df = load_longitudinal_csv(csv)
    cohort = CohortBuilder(CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=cfg.cohort.min_observation_visits,
        max_observation_visits=cfg.cohort.max_observation_visits,
        prediction_horizon_days=cfg.cohort.prediction_horizon_days,
        exclude_known_diabetes_at_baseline=cfg.cohort.exclude_known_diabetes_at_baseline,
        require_followup_after_cutoff=cfg.cohort.require_followup_after_cutoff,
    )).build(full_df)
    split = PatientSplitter(SplitConfig(
        train_size=cfg.split.train_size,
        validation_size=cfg.split.validation_size,
        test_size=cfg.split.test_size,
        seed=cfg.split.random_seed,
        stratify=cfg.split.stratify,
    )).split(cohort.observation_data, cohort.patient_targets)

    mw = cfg.cohort.max_observation_visits
    train_wide = WideRepresentationBuilder(max_waves=mw).build(split.train_df, split.train_targets)
    val_wide = WideRepresentationBuilder(max_waves=mw).build(split.validation_df, split.validation_targets)
    test_wide = WideRepresentationBuilder(max_waves=mw).build(split.test_df, split.test_targets)

    # Ground-truth label by patient (from the cohort targets).
    tgt = cohort.patient_targets
    expected = {
        "train": {"ids": train_wide.patient_ids, "y": train_wide.y.values.astype(int)},
        "val": {"ids": val_wide.patient_ids, "y": val_wide.y.values.astype(int)},
        "test": {"ids": test_wide.patient_ids, "y": test_wide.y.values.astype(int)},
    }
    # Confirm the builder .y equals the per-patient cohort target (ground truth).
    for part in expected.values():
        for pid, lbl in zip(part["ids"], part["y"]):
            assert int(tgt.loc[pid]) == int(lbl)
    return expected


@SKIP_LGBM
def test_pipeline_passes_aligned_labels_to_optimizer(tmp_path):
    """Spy on the optimizer: y_train/y_val must equal builder-aligned labels."""
    from dialong_automl.training import pipeline as pipe_mod

    csv = _tiny_full_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path, "lightgbm")
    expected = _rebuild_expected_labels(cfg, csv)

    captured: dict = {}
    real_optimize = pipe_mod.AutoMLOptimizer.optimize

    def spy_optimize(self, *args, **kwargs):
        captured["y_train"] = np.asarray(kwargs["y_train"]).astype(int)
        captured["y_val"] = np.asarray(kwargs["y_val"]).astype(int)
        return real_optimize(self, *args, **kwargs)

    with mock.patch.object(pipe_mod.AutoMLOptimizer, "optimize", spy_optimize):
        pipe_mod.TrainingPipeline(cfg, full_csv_path=csv).run()

    np.testing.assert_array_equal(captured["y_train"], expected["train"]["y"])
    np.testing.assert_array_equal(captured["y_val"], expected["val"]["y"])


@SKIP_LGBM
def test_pipeline_passes_aligned_labels_to_final_eval(tmp_path):
    """Spy on evaluate(): the test labels must equal builder-aligned test labels."""
    from dialong_automl.training import pipeline as pipe_mod

    csv = _tiny_full_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path, "lightgbm")
    expected = _rebuild_expected_labels(cfg, csv)

    eval_calls: list = []
    real_evaluate = pipe_mod.evaluate

    def spy_evaluate(y_true, y_proba, *a, **k):
        eval_calls.append({
            "prefix": k.get("prefix", ""),
            "y_true": np.asarray(y_true).astype(int),
        })
        return real_evaluate(y_true, y_proba, *a, **k)

    with mock.patch.object(pipe_mod, "evaluate", spy_evaluate):
        pipe_mod.TrainingPipeline(cfg, full_csv_path=csv).run()

    test_call = next(c for c in eval_calls if c["prefix"] == "test_")
    np.testing.assert_array_equal(test_call["y_true"], expected["test"]["y"])


@SKIP_LGBM
def test_pipeline_wide_retrain_labels_aligned(tmp_path):
    """The wide final-retrain fit must receive builder-aligned train+val labels."""
    from dialong_automl.training import pipeline as pipe_mod
    from dialong_automl.models import lightgbm_model as lgbm_mod

    csv = _tiny_full_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path, "lightgbm")

    fit_calls: list = []
    real_fit = lgbm_mod.LightGBMModel.fit

    def spy_fit(self, X, y, *a, **k):
        fit_calls.append(np.asarray(y).astype(int))
        return real_fit(self, X, y, *a, **k)

    with mock.patch.object(lgbm_mod.LightGBMModel, "fit", spy_fit):
        pipe_mod.TrainingPipeline(cfg, full_csv_path=csv).run()

    # The LAST fit is the final retrain on train+val. Its label vector must
    # match a freshly rebuilt train+val wide builder .y (aligned by patient).
    from dialong_automl.data.loader import load_longitudinal_csv
    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode

    full_df = load_longitudinal_csv(csv)
    cohort = CohortBuilder(CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=cfg.cohort.min_observation_visits,
        max_observation_visits=cfg.cohort.max_observation_visits,
        prediction_horizon_days=cfg.cohort.prediction_horizon_days,
        exclude_known_diabetes_at_baseline=cfg.cohort.exclude_known_diabetes_at_baseline,
        require_followup_after_cutoff=cfg.cohort.require_followup_after_cutoff,
    )).build(full_df)
    split = PatientSplitter(SplitConfig(
        train_size=cfg.split.train_size, validation_size=cfg.split.validation_size,
        test_size=cfg.split.test_size, seed=cfg.split.random_seed,
        stratify=cfg.split.stratify,
    )).split(cohort.observation_data, cohort.patient_targets)

    tv_df = pd.concat([split.train_df, split.validation_df], ignore_index=True)
    tv_targets = pd.concat([split.train_targets, split.validation_targets])
    tv_wide = WideRepresentationBuilder(
        max_waves=cfg.cohort.max_observation_visits
    ).build(tv_df, tv_targets)
    expected_tv = tv_wide.y.values.astype(int)

    np.testing.assert_array_equal(fit_calls[-1], expected_tv)


# ---------------------------------------------------------------------------
# GRU-path pipeline: final test + validation-metric labels stay aligned with
# the SEQUENCE representation rows. (Gap #1 from the final review.)
# ---------------------------------------------------------------------------

def _rebuild_expected_seq_labels(cfg, csv):
    """Rebuild builder-aligned labels per split using the SEQUENCE builder.

    The pipeline sources y_val/y_test from the wide builder but the
    _assert_same_patient_order guard forces wide/sequence patient order to be
    identical, so sequence-builder .y must equal what the pipeline pairs with
    sequence predictions. We reconstruct via the sequence builder to prove the
    GRU evaluation receives sequence-row-aligned labels.
    """
    from dialong_automl.data.loader import load_longitudinal_csv
    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode

    full_df = load_longitudinal_csv(csv)
    cohort = CohortBuilder(CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=cfg.cohort.min_observation_visits,
        max_observation_visits=cfg.cohort.max_observation_visits,
        prediction_horizon_days=cfg.cohort.prediction_horizon_days,
        exclude_known_diabetes_at_baseline=cfg.cohort.exclude_known_diabetes_at_baseline,
        require_followup_after_cutoff=cfg.cohort.require_followup_after_cutoff,
    )).build(full_df)
    split = PatientSplitter(SplitConfig(
        train_size=cfg.split.train_size,
        validation_size=cfg.split.validation_size,
        test_size=cfg.split.test_size,
        seed=cfg.split.random_seed,
        stratify=cfg.split.stratify,
    )).split(cohort.observation_data, cohort.patient_targets)

    mw = cfg.cohort.max_observation_visits
    train_seq = SequenceRepresentationBuilder(max_seq_len=mw).build(
        split.train_df, split.train_targets
    )
    vocab = dict(train_seq.categorical_vocabularies)
    val_seq = SequenceRepresentationBuilder(
        max_seq_len=mw, categorical_vocabularies=vocab
    ).build(split.validation_df, split.validation_targets)
    test_seq = SequenceRepresentationBuilder(
        max_seq_len=mw, categorical_vocabularies=vocab
    ).build(split.test_df, split.test_targets)

    tgt = cohort.patient_targets
    out = {
        "val": {"ids": val_seq.patient_ids, "y": val_seq.y.values.astype(int)},
        "test": {"ids": test_seq.patient_ids, "y": test_seq.y.values.astype(int)},
    }
    # Sanity: sequence .y equals the per-patient cohort target (ground truth).
    for part in out.values():
        for pid, lbl in zip(part["ids"], part["y"]):
            assert int(tgt.loc[pid]) == int(lbl)
    return out


@SKIP_TORCH
def test_pipeline_gru_eval_labels_aligned_with_sequence_rows(tmp_path):
    """GRU run: evaluate() test_ and val_ labels equal sequence-builder .y."""
    from dialong_automl.training import pipeline as pipe_mod

    csv = _tiny_full_csv(tmp_path)
    cfg = _tiny_cfg(tmp_path, "gru")
    # Keep the GRU fast and deterministic.
    cfg.gru.epochs = 2
    cfg.gru.patience = 2
    expected = _rebuild_expected_seq_labels(cfg, csv)

    eval_calls: list = []
    real_evaluate = pipe_mod.evaluate

    def spy_evaluate(y_true, y_proba, *a, **k):
        eval_calls.append({
            "prefix": k.get("prefix", ""),
            "y_true": np.asarray(y_true).astype(int),
        })
        return real_evaluate(y_true, y_proba, *a, **k)

    with mock.patch.object(pipe_mod, "evaluate", spy_evaluate):
        pipe_mod.TrainingPipeline(cfg, full_csv_path=csv).run()

    test_call = next(c for c in eval_calls if c["prefix"] == "test_")
    val_call = next(c for c in eval_calls if c["prefix"] == "val_")

    np.testing.assert_array_equal(test_call["y_true"], expected["test"]["y"])
    np.testing.assert_array_equal(val_call["y_true"], expected["val"]["y"])


# ---------------------------------------------------------------------------
# Stratified-split alignment: reproduce the shuffled patient-ID ordering that
# triggered the original bug, under stratify=True. (Gap #2/#4.)
# ---------------------------------------------------------------------------

@SKIP_LGBM
def test_pipeline_aligned_labels_with_stratified_split(tmp_path):
    """With stratify=True (the research setting), optimizer labels stay aligned."""
    from dialong_automl.training import pipeline as pipe_mod

    csv = _tiny_full_csv(tmp_path, n_patients=120)
    cfg = _tiny_cfg(tmp_path, "lightgbm")
    cfg.split.stratify = True  # research-style stratified split (shuffles IDs)

    expected = _rebuild_expected_labels(cfg, csv)

    # Precondition: the stratified split must produce a non-sorted ID order,
    # otherwise this case would not exercise the original bug.
    from dialong_automl.data.loader import load_longitudinal_csv
    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
    full_df = load_longitudinal_csv(csv)
    cohort = CohortBuilder(CohortConfig(
        mode=CohortMode.FUTURE_OUTCOME,
        min_observation_visits=cfg.cohort.min_observation_visits,
        max_observation_visits=cfg.cohort.max_observation_visits,
        prediction_horizon_days=cfg.cohort.prediction_horizon_days,
        exclude_known_diabetes_at_baseline=cfg.cohort.exclude_known_diabetes_at_baseline,
        require_followup_after_cutoff=cfg.cohort.require_followup_after_cutoff,
    )).build(full_df)
    split = PatientSplitter(SplitConfig(
        train_size=cfg.split.train_size, validation_size=cfg.split.validation_size,
        test_size=cfg.split.test_size, seed=cfg.split.random_seed, stratify=True,
    )).split(cohort.observation_data, cohort.patient_targets)
    assert split.train_ids != sorted(split.train_ids), (
        "stratified split precondition: train_ids must not already be sorted"
    )

    captured: dict = {}
    real_optimize = pipe_mod.AutoMLOptimizer.optimize

    def spy_optimize(self, *args, **kwargs):
        captured["y_train"] = np.asarray(kwargs["y_train"]).astype(int)
        captured["y_val"] = np.asarray(kwargs["y_val"]).astype(int)
        return real_optimize(self, *args, **kwargs)

    with mock.patch.object(pipe_mod.AutoMLOptimizer, "optimize", spy_optimize):
        pipe_mod.TrainingPipeline(cfg, full_csv_path=csv).run()

    np.testing.assert_array_equal(captured["y_train"], expected["train"]["y"])
    np.testing.assert_array_equal(captured["y_val"], expected["val"]["y"])


# ---------------------------------------------------------------------------
# _assert_same_patient_order exercised through REAL builder outputs. (Gap #3.)
# ---------------------------------------------------------------------------

def test_assert_same_patient_order_passes_on_real_builders():
    """Real wide + sequence builds of the same split must satisfy the guard."""
    obs, targets, _ = _labelled_cohort(n=30)
    split = _shuffled_split(obs, targets)
    for name, df, tgts in (
        ("train", split.train_df, split.train_targets),
        ("validation", split.validation_df, split.validation_targets),
        ("test", split.test_df, split.test_targets),
    ):
        wide = WideRepresentationBuilder(max_waves=4).build(df, tgts)
        seq = SequenceRepresentationBuilder(max_seq_len=4).build(df, tgts)
        # Must not raise, and the orders must genuinely match (not vacuously).
        _assert_same_patient_order(wide, seq, name)
        assert wide.patient_ids == seq.patient_ids
        assert len(wide.patient_ids) > 0
