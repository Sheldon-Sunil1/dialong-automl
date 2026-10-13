"""Phase 8 regression tests — GRU final-retrain early stopping fix.

Context
-------
Previously the pipeline retrained the selected GRU on train+val WITHOUT any
validation signal, so ``GRUModel.fit`` ran all epochs with no early stopping
and never restored best-epoch weights (the saved model kept last-epoch
weights). The fix carves an INTERNAL patient-level validation split out of the
combined original train+val patients and passes it to ``GRUModel.fit``, while
leaving the held-out TEST set untouched until final evaluation.

These tests verify:
1. The final GRU retrain call receives non-empty internal validation tensors.
2. ``GRUModel.fit`` restores best-epoch weights when validation is supplied.
3. The internal split never touches the test patients (no selection leakage).
4. The non-GRU (wide) retrain path is unchanged (no internal split, fit called
   without validation).
5. Internal train/val patient IDs are disjoint and cover all train+val patients.

Spy/mock assertions are preferred over fragile AUROC thresholds.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from dialong_automl.config.schema import AppConfig
from dialong_automl.data.synthetic_generator import GeneratorConfig, SyntheticGenerator
from dialong_automl.training.pipeline import TrainingPipeline, _internal_patient_split


def _torch_ok() -> bool:
    try:
        import torch
        torch.tensor([1.0])
        return True
    except (ImportError, OSError):
        return False


def _lgbm_ok() -> bool:
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


SKIP_TORCH = pytest.mark.skipif(not _torch_ok(), reason="torch unavailable")
SKIP_LGBM = pytest.mark.skipif(not _lgbm_ok(), reason="lightgbm unavailable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_full_csv(tmp_path, n_patients: int = 120) -> str:
    gen_cfg = GeneratorConfig(
        n_patients=n_patients,
        min_obs_visits=2, max_obs_visits=3,
        min_future_visits=2, max_future_visits=3,
        seed=42, progression_prevalence=0.40, missingness_rate=0.0,
    )
    res = SyntheticGenerator(config=gen_cfg).generate()
    paths = res.save(output_dir=tmp_path / "data" / "synthetic")
    return str(paths["full_csv"])


def _cfg(tmp_path, model: str) -> AppConfig:
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
    cfg.gru.epochs = 3
    cfg.gru.patience = 2
    cfg.model.enabled_models = [model]
    cfg.optuna.n_trials = 1
    cfg.optuna.storage = None
    cfg.optuna.study_name = f"retrain_test_{model}"
    cfg.optuna.seed = 42
    return cfg


# ---------------------------------------------------------------------------
# 1. The final GRU retrain receives non-empty internal validation data
# ---------------------------------------------------------------------------

@SKIP_TORCH
def test_gru_final_retrain_receives_internal_validation(tmp_path):
    """Spy on GRUModel.fit: the LAST (retrain) call must pass internal val data."""
    from dialong_automl.models import gru as gru_mod

    csv = _tiny_full_csv(tmp_path)
    cfg = _cfg(tmp_path, "gru")

    real_fit = gru_mod.GRUModel.fit
    calls: list[dict] = []

    def spy_fit(self, X, y, **kwargs):
        calls.append({
            "n_train": len(X),
            "X_val": kwargs.get("X_val"),
            "y_val": kwargs.get("y_val"),
            "mask_val": kwargs.get("mask_val"),
            "lengths_val": kwargs.get("lengths_val"),
        })
        return real_fit(self, X, y, **kwargs)

    with mock.patch.object(gru_mod.GRUModel, "fit", spy_fit):
        TrainingPipeline(cfg, full_csv_path=csv).run()

    # The final retrain is the LAST GRU fit call.
    retrain_call = calls[-1]
    assert retrain_call["X_val"] is not None, "retrain must pass X_val"
    assert retrain_call["y_val"] is not None, "retrain must pass y_val"
    assert retrain_call["mask_val"] is not None, "retrain must pass mask_val"
    assert retrain_call["lengths_val"] is not None, "retrain must pass lengths_val"
    assert len(retrain_call["X_val"]) > 0, "internal validation must be non-empty"
    assert len(retrain_call["y_val"]) == len(retrain_call["X_val"])
    # Both classes should normally be present, but at minimum y_val is non-empty.
    assert len(retrain_call["y_val"]) > 0


# ---------------------------------------------------------------------------
# 2. GRUModel.fit restores best-epoch weights when validation is provided
# ---------------------------------------------------------------------------

@SKIP_TORCH
def test_gru_restores_best_epoch_weights():
    """With a decreasing val-AUROC sequence, the fitted net == best (epoch-0) net."""
    import torch
    from dialong_automl.models.gru import GRUModel

    rng = np.random.default_rng(0)
    n, T, F = 24, 3, 6
    X = rng.standard_normal((n, T, F)).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(np.int32)
    mask = np.ones((n, T), dtype=np.uint8)

    Xv = rng.standard_normal((10, T, F)).astype(np.float32)
    yv = (rng.random(10) > 0.5).astype(np.int32)
    maskv = np.ones((10, T), dtype=np.uint8)

    model = GRUModel({
        "hidden_size": 8, "num_layers": 1, "dropout": 0.0,
        "learning_rate": 1e-2, "batch_size": 8,
        "epochs": 6, "patience": 2, "seed": 7,
    })

    captured = {}
    seq = iter([0.90, 0.70, 0.60, 0.50, 0.40, 0.30])  # strictly decreasing

    real_val_auroc = GRUModel._val_auroc

    def fake_val_auroc(self, *a, **k):
        score = next(seq)
        if "best_snapshot" not in captured:
            # First call is epoch 0 -> this is the best epoch's weights.
            captured["best_snapshot"] = {
                name: p.detach().clone() for name, p in self._net.state_dict().items()
            }
        return score

    with mock.patch.object(GRUModel, "_val_auroc", fake_val_auroc):
        model.fit(X, y, mask=mask, lengths=mask.sum(1),
                  X_val=Xv, y_val=yv, mask_val=maskv, lengths_val=maskv.sum(1))

    # Early stopping (patience=2) must trigger before all 6 epochs run, and the
    # restored weights must equal the epoch-0 (best) snapshot, NOT a later one.
    final_state = model._net.state_dict()
    assert "best_snapshot" in captured
    for name, best_tensor in captured["best_snapshot"].items():
        assert torch.allclose(final_state[name], best_tensor), (
            f"Parameter {name!r} was not restored to the best epoch."
        )


# ---------------------------------------------------------------------------
# 3. Internal split never touches the test set (no selection leakage)
# ---------------------------------------------------------------------------

@SKIP_TORCH
def test_internal_split_excludes_test_patients(tmp_path):
    """The internal train/val IDs must all be train+val patients, never test."""
    csv = _tiny_full_csv(tmp_path)
    cfg = _cfg(tmp_path, "gru")

    pipeline = TrainingPipeline(cfg, full_csv_path=csv)
    pipeline.run()

    assert pipeline._last_internal_split is not None, "GRU run must record a split"
    internal_train, internal_val = pipeline._last_internal_split
    internal_all = set(internal_train) | set(internal_val)

    # Rebuild the cohort + outer split exactly as the pipeline does to get test IDs.
    from dialong_automl.data.loader import load_longitudinal_csv
    from dialong_automl.data.cohort import CohortBuilder, CohortConfig, CohortMode
    from dialong_automl.data.splitter import PatientSplitter, SplitConfig

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

    test_ids = set(split.test_ids)
    train_val_ids = set(split.train_ids) | set(split.validation_ids)

    assert internal_all.isdisjoint(test_ids), "internal split leaked test patients"
    assert internal_all.issubset(train_val_ids), (
        "internal split must come only from train+val patients"
    )


# ---------------------------------------------------------------------------
# 4. Non-GRU (wide) retrain path is unchanged
# ---------------------------------------------------------------------------

@SKIP_LGBM
def test_wide_retrain_path_unaffected(tmp_path):
    """A wide model run performs no internal split and fits without validation."""
    from dialong_automl.models import lightgbm_model as lgbm_mod

    csv = _tiny_full_csv(tmp_path)
    cfg = _cfg(tmp_path, "lightgbm")

    real_fit = lgbm_mod.LightGBMModel.fit
    fit_kwargs: list[dict] = []

    def spy_fit(self, X, y, *args, **kwargs):
        fit_kwargs.append(dict(kwargs))
        return real_fit(self, X, y, *args, **kwargs)

    with mock.patch.object(lgbm_mod.LightGBMModel, "fit", spy_fit):
        pipeline = TrainingPipeline(cfg, full_csv_path=csv)
        pipeline.run()

    # No internal split should be recorded for a wide model.
    assert pipeline._last_internal_split is None
    # The wide fit must never receive validation kwargs.
    for kw in fit_kwargs:
        assert "X_val" not in kw and "y_val" not in kw


# ---------------------------------------------------------------------------
# 5. Internal split is disjoint, exhaustive, reproducible
# ---------------------------------------------------------------------------

def test_internal_patient_split_disjoint_and_exhaustive():
    pids = [f"P{i:03d}" for i in range(40)]
    targets = pd.Series(
        [i % 3 == 0 for i in range(40)], index=pids
    ).astype(int)
    targets.index.name = "patient_id"

    tr, va = _internal_patient_split(
        pids, targets, val_fraction=0.25, seed=42, stratify=True
    )
    assert set(tr).isdisjoint(set(va)), "internal train/val must not overlap"
    assert set(tr) | set(va) == set(pids), "internal split must cover all patients"
    assert len(va) >= 1 and len(tr) >= 1

    # Reproducible with the same seed.
    tr2, va2 = _internal_patient_split(
        pids, targets, val_fraction=0.25, seed=42, stratify=True
    )
    assert tr == tr2 and va == va2


def test_internal_patient_split_falls_back_without_two_classes():
    """Single-class targets cannot stratify; fall back to a seeded random split."""
    pids = [f"P{i:03d}" for i in range(10)]
    targets = pd.Series([0] * 10, index=pids)
    targets.index.name = "patient_id"

    tr, va = _internal_patient_split(
        pids, targets, val_fraction=0.3, seed=1, stratify=True
    )
    assert set(tr).isdisjoint(set(va))
    assert set(tr) | set(va) == set(pids)
    assert len(va) >= 1 and len(tr) >= 1
