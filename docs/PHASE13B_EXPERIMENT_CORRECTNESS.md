# Phase 13B: Experimental correctness fixes

Scope: four correctness fixes identified in
`docs/research/phase_13a_synthetic_validity_and_protocol.md`. Nothing here changes the
synthetic generator, datasets, or any stored experiment output.

## 1. Metric labelling (`model_metadata.json`, `run_summary.json`, `metrics.json`)

| Key | Model | Evaluated on | Status |
|---|---|---|---|
| `validation_metrics` (`val_*`) | best Optuna trial, fitted on **train only** | validation split | Not that model's training data, but **biased by selection** (see "Limitations of validation metrics") |
| `retrain_resubstitution_metrics` (`resub_*`) | **final** model retrained on train+validation | validation patients (in its training data) | **Resubstitution**, not generalisation |
| `test_metrics` (`test_*`) | final retrained model | untouched test split | Not used for any selection |

`validation_metrics.val_auroc` now equals `validation_auroc_best`. Artefacts written before
Phase 13B (for example `artifacts/research*`) used the old, mislabelled `validation_metrics`
(post-retrain) and should be treated as such; they were not modified.

## 2. Optuna study isolation

Effective study name: `<optuna.study_name>__<12-hex identity hash>`. The identity covers dataset
content (SHA-256), cohort, split, training seed/threshold, GRU and model config, sampler seed,
direction, metric, sampler/pruner names and optional `optuna.run_id`. It excludes `n_trials`,
`timeout_seconds`, `storage` and `training.threshold_strategy`.

* Persistent storage requires an identity (the pipeline always supplies one).
* A study with the same identity already in storage is **refused** (`StudyExistsError`) unless
  `optuna.resume: true`. Use `optuna.run_id` (for example a fold label) to start an independent study
  for an otherwise identical configuration.
* Resume: `n_trials` is the total; completed trials are re-hydrated (hyperparameters and validation
  predictions are stored as trial attributes) so the leaderboard and best-trial selection cover the
  whole study. The stored identity must match exactly.
* The study name and hash are recorded in `best_config.json`, `model_metadata.json`,
  `run_summary.json` and `PipelineResult`.
* Existing databases are never deleted or rewritten. New studies are added alongside old ones.

## 3. GRU pruning

`GRUModel.fit` now raises `optuna.TrialPruned` when `trial.should_prune()` is true (after reporting
the epoch's validation AUROC). Previously it `break`-ed and returned, so the trial was recorded as
COMPLETE with a partial score. Normal patience-based early stopping and best-epoch restoration are
unchanged. The post-search log line no longer fails when no trial completed.

## 4. Threshold strategy (`training.threshold_strategy`)

* `fixed`: use `training.threshold`.
* `validation_f1`: maximise F1 on the validation split using the best trial's pre-retrain
  validation predictions (`evaluation/threshold.py`). The selector has no test-set parameters.
* Any other value raises `ValueError`.
* The chosen value and provenance (`threshold_selection`: strategy, `selected_on`, sample counts,
  fallback reason, `test_set_used: false`) are stored in the metadata. The test set is evaluated
  with this threshold unchanged.
* If F1 is undefined at every candidate, the configured fixed threshold is used and the fallback
  is recorded explicitly.

## Limitations of validation metrics (read before citing any number)

The single validation split is used for **all** of the following:

* choosing the model family and hyperparameters (Optuna objective = validation AUROC);
* GRU early stopping and best-epoch restoration, in every search trial;
* the decision threshold when `training.threshold_strategy = validation_f1`.

Consequently `validation_metrics` are **not an unbiased estimate of generalisation performance**
and must not be described as independent or held-out performance. The best-of-N selection and
the early stopping make them optimistic. The threshold is derived from the **pre-retrain**
model's validation probabilities and applied unchanged to the **final retrained** model, whose
probabilities may be distributed differently (`threshold_selection.applied_to`). `test_metrics`
are the only figures not touched by selection, and with few test positives they are noisy.
Nested or repeated cross-validation, which would address this, is deliberately out of scope here.
These caveats are stored in `model_metadata.json` (`validation_metrics_source`, `limitations`).

---

# Phase 13B.1: remaining correctness fixes

## GRU search vs final retraining

Reviewed and found **identical**: network architecture and every hyperparameter (the selected set
is passed to `build_model` unchanged, including `epochs`, `patience`, `seed`), loss (BCE with
logits), optimiser, validation-AUROC early stopping with best-epoch restoration, masks/lengths
derived from the same mask rule, and the sequence representation and categorical vocabulary
(taken from the train split).

Differences that are inherent to retraining (now recorded in `retrain_policy`, not hidden):

| | Search trial | Final GRU retrain |
|---|---|---|
| Training patients | train split | stratified share of train+validation (same size as the validation split is held out) |
| Early-stopping data | validation split | internal holdout of the train+validation pool |
| Preprocessor statistics | fitted on train | fitted on internal-train |

Wide models are refit on all of train+validation with no early stopping.

Corrected hazards:

* The internal holdout is now always label-stratified (previously only when `split.stratify` was
  true) and the pipeline raises if either internal partition has a single class.
* `GRUModel.fit` raises if the early-stopping labels contain a single class. Previously the
  undefined AUROC was silently scored as 0.5, which restores the epoch-0 weights.
* The retrained network's input layout is checked against the searched layout.

## Failed vs pruned trials

* `optuna.TrialPruned` is re-raised untouched (genuine pruning).
* Any other exception in training, inference or evaluation is recorded on the trial
  (`failure_stage`, `failure_type`, `failure_message`, `failure_traceback`) and re-raised, so Optuna
  marks it **FAIL**. The search continues (`catch=(Exception,)`); `KeyboardInterrupt` is not caught.
* Undefined validation AUROC (single-class labels) and missing sequence data are failures, not pruning.
* Failed trials never enter the leaderboard or best-trial selection. If no trial completes and
  at least one failed, `NoCompletedTrialsError` is raised with the first failure.
* The pipeline writes `outputs/failed_trials.json` and records `optuna_n_trials_failed` and
  `optuna_trial_states` in the metadata.

## Sampler, pruner and other `OptunaConfig` fields

| Field | Supported values | Applied by |
|---|---|---|
| `sampler` | `TPESampler`, `RandomSampler` (seeded with `optuna.seed`) | `make_sampler` |
| `pruner` | `MedianPruner`, `NopPruner` | `make_pruner` |
| `metric` | `roc_auc` | the objective (validation AUROC) |
| `direction` | `maximize` | best-trial selection maximises AUROC |
| `n_jobs` | `1` | parallel trials are not supported |

Anything else raises a validation error at configuration load (and `ValueError` in the optimizer).
All of these values are part of the study identity (identity version `13b1-1`).

## Resume semantics

* `n_trials` is the **total** number of finished trials (COMPLETE + PRUNED + FAIL) wanted; resuming
  runs only the remainder, so repeated resumes never add duplicates.
* Only COMPLETE trials are restored into the leaderboard, with hyperparameters and validation
  predictions read back from stored trial attributes. PRUNED and FAIL trials count towards the total
  but are never treated as completed.
* Interrupted (RUNNING/WAITING) trials left by a killed process are ignored and not modified.
* A COMPLETE trial lacking the stored metadata (`model_name`, `representation`,
  `hyperparams_json`, `val_proba`) or with unreadable metadata makes the resume fail with a clear
  `StudyIdentityMismatchError`. Existing trials and other studies in the same database are never rewritten.
