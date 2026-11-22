# DiaLong-AutoML: Final Development Readiness Audit

Date: 2026-11-12
Scope: read-only audit of the repository at commit `fead672d17d42544053c32966d4aca509d3cc55e` (branch `phase/12-real-dataset`).
Method: Git inspection, full code reads of the Python package and React frontend (parts delegated to read-only sub-agents and cross-checked by grep and direct reads), a full run of the backend and frontend test suites, `ruff`, and `tsc --noEmit`.
No production code, configuration, dependency or dataset was modified. The only file written is this report.

> All performance figures quoted here come from the **synthetic** dataset. They are not clinical evidence and must never be presented as such.

Severity scale: **Critical** (invalidates research claims or corrupts results), **High** (likely to cause incorrect behaviour or blocks the objective), **Medium** (should be fixed before final), **Low** (polish).

Items marked *(verified)* were confirmed by me directly (code read, grep or executed command). Items marked *(sub-agent)* come from a delegated code read; I spot-checked the load-bearing ones but not every line number. Items marked *(not verified)* are inferences.

---

## 1. Executive summary

The software is functionally complete as an engineering artefact. The data, cohort, split, feature, model, AutoML, training, prediction, API, SQLite and React layers all exist and work together. The backend suite passes (557 passed, 8 skipped, 0 failed) and the frontend suite passes (41 passed). Patient-level splitting and train-only preprocessing are implemented correctly.

It is **not yet complete as a research framework**. The gap is in the evaluation layer, not the models:

1. **The Phase 11 research modules are not wired in.** Calibration, explanation stability, perturbation and the versioned result schema are implemented and unit-tested, but nothing in the training pipeline or `scripts/run_experiments.py` calls them. No saved run contains an ECE, a stability score, a robustness result, or a `RunResult` record. *(verified by grep)*
2. **There is no statistical evaluation.** One seed, one split, and no confidence intervals or significance tests. The latest valid run (v3) has 7 positive patients in its test set, so every headline metric is statistically uninformative.
3. **The reported "validation metrics" are not held-out.** They are computed after the model was retrained on train+validation. In `research_v2` this shows validation AUROC 1.0 and Brier 2.5e-8.
4. **The synthetic label is nearly a deterministic function of the observable HbA1c trend.** High AUROC (0.95 to 0.99) is the expected result of line extrapolation and says nothing about real-world performance.
5. **Temporal occlusion is confounded for every visit except the last.** The GRU derives sequence length from the mask sum, so occluding a middle visit drops the final visit.
6. **One functional bug blocks the UI:** CORS does not allow PUT or DELETE, so deleting a saved visit from the browser will fail.
7. **Repository hygiene:** Phases 10 and 11 are committed on feature branches but not merged to `main`. A raw real-data folder (MIMIC-IV demo `.csv.gz`, Shanghai zip) is untracked and not covered by `.gitignore`. The README is stale (it says Phase 2+3 is the current status).

**Real-data readiness:** the models, preprocessors and evaluation modules can be reused, but only after a separate adapter produces the same long-table schema (about 20 hard-coded columns, HbA1c in %). The Phase 12 profiling reports conclude that neither UCI-130 nor ShanghaiT2DM supports the original progression objective. No dataset has yet been identified that does.

**Single highest-priority next task:** build a unified research evaluation runner. See section 11.

---

## 2. Current repository and Git status

| Item | Finding |
|---|---|
| Current branch | `phase/12-real-dataset` |
| HEAD | `fead672` "feat: add research evaluation foundation" |
| HEAD vs `phase/11-research-foundation` | Same commit. Phase 12 has **no commits of its own** yet. |
| `main` | `6b0d338` (merge of Phase 9 PR #7) |
| Commits on HEAD not in `main` | 2: `e3cbfb1` (Phase 10: SQLite persistence and visit selection), `fead672` (Phase 11). **Phases 10 and 11 are committed and pushed to `origin/phase/10-frontend` and `origin/phase/11-research-foundation`, but not merged to `main`.** |
| Stashes | None |
| Remote | `origin` = `github.com/Sheldon-Sunil1/dialong-automl` |
| Working tree | **Not clean.** Tracked files are unmodified. Only untracked entries exist. |

Untracked entries (`git status --porcelain`):

| Path | Note |
|---|---|
| `data/` (MIMIC-IV demo 2.2 `.csv.gz` files, `LICENSE.txt`, `README.txt`, `SHA256SUMS.txt`; `data/real/shanghai_diabetes/data.zip`) | Raw third-party datasets. `.gitignore` covers `*.csv` and `*.db` but **not `*.csv.gz` or `*.zip`**, so `git add data/` would stage them. |
| `docs/research/phase_12_1_uci_dataset_audit.md` | Phase 12.1 report |
| `docs/research/phase_12_3_shanghai_dataset_profile.md` | Phase 12.3 report |
| `scripts/profile_uci_dataset.py`, `scripts/profile_shanghai_dataset.py` | Profiling scripts |

Ignored but present: `artifacts/`, `outputs/`, `*.log` (`run_v2_console.log`, `run_v3_console.log`), `.venv/`, `data/dialong_patients.db`.

Phase 12.2 (MIMIC-IV demo) has files on disk but no report, script or commit. The tooling for 12.1 and 12.3 is uncommitted.

Notes:
- `artifacts/` and `outputs/` are git-ignored, so **no trained artefact or experiment result is version-controlled**. The reported v2/v3 numbers cannot be recovered from Git.
- The `.dockerignore` file is itself listed in `.gitignore` (line 113 per sub-agent), so it cannot be committed. *(sub-agent)*

---

## 3. Phase-by-phase completion table

Phase names are taken from commit history and `docs/`. There is no single authoritative phase plan: `README.md` still says "Current status: Phase 2 + Phase 3 complete", lists Phases 4 to 8 as "Planned", and uses a different phase numbering from the commits. Only `docs/PHASE10_PERSISTENCE.md` exists as a phase document.

Status key: **Complete** = implemented, tested, and I found no blocking defect. **Partial** = implemented but with material gaps. **Not complete** = not delivered.

| Phase | Scope (from commits) | Committed / merged | Actual status | Evidence and gaps |
|---|---|---|---|---|
| 1 Foundation | Scaffold, config, logging, seed, registry | Merged to main | **Complete** | Config schema (Pydantic v2), seed utilities, tests pass. Some config fields are inert (see 8.2). |
| 2 Data | Synthetic generator, loader, validator, profiler | Merged | **Partial** | Generator, loader, profiler work. `DataValidator` is **not called by the training pipeline**; it is a standalone utility. *(verified by grep)* |
| 3 Cohort and splitting | Cohort builder, leakage protection, patient splitter | Merged | **Complete** (for synthetic data) | Future-outcome cohort, horizon, baseline-diabetes exclusion, leakage guard, stratified patient split with `assert_no_overlap`. Real-data weaknesses in 7. |
| 4 Representations | Wide and sequence builders, preprocessing | Merged | **Complete** (for synthetic data) | Train-only fitting confirmed. No missing indicators or time-delta features. |
| 5 Models | LR, RF, XGBoost, LightGBM, GRU | Merged | **Complete** | All five implemented and tested. GRU pruning is ineffective; `GRUModel.predict_proba` ignores padding. |
| 6 AutoML | Optuna search, leaderboard | Merged | **Partial** | Works, but single validation split (no CV), exceptions swallowed into pruned trials, study reuse can mix stale trials. |
| 7 Explainability | SHAP, temporal occlusion | Merged | **Partial** | SHAP works for tree models. Occlusion has a correctness bug for non-final visits. Not exposed through API or UI. |
| 8 CLI, prediction, experiments | `train`, `predict`, `run_experiments`, label and cohort fixes | Merged | **Partial** | CLI works. Experiment runner records no provenance (git hash, package versions, data hash) and mislabels post-retrain validation metrics. |
| 9 API | FastAPI serving | Merged (PR #7) | **Partial** | `/health`, `/metadata`, `/predict` work. No auth, no backend range validation, model reloaded from disk on every request. |
| 10 Frontend and persistence | React UI, SQLite patients and visits | Committed on `phase/10-frontend`, **not merged** | **Partial** | Works and is tested (41 frontend tests). Defects: CORS blocks PUT and DELETE, delete-then-switch race, partial-save duplicates, no explanation UI. |
| 11 Research foundation | Calibration, stability, perturbation, result schema | Committed on `phase/11-research-foundation`, **not merged** | **Not complete** (library only) | Four modules and tests exist. **None is used by any pipeline or script.** `enable_calibration_metrics` and `enable_stability_metrics` default to `False` and are documented as "Reserved". No run has ever produced Phase 11 evidence. |
| 12 Real dataset | UCI audit, MIMIC demo, Shanghai profile | Nothing committed. 2 reports and 2 scripts untracked | **Not complete** (audit stage only) | UCI-130 and ShanghaiT2DM both reported as unsuitable for the original objective. MIMIC demo has no report. No adapter, no loader, no real-data model run. |

Phases 1 to 9 exist on `main`. Phases 10 and 11 are on branches only.

---

## 4. Core ML pipeline assessment

### 4.1 Verified working

| Area | Finding |
|---|---|
| Synthetic generation | Seeded generator (`GENERATOR_VERSION 1.1.0`), 4 to 6 observation visits plus 2 to 4 future visits, 10 numeric and 6 categorical features. |
| Patient and label alignment | `_aligned_labels` takes labels from the builder's `.y` rather than the splitter's shuffled order, and `_assert_same_patient_order` fails loudly on mismatch. Dedicated tests in `test_label_alignment.py` pass. |
| Cohort | Index date = last observation visit; 365-day horizon; exclusion of patients already diabetic in the observation window (by diagnosis or HbA1c >= 6.5); runtime assertion that no observation row postdates the cutoff. |
| Patient-level split | Two-step stratified shuffle on patient-level target, deterministic seed, both classes verified per split. |
| Leakage prevention | Train-only fitting of `WidePreprocessor` and `SequencePreprocessor`; categorical vocabulary built from train and reused for val and test; leakage columns rejected in builders and API. |
| Test-set discipline | Test arrays are transformed with train-fitted preprocessors and evaluated once after model selection. |
| Models | All five wrappers implemented, with registry and persistence. Tests pass. |
| Optuna | Seeded TPE sampler, SQLite storage, per-trial metrics stored. |
| Prediction | Loads metadata, model and preprocessor, rejects leakage columns, returns probability, class and disclaimer. |

### 4.2 Issues

| ID | Severity | Issue | Location |
|---|---|---|---|
| ML-1 | **Critical** | **Synthetic label is nearly deterministic from observable HbA1c dynamics.** `will_progress` only sets the HbA1c slope; the target is "any future HbA1c >= 6.5". Slope is recoverable from 3 to 4 noisy observations (noise SD 0.1). Observed performance (v3 val AUROC 0.991, test 0.950) is the expected outcome of trend extrapolation. The task is also structurally "predict threshold crossing of an input variable". | `synthetic_generator.py` L411, L457-459, L527-560 *(sub-agent)*; metrics from `artifacts/research_v3/model_metadata.json` |
| ML-2 | **Critical** | **Reported "validation metrics" are post-retrain, not held-out.** `val_metrics` are computed with the model retrained on train+val and saved as `validation_metrics` in `model_metadata.json` and `metrics.json`. The honest held-out figure is `validation_auroc_best`. | `training/pipeline.py` (retrain at "Step 9", `val_metrics = evaluate(y_val, val_proba_final ...)` in "Step 10") *(verified by read)* |
| ML-3 | **High** | **No cross-validation or repeated splits; test set has 7 positives (v3).** Optuna selects the best of 40 trials on one 118-patient validation set (about 7 positives). Selection noise is large. | `automl/optimizer.py` `_objective`; `artifacts/research_v3/model_metadata.json` |
| ML-4 | **High** | **GRU early stopping, checkpoint restore, pruning and model selection all use the same validation set.** The validation score is optimistic. | `models/gru.py` fit; `optimizer.py` `_objective` *(verified by read)* |
| ML-5 | **High** | **Inert configuration.** `threshold_strategy: "validation_f1"` is validated but never read; the threshold is fixed at 0.5 (recall 3/7 on the v3 test set). `select_threshold_by_f1` is imported in the pipeline and unused. The pipeline hard-codes `CohortMode.FUTURE_OUTCOME` and ignores `cfg.cohort.mode`. Other fields reported as unread by the sub-agent: `optuna.pruner/sampler/n_jobs/metric`, `model.hyperparameter_overrides`. | `config/schema.py` L181-197; `pipeline.py` L55, L350 *(verified for threshold and cohort mode; others sub-agent)* |
| ML-6 | **High** | **Optuna: swallowed exceptions and ineffective GRU pruning.** Any training or inference exception becomes `TrialPruned`, so a systematic failure shows up only as "no completed trials". The GRU `trial.report`/`should_prune` path only `break`s from the loop and still returns a completed value. | `optimizer.py` `_objective`; `gru.py` fit *(verified by read)* |
| ML-7 | **High** | **Optuna study reuse.** `load_if_exists=True` with a fixed `study_name`: a rerun with changed data or code continues from stale trials, while `best_trial_result()` uses only the in-memory results of the current run. The DB and leaderboard can disagree. A stray `artifacts/research/artifacts/research/optuna_study.db` records a previous path-nesting bug. | `optimizer.py` (create_study); `pipeline.py` storage resolution *(verified for create_study; rest sub-agent)* |
| ML-8 | **High** | **Missing values are treated as clinical negatives.** A missing future HbA1c is dropped, so "unmeasured" becomes label 0. A missing observation HbA1c cannot trigger the baseline-diabetes exclusion. The synthetic data never exercises this because future rows have no missingness. | `cohort.py` `_derive_future_target` and exclusion block *(verified by read)* |
| ML-9 | **Medium** | No missing-value indicators and no time-delta features. Median imputation turns missing and padded values into 0 after scaling. The wide slope is per visit index, not per day. Visit gaps vary from 90 to 240 days, so timing information is discarded. | `preprocessing.py`; `wide.py` L336-349; `sequence.py` *(sub-agent)* |
| ML-10 | **Medium** | GRU final retrain uses a different recipe from the one validated in Optuna: an internal patient split of train+val, preprocessor fitted on about 85% of the data. | `pipeline.py` GRU retrain branch *(verified by read)* |
| ML-11 | **Medium** | Wide-model train and serve skew: training sees the last 4 visits (cohort tail), but at prediction time a longer trajectory gives waves from the **oldest** 4 visits while summary features use all visits. The sequence path truncates consistently. | `wide.py` L316; `predict.py` *(sub-agent, not verified)* |
| ML-12 | **Medium** | `predict.py` does no unit or range validation, no minimum-visit rule, no baseline-diabetes exclusion, no feature-count check, and no library-version check. It uses fixed artefact filenames, so files from different runs in one directory can be mixed. | `predict.py` *(sub-agent)* |
| ML-13 | **Medium** | `HeldOutTestSet` guard class is never instantiated. The metadata flag `test_set_accessed_for_selection: false` is hard-coded rather than measured. | `pipeline.py`; grep confirms no `HeldOutTestSet(` call *(verified)* |
| ML-14 | **Medium** | Artefact loading uses pickle and `torch.load(weights_only=False)`. Only safe for trusted artefacts. | `models/*.py`, `preprocessing.py`, `gru.py` L219 *(sub-agent)* |
| ML-15 | **Medium** | Loader converts categorical NaN to the string `"nan"` and coerces bad numerics to NaN before any validation. Combined with ML-8 (validator not in the pipeline), bad data passes silently. | `loader.py` L194-200 *(sub-agent)* |
| ML-16 | **Medium** | Wide builder orders patients by first appearance (`groupby(sort=False)`), sequence builder by sorted ID. The assertion passes only because the synthetic CSV is sorted. On unsorted real data it will raise (loud, not silent). | `wide.py`; `sequence.py` L280 *(sub-agent)* |
| ML-17 | **Low** | `GRUModel.predict_proba` ignores padding (full-length assumption); only `predict_proba_with_mask` is correct. Global `np.random` used for shuffling; `random.seed()` global side effect in the generator; `deterministic_torch: false` in all research configs. | `gru.py`; `synthetic_generator.py` L284; `configs/` *(sub-agent)* |
| ML-18 | **Low** | Unused imports (`select_threshold_by_f1`, `metrics_to_jsonable` in the pipeline). `ruff check dialong_automl` reports **174 errors** (63 auto-fixable). | `ruff` run *(verified)* |

---

## 5. Research evaluation assessment

### 5.1 Component status

| Component | Implemented | Correct | Used by any pipeline or run |
|---|---|---|---|
| Calibration (equal-width ECE, MCE, reliability bins, Brier) | Yes | Yes *(verified by read)* | **No.** Flag `enable_calibration_metrics=False`, "Reserved". No saved run contains ECE or MCE. |
| Calibration slope, intercept, post-hoc calibration (Platt, isotonic) | **No** | n/a | n/a |
| Explanation stability (top-k Jaccard, Spearman) | Yes | Yes, with caveats | **No.** |
| Temporal occlusion | Yes | **Incorrect for non-final visits** (RE-2) | Offline only |
| SHAP (TreeExplainer plus permutation-importance fallback) | Yes | Mostly | Offline only; **not exposed by API or UI** |
| Perturbation utilities (remove measurements, remove visits, truncate, jitter) | Yes, seeded, deterministic | Yes | **No consumer except tests.** |
| Result schema (`RunResult`, schema version 1.0) | Yes | Structurally valid | **No.** `run_experiments.py` writes ad-hoc JSON instead. |
| Confidence intervals, bootstrap | **No** | n/a | n/a |
| Significance tests (DeLong, paired bootstrap) | **No** | n/a | n/a |
| Multi-seed or repeated-split evaluation | **No** | n/a | n/a |
| Clinical or trivial baselines | **No** | n/a | n/a |
| Ablations (wide vs sequence under matched tuning, feature groups) | **No** | n/a | n/a |
| Decision-curve or threshold analysis | **No** (`select_threshold_by_f1` unused) | n/a | n/a |
| Subgroup analysis | **No** | n/a | n/a |

### 5.2 Saved results (all from synthetic data)

| Run | Model | Test AUROC | Test AUPRC | Brier | Test positives | Validity |
|---|---|---|---|---|---|---|
| `research` | GRU | 0.481 | 0.228 | 0.268 | 36 of 150 | Pre-label-fix. Invalid. |
| `research_v2` | LightGBM | 0.996 | 0.984 | 0.0122 | 36 of 150 | Cohort included already-diabetic patients ("near-trivial", per the v3 config header). Invalid. |
| `research_v3` | GRU | 0.950 | 0.666 | 0.0402 | **7 of 118** | Methodologically valid cohort, statistically uninformative. |

All three use the same CSV, seed 42, and one split. Source: `artifacts/*/model_metadata.json` *(sub-agent)*. The three artefact folders carry the same metadata template and no "superseded" flag.

### 5.3 Issues

| ID | Severity | Issue | Location |
|---|---|---|---|
| RE-1 | **Critical** | **No uncertainty quantification and no repeated evaluation.** With 7 test positives, AUROC 0.950 and recall 3/7 have very wide intervals. Model-family comparison and the GRU-vs-LightGBM winner flip between runs are not interpretable. | whole evaluation layer; `run_experiments.py` |
| RE-2 | **High** | **Temporal occlusion is confounded.** Occluding step `t` zeros its features and sets `mask[t]=0`. The GRU ignores mask values and uses `mask.sum()` as the sequence length, then packs the first `length` steps. For any non-final real visit, the last real visit is dropped and a zero step is inserted. Only the final visit is a clean occlusion. The influence ranks for earlier visits are unreliable. Tests do not detect this. | `temporal_occlusion.py` L105-135 (verified); `gru.py` length derivation *(verified `lengths = mask.sum` in `predict_proba_with_mask`)*; packing behaviour *(sub-agent)* |
| RE-3 | **High** | **Phase 11 modules are dead code in the research pipeline.** No claim about calibration, robustness, or stability can be supported by current artefacts. | grep: no consumers outside `evaluation/` and `tests/` *(verified)* |
| RE-4 | **High** | **Reproducibility metadata incomplete.** No git hash, package versions, Python version, data hash, split hash or patient-ID list in any artefact. The dataset CSV is reused if present ("if absent" generation), so v2 and v3 can silently share a stale CSV. Run directory names use second resolution with `exist_ok=True`. Rerunning the same config overwrites `artifacts/<name>/`. `deterministic_torch: false` in all three research configs. | `run_experiments.py` (`ensure_dataset`, `run_name`, `experiment_meta.json`) *(verified by read)* |
| RE-5 | **High** | **No baselines.** Without "latest HbA1c" or an HbA1c-threshold rule as comparators, a high AUROC on this synthetic task is uninformative (see ML-1). | n/a |
| RE-6 | **High** | **The explainability demo explains random noise.** `run_explainability_demo.py` builds a DataFrame from standard-normal noise with random labels and writes "global importance" and SHAP JSON that carry only the generic disclaimer. It also loads `artifacts/best_model.joblib`, a path that does not match the research artefact layout. | `scripts/run_explainability_demo.py` L60-73, L97-99 *(sub-agent)* |
| RE-7 | **Medium** | Metric implementations are hand-written NumPy: AUROC and AUPRC do not group tied scores and use an unstable `argsort`; AUPRC uses trapezoidal interpolation (optimistic versus step-wise average precision). Bare `except Exception` returns `None`. Results can differ from scikit-learn, especially with saturated or tied probabilities (v2). | `evaluation/metrics.py` L34-118 *(verified by read: `np.argsort(-y_score)`, per-sample cumulative sums, trapezoid)* |
| RE-8 | **Medium** | `metrics.json` and the console summary carry no synthetic-data flag or disclaimer (only `experiment_meta.json` and `model_metadata.json` do). | `run_experiments.py` *(verified by read)* |
| RE-9 | **Medium** | Stability: Spearman uses signed scores while Jaccard uses absolute scores; no aggregation across patients or perturbations; no CI. Perturbation: MCAR only, categorical or static columns blanked at the same rate as labs, jitter pinned at the first and last visit, and an O(cells) RNG construction cost. | `stability.py`; `perturbation.py` L130-146, L302-307 *(sub-agent)* |
| RE-10 | **Medium** | SHAP direction is reported in log-odds space without labelling; a mean of exactly 0 is labelled "negative"; "mixed" and "shap_linear" are documented but never emitted. `_model_supports_tree_shap` accepts logistic regression. | `shap_explainer.py` *(sub-agent)* |
| RE-11 | **Medium** | Config `progression_prevalence: 0.30` differs from the evaluated prevalence after exclusions (v3 test 5.9%). | `configs/research_experiment_v3.yaml` |
| RE-12 | **Low** | Stale artefacts: `research` and `research_v2` have no "superseded" flag. `model_metadata.json` limitations boilerplate says "GRU trained on CPU" even for the LightGBM run. | `artifacts/` |

### 5.4 Missing experiments for the stated objective

1. Repeated patient-level stratified splits (at least 5 seeds) or nested cross-validation, with per-model test metrics and 95% bootstrap CIs.
2. Baselines: prevalence, last-HbA1c, HbA1c >= 5.7 rule, logistic regression on last values only.
3. Wide versus sequence comparison under matched tuning budget.
4. Calibration on held-out predictions (ECE, MCE, slope, intercept, reliability plot) and post-hoc calibration fitted on validation only.
5. Robustness curves using the perturbation utilities (measurement removal, visit removal, truncation, jitter).
6. Explanation stability under perturbation, aggregated over patients.
7. Occlusion and SHAP validated against the generator's known drivers (the generator knows the true causal structure, which is a unique advantage of synthetic data).
8. A harder synthetic variant (lower signal, informative missingness, irregular timing) so that results are not trivially near 1.0.

---

## 6. Backend and frontend assessment

### 6.1 Endpoints

| Method and path | Purpose | Status |
|---|---|---|
| `GET /health` | Model-loaded status | Always `status: ok`. Ignores DB state. |
| `GET /metadata` | Model metadata (paths stripped) | Works |
| `POST /predict` | Stateless prediction | Works. Does not enforce the observation window. |
| `POST /patients` | Find or create | Works. Body is a raw `dict` with no OpenAPI schema. |
| `GET /patients/{id}` | History | Works |
| `POST /patients/{id}/visits` | Add visit | Works |
| `PUT /patients/{id}/visits/{vid}` | Update visit | Works server-side. **Blocked by CORS in the browser.** No UI. |
| `DELETE /patients/{id}/visits/{vid}` | Delete visit | Works server-side. **Blocked by CORS in the browser.** |
| `POST /patients/{id}/predict` | Predict from selected saved visits | Works; enforces window; ownership checks |

There is no `/explain`, `/leaderboard`, patient list or patient delete endpoint. The UI states this explicitly in an "offline research code" panel and does not fabricate explanations.

### 6.2 Verified

- Ownership checks on visits return 404 (not 403), avoiding disclosure.
- Patient find-or-create is race-safe via a unique constraint.
- SQLite uses WAL, foreign keys, and per-call sessions.
- Error messages are sanitised against path leakage.
- Frontend `usePatient` uses a monotonic request counter to ignore stale responses; the mandatory patient-switch test passes.
- A synthetic-data disclaimer appears in a banner, a footer, and the prediction card.
- Type shapes in `frontend/src/api/types.ts` match the backend schemas field-for-field *(sub-agent)*.

### 6.3 Issues

| ID | Severity | Issue | Location |
|---|---|---|---|
| FB-1 | **High** | **CORS `allow_methods=["GET","POST","OPTIONS"]` omits PUT and DELETE.** `deleteVisit` is a cross-origin DELETE from `:5173` to `:8000` and will be rejected at preflight. Tests use `TestClient`, which does not model CORS, so this is invisible to the suite. | `api/app.py` L190 *(verified)*; `App.tsx` `deleteSaved` |
| FB-2 | **High** | **No authentication or authorisation on any route**, including read and delete of patient data. Free-text patient IDs mean real identifiers could be stored. The Dockerfile is intended to bind `0.0.0.0`. Acceptable for a local research tool; unacceptable if exposed. | `api/*`; `docs/PHASE10_PERSISTENCE.md` *(sub-agent)* |
| FB-3 | **Medium** | No backend validation of numeric ranges, `visit_date` format, patient-ID length or charset, or NaN/Infinity. Visits are ordered by the **string** date. Values such as `hba1c=-5` can be stored by direct API calls. Duplicate visit dates are allowed. Duplicate `visit_ids` in a prediction request are accepted and counted as separate visits. | `schemas.py`; `repository.py` *(sub-agent)* |
| FB-4 | **Medium** | Model and preprocessor are reloaded from disk on **every** prediction. `/health` can report `model_loaded: true` while weights are unusable. | `predict.py` `predict_from_artifacts` *(sub-agent)* |
| FB-5 | **Medium** | Default artefact directory `artifacts/` contains no `model_metadata.json`. A plain `uvicorn dialong_automl.api.app:app` starts degraded (503 on predict) unless `DIALONG_ARTIFACT_DIR` is set. `app = create_app()` at module level runs at import time. | `app.py` *(verified `_resolve_artifact_dir`; directory listing verified)* |
| FB-6 | **Medium** | Delete-then-switch race: `deleteSaved` does not set `saving`, so the Change-patient button stays enabled. The stale `reload` closure then resets `intendedId` to the old patient and can overwrite the new patient's visits. | `App.tsx` `deleteSaved`; `usePatient.ts` `reload` *(verified by read: `reload` closes over `state.activeId` and sets `intendedId.current = id`)* |
| FB-7 | **Medium** | Partial-save duplicates: `saveDrafts` saves sequentially; if visit 3 of 5 fails, 1 and 2 persist, drafts remain, and a retry duplicates them (no server-side duplicate-date guard). | `App.tsx` `saveDrafts` *(verified by read)* |
| FB-8 | **Medium** | Prediction shown as a bare percentage. No calibration status, no target definition or horizon, no uncertainty. For a model with validation AUROC about 0.99 on synthetic data, this invites over-reading. | `PredictionResult.tsx` *(sub-agent)* |
| FB-9 | **Medium** | No migrations (`create_all` only). Any schema change will not apply to an existing DB file. | `db/engine.py` *(sub-agent)* |
| FB-10 | **Medium** | `/patients/{id}/predict` maps every `PredictionError` to 422, including artefact failures that `/predict` maps to 503. | `patients.py` *(verified by read)* |
| FB-11 | **Low** | Dockerfile CMD only imports the package; no API entrypoint; HEALTHCHECK does not call `/health`; compose API service is commented out. `.env.example` lists variables the code never reads and omits the ones it does (`DIALONG_ARTIFACT_DIR`, `DIALONG_DB_PATH`, `DIALONG_CORS_ORIGINS`). | `Dockerfile`, `docker-compose.yml`, `.env.example` *(sub-agent)* |
| FB-12 | **Low** | Stale UI text ("The backend does not store or look up patient records" in `PatientIdStep.tsx`). `Visit` type is a loose index signature. Selection resets to "all" after each save or delete. No request cancellation. Three different disclaimer wordings. Mojibake in some docs. | `frontend/src/**` *(sub-agent)* |
| FB-13 | **Low** | Frontend duplicates feature names, leakage set and ranges by hand (`fields.ts`) with no parity test against `wide.py`. | `fields.ts` *(sub-agent)* |

Not assessed: WCAG compliance. That requires manual testing with assistive technologies and expert review.

---

## 7. Synthetic-to-real architecture assessment

### 7.1 Can a future real dataset use the existing models and evaluation modules?

**Yes, for the models, preprocessors, metrics, calibration, stability and perturbation modules, provided a separate adapter produces the expected long table.** There is no adapter interface today, and the cohort, loader and feature builders are coupled to the synthetic schema. Of the `dialong_automl` Python files, the ones that carry schema assumptions are `data/loader.py`, `data/validator.py`, `data/cohort.py`, `data/splitter.py`, `features/wide.py`, `features/sequence.py`, `features/preprocessing.py`, `predict.py`, and the frontend `fields.ts`.

### 7.2 Hard-coded assumptions

| Aspect | Current assumption | Impact on real data |
|---|---|---|
| Schema coupling | `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` in `wide.py` L91-112 are imported by `sequence.py`, `predict.py`, `preprocessing.py`. Builders accept constructor arguments for feature lists, but the pipeline never passes them. `loader.py` `EXPECTED_COLUMNS`; validator's expected and binary column lists. | Feature lists need to be configuration-driven (about 6 files). |
| Patient ID | Column name `patient_id` hard-coded across loader, cohort, splitter, builders, predict. IDs formatted `SYN_xxxxx`. | An adapter must rename. Low effort. |
| Timestamps | Real dates are used for cutoff and horizon (good). Features ignore elapsed time (slope per visit index; no delta-time in sequences). `visit_number` and `is_future_visit` are synthetic-only; without the flag, cohort falls back to "first N visits are observation", which is not a clinical index date. | Needs a defined index date and a time-aware feature set. |
| Missing values | Median imputation, no indicators. Missing future HbA1c becomes label 0. Missing baseline HbA1c skips the exclusion rule. Loader turns NaN categoricals into `"nan"`. Synthetic missingness is 5% MCAR on observation rows only. | Material label noise on sparse real labs (ML-8, ML-9, ML-15). |
| Units | None recorded or converted. HbA1c in %, glucose and lipids in mg/dL, BP in mmHg. The 6.5 threshold is in %. | A dataset in mmol/mol or SI units breaks both the threshold and scaling. The adapter must convert and record units. |
| Outcome definition | Hard-coded in `CohortBuilder._derive_future_target`: `diabetes_diagnosis == 1` or `hba1c >= 6.5` within 365 days. `hba1c` is also an input feature. | A different outcome needs a pluggable target definition. |
| Population | Excludes baseline-diabetic patients; requires at least 3 observation visits. | Real cohorts must support the same inclusion logic or define their own. |
| Wide features | Fixed `max_waves`, `n_obs_visits`. | Fine for 3 to 4 visits; breaks for dense series such as CGM. |
| Unit/ID handling in API and UI | Patient ID is free text; visit schema fixed to 10 numerics plus 6 categoricals. | The UI cannot represent a different feature set without a code change. |
| Model artefacts | No stored feature names, units or schema hash. | Cannot detect a schema mismatch at inference. |

### 7.3 Evidence from the Phase 12 reports (uncommitted)

- **UCI Diabetes 130-US Hospitals:** no dates, all patients already diabetic, no numeric HbA1c, 76.5% single-encounter patients. The report recommends it only for a separate, re-scoped 30-day readmission experiment, and calls it "not suitable" for the original objective.
- **ShanghaiT2DM / T1DM:** 100 T2DM patients, every patient already diabetic, at most 179 days of follow-up, one HbA1c value per recording period. Objective A "not supported". Supports only short-term CGM forecasting, which needs a different adapter and task.
- **MIMIC-IV clinical demo 2.2:** files are on disk but there is no report. It is the only candidate named by the UCI report with dated labs, but the demo cohort is small and I did not inspect it. *(not verified)*

No identified dataset currently supports "future diabetes onset from dated repeated HbA1c". A real-data validation of the stated objective therefore depends on gaining access to a suitable cohort. The software can be completed without it.

### 7.4 Recommended boundary for a future adapter (design only, not built)

A `DatasetAdapter` should output (a) a long table with `patient_id`, `visit_date`, numeric and categorical features in documented units, (b) an explicit index date per patient, (c) a target and eligibility flag, (d) a schema/units manifest. The existing cohort builder, splitter, feature builders and models would consume that. Feature lists, units and outcome definition should come from configuration rather than module constants.

---

## 8. Existing bugs and technical debt

Consolidated, with the section IDs above. Some items are grouped, so no exact totals are given.

### 8.1 Critical

| ID | Issue |
|---|---|
| ML-1 | Synthetic label is a near-deterministic function of observable HbA1c trend; performance is uninformative. |
| ML-2 | `validation_metrics` are computed after retraining on train+val, mislabelled as validation performance. |
| RE-1 | No confidence intervals, repeated splits or significance tests; 7 test positives in the latest valid run. |

(In a research-paper context, "synthetic results readable as clinical evidence" is also Critical. The metadata disclaimers are present and consistent, which mitigates it, but `metrics.json`, the console summary and stale invalid artefacts lack them, see RE-8 and RE-12.)

### 8.2 High

ML-3, ML-4, ML-5, ML-6, ML-7, ML-8, RE-2, RE-3, RE-4, RE-5, RE-6, FB-1, FB-2, plus repository items:

| ID | Issue |
|---|---|
| REPO-1 | Phases 10 and 11 are not merged to `main`. |
| REPO-2 | Raw real datasets (MIMIC `.csv.gz`, Shanghai `.zip`) are untracked and unprotected by `.gitignore`. |
| REPO-3 | Trained artefacts and experiment outputs are git-ignored, so published numbers cannot be reproduced from the repository alone. |
| REPO-4 | `README.md` is stale (Phase 2+3 status; Phases 4 to 8 "Planned"; wrong phase numbering). No phase plan exists. |

### 8.3 Medium

ML-9 to ML-16, RE-7 to RE-11, FB-3 to FB-10, plus:

| ID | Issue |
|---|---|
| TD-1 | `ruff check dialong_automl` reports 174 errors (63 auto-fixable). *(verified)* |
| TD-2 | `mypy` fails to run: `python_version = "3.11"` in `pyproject.toml` while the venv is Python 3.13 and numpy stubs use Python 3.12 syntax ("Type statement is only supported in Python 3.12 and greater"). Type checking currently provides no signal. *(verified)* |
| TD-3 | `requirements.txt` lists FastAPI, uvicorn and SQLAlchemy but a grep found no `httpx` entry (it is only in the `dev` extras in `pyproject.toml`). The installed test client emits a deprecation warning that asks for `httpx2`. *(verified by grep)* |
| TD-4 | Duplicate scratch and log files at repo root (`check_deps.py`, `check_sklearn.py`, `run_v2_console.log`, `run_v3_console.log`). |

### 8.4 Low

ML-17, ML-18, RE-12, FB-11 to FB-13, plus a pandas date-parsing `UserWarning` in `validator.py:275`, a SHAP `UserWarning` about LightGBM output format, and an unused `PatientHistoryResponse.created` field.

---

## 9. Missing tests

### 9.1 Test run result *(verified, executed in this audit)*

| Suite | Command | Collected | Passed | Failed | Skipped | Warnings | Runtime |
|---|---|---|---|---|---|---|---|
| Backend (pytest, Python 3.13.15) | `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -rs` | **565** | **557** | **0** | **8** | **14** | **72.26 s** |
| Frontend (vitest) | `npx vitest --run` | 41 | **41** | 0 | 0 | not captured | 8.92 s (vitest-reported) |
| Frontend type check | `npx tsc --noEmit` | n/a | exit code 0 | | | | |

The "more than 550 passing tests" in earlier reports is consistent with the current state: 557 backend plus 41 frontend.

Skipped tests (8): five are "dependency is available, so the blocked-import path is not applicable" guards (torch, sklearn, xgboost), which is expected. **Two are real gaps:** `test_explainability.py:953` and `:1000` skip because "Real model artifact not found". That means no test exercises explanations on a real trained artefact in this environment.

Warnings (14): FastAPI/Starlette `httpx` deprecation; 12 SHAP/LightGBM output-format warnings; 1 pandas date-format warning from `validator.py`.

Not run: `ruff` and `mypy` are quality tools, covered in TD-1 and TD-2. No test was modified, and no code was changed to obtain these results.

### 9.2 Gaps

| Priority | Missing test |
|---|---|
| High | CORS preflight for PUT and DELETE (would have caught FB-1). |
| High | Temporal occlusion correctness against a mask-aware reference (would have caught RE-2). A test using the generator's known HbA1c driver is possible. |
| High | The pipeline reports held-out validation metrics, not post-retrain ones (ML-2). |
| High | Integration test that a research run writes calibration, stability, robustness and a `RunResult` (does not exist because the wiring does not exist). |
| High | End-to-end test with an unsorted-ID, sparse-lab, non-synthetic-schema table to expose adapter coupling (ML-8, ML-15, ML-16). |
| Medium | Metric parity: custom `auroc`/`auprc` versus scikit-learn, including heavily tied scores. |
| Medium | Backend validation: ranges, date formats, NaN, odd patient IDs (`/`, unicode), duplicate `visit_ids`. |
| Medium | 503-versus-422 on `/patients/{id}/predict` when artefacts disappear. |
| Medium | Frontend: delete-then-switch race; partial-save failure. |
| Medium | Frontend/backend constant parity (`fields.ts` versus `wide.py`). |
| Medium | Optuna rerun with a changed config reusing a stored study. |
| Medium | Train/serve parity for the wide model when more than `max_waves` visits are submitted. |
| Low | Docker image build and compose smoke test; multi-worker SQLite behaviour. |

---

## 10. Remaining development tasks, ranked by priority

| Rank | Task | Severity addressed | Effort |
|---|---|---|---|
| 1 | **Unified research evaluation runner:** repeated patient-level splits (or nested CV) across seeds; per-model test metrics; bootstrap 95% CIs; baselines; calibration (ECE, MCE, slope, intercept) wired in; robustness via perturbation; stability aggregated across patients; every run emitted as a `RunResult` with git hash, package versions, data hash and split hash. | RE-1, RE-3, RE-4, RE-5, ML-3 | Large |
| 2 | **Fix metric reporting:** report held-out validation metrics (from the search); label post-retrain numbers as such or remove them; add synthetic-data flag to `metrics.json`; mark superseded runs. | ML-2, RE-8, RE-12 | Small |
| 3 | **Fix CORS** to allow PUT and DELETE; add a preflight test. | FB-1 | Trivial |
| 4 | **Repository hygiene:** add `*.csv.gz`, `*.zip`, `data/real/` to `.gitignore`; commit the Phase 12.1 and 12.3 reports and scripts (not the data); merge Phases 10 and 11 to `main`; update `README.md` and write a phase plan. | REPO-1 to REPO-4 | Small |
| 5 | **Harder synthetic benchmark:** add a generator mode with lower signal, informative missingness and irregular timing, so the framework is not evaluated only on a near-trivial task. | ML-1 | Medium |
| 6 | **Fix temporal occlusion:** make occlusion mask-aware (for example, a GRU path that skips the occluded step, or replace with a learned baseline), validate against the generator's known drivers. | RE-2 | Medium |
| 7 | **Pipeline honesty fixes:** wire or remove `threshold_strategy`, honour `cohort.mode`, remove inert config, stop swallowing training exceptions, make GRU pruning real, validate Optuna study reuse (suffix the study name with a config hash), instantiate `HeldOutTestSet` or remove it. | ML-4 to ML-7, ML-13 | Medium |
| 8 | **Make the data layer schema-driven:** feature lists, units, ID and time column names, and outcome definition from configuration; call `DataValidator` in the pipeline; handle missing future HbA1c explicitly; add missing-indicator and time-delta features. | Section 7; ML-8, ML-9, ML-15, ML-16 | Large |
| 9 | **Expose explanations:** add an `/explain` endpoint and UI panel for SHAP and (fixed) occlusion, with the synthetic-data disclaimer. Optionally show calibration context beside the risk. | FB-8 | Medium |
| 10 | **API hardening:** backend range and date validation, cache loaded model, health reflects DB state, optional API key, `/patients` Pydantic body, 503 mapping, migrations (Alembic). | FB-2 to FB-5, FB-9, FB-10 | Medium |
| 11 | **Frontend fixes:** delete race, partial-save duplicates, stale text, selection reset. | FB-6, FB-7, FB-12 | Small |
| 12 | **Quality tooling:** fix `ruff` findings; align `mypy` Python version; confirm `requirements.txt`; remove scratch files; fix Docker entrypoint. | TD-1 to TD-4, FB-11 | Small |
| 13 | **Real-data adapter (deferred by decision):** after a suitable dataset is secured. Design boundary in 7.4. | Section 7 | Deferred |

---

## 11. Recommended implementation sequence

0. **Housekeeping (hours):** items 3 and 4 above. Add `.gitignore` rules before anything else touches `data/`, commit the Phase 12 audit artefacts, merge Phases 10 and 11. No behaviour change besides CORS.
1. **Metric honesty (item 2):** small, and prevents anyone from citing the contaminated validation numbers.
2. **Fix occlusion (item 6) and pipeline honesty (item 7):** these change what the evaluation runner will measure, so do them first.
3. **Harder synthetic benchmark (item 5):** so the runner has a non-trivial task to evaluate.
4. **Unified research evaluation runner (item 1):** the core deliverable of the revised objective.
5. **Schema-driven data layer (item 8):** the preparation for real-data validation, done against the existing synthetic data so that no result changes. Validate by checking that the synthetic pipeline reproduces the same metrics.
6. **Explain endpoint and UI (item 9), API hardening (item 10), frontend fixes (item 11).**
7. **Quality tooling (item 12) and documentation refresh.**
8. **Final verification run:** full backend and frontend test suites, a complete multi-seed research run on a clean tree, and archiving of the config, code hash and data hash.
9. **Later, outside this scope:** real-data adapter (item 13).

Preserving existing working implementations: items 1, 5 and 8 are additive or parameterising changes. None requires rewriting the models, preprocessors, repository or React components.

---

## 12. Definition of done for the final project

The project can be called complete when **all** of the following hold:

**Repository**
- All phases are merged to `main`, working tree clean, no raw datasets or credentials tracked, `.gitignore` covers dataset archives.
- `README.md` and a phase plan reflect actual status.

**Methodology**
- Reported metrics are all held-out; no post-retrain number is labelled validation.
- Every research result is produced from repeated seeded splits with 95% CIs, per-model test metrics, and at least two baselines.
- Calibration (ECE, MCE, slope, intercept) is reported for the selected model; any post-hoc calibration is fitted on validation data only.
- Robustness and explanation-stability results exist for the selected model, aggregated over patients and perturbation levels.
- Occlusion and SHAP are validated against the generator's known drivers.
- At least one non-trivial synthetic benchmark variant is evaluated in addition to the baseline generator.

**Reproducibility**
- Every run emits a `RunResult` including config hash, git commit, package versions, Python version, dataset hash, split hash and seeds, and does not overwrite a previous run.
- `deterministic_torch` is enabled for the final runs, or its non-determinism is documented.

**Engineering**
- Backend and frontend test suites pass with 0 failures; the skipped explainability-on-real-artefact tests are either executed or removed; the new tests listed in section 9.2 (High) exist.
- `ruff` is clean; `mypy` runs and passes on a pinned Python version.
- CORS, validation, health, and error mapping defects fixed. Auth is either implemented or the deployment is explicitly documented as local-only.
- Docker image builds and serves the API.

**Real-data readiness**
- Feature lists, units, ID and time columns and outcome definition are configuration-driven; `DataValidator` runs in the pipeline; missing-label handling is explicit; a written adapter contract exists.
- Documentation states clearly that all current results are synthetic and that real-world validation is future work.

**Integrity of claims**
- No document, UI text, artefact or report presents synthetic results as clinical evidence. Superseded runs are marked as such.

---

## Appendix: what I could not verify

- I did not read every line of `lightgbm_model.py`, `logistic_regression.py`, `random_forest.py`, `xgboost_model.py`, `registry.py`, `train.py`, `seed.py`, or the `demo_fast`, `research_experiment` and `research_experiment_v2` configs. Findings about those files come from grep-level checks.
- Several line numbers marked *(sub-agent)* are approximate.
- I did not run the API server or browser UI; FB-1 and FB-6 are deduced from code. I did not test how Starlette routes percent-encoded `/` in patient IDs.
- I did not open `outputs/research` or `outputs/research_v2`, and did not inspect the MIMIC-IV demo contents.
- I did not run the research pipeline or any training, and did not recompute the model metrics quoted in section 5.2; they are copied from saved metadata.
- The system clock reads 2026; I cannot confirm the artefact timestamps correspond to real elapsed time.
- The count of issues in section 8 is approximate because some items are grouped.
