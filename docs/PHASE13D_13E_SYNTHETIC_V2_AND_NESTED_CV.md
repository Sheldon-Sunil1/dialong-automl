# Phase 13D and 13E: synthetic scenario v2 and the nested-CV evaluation runner

> All data and results in this document are **synthetic**. They are not clinical
> evidence and say nothing about real-world predictive performance.

Plan: `docs/research/phase_13a_synthetic_validity_and_protocol.md` (sections 5 and 7).

## Phase 13D: synthetic scenarios

Module: `dialong_automl/data/synthetic_v2.py`, entry point `generate_scenario(scenario, n_patients, seed)`.
The original generator (`dialong_automl/data/synthetic_generator.py`) is **unchanged**.

| Scenario | What it is |
|---|---|
| `legacy_v1` | The original generator, called unchanged. Seed 42 / 1,000 patients reproduces the historical `data/synthetic/diabetes_progression_full.csv` exactly (SHA-256 `277894ab...`; pinned in a regression test). No ground truth. |
| `v2_standard` | New continuous-time generator. All parameters in the frozen dataclass `V2Params` (hash recorded in metadata). Parameters were fixed from data properties only, before any model comparison. |

### What `v2_standard` implements (Phase 13A items)

| Item | Implementation |
|---|---|
| G1 size / prevalence | Default 5,000 patients. Measured eligible prevalence about 10-12 % (test: 6-18 % across seeds). |
| G2 ground truth | `truth_patients` (subgroup, true rate, true HbA1c at index, oracle probability, ...) and `truth_visits` (noise-free values, measured flags). Returned separately; `save()` writes them to a `truth/` subfolder that no loader reads. |
| G3 calendar time | HbA1c is a function of elapsed days. Gaps are log-normal (median 150 days, clipped 30-540), shortened by 30 % after a measured HbA1c >= 6.0 (past only). Up to 730 days of follow-up, up to 8 future visits; 12 % drop out after the index visit. |
| G4 heterogeneity | Continuous patient-specific rate from an overlapping mixture: stable, slow, fast, improver; mixture weights depend on a latent risk score. Metformin lowers the rate. |
| G5 noise | Biological variability as an Ornstein-Uhlenbeck process (SD 0.15 %, 0.5-year memory) plus assay error (SD 0.10 %); HbA1c reported to 0.1 %. |
| G6 outcome uncertainty | A diagnosis is recorded from one measured HbA1c >= 6.5 % or one fasting glucose >= 126 mg/dL. Both are noisy, so a single reading can cross by chance. |
| G7 missingness | `mcar`, `mar` (default; HbA1c measured more often after a previous value >= 5.7), `mnar` (depends on the current true value). Applies to observation **and** future visits. |
| G8 coupled glucose | Fasting glucose derived from the latent glycaemic level (ADAG relation) plus noise. |
| G9 variable histories | 2-7 observation visits, so patients with too few visits are excluded and 3-visit (padded) sequences reach the models. |
| Not done | G10 nonlinear progression, G11 time-varying treatment, G12 phenotypes (Phase 13A Tier 2/3). |

### Oracle ceiling

`oracle_prob` = P(label = 1 | latent state at the index visit, outcome ascertained), estimated by Monte Carlo
(default 32 draws) over the future visit, measurement and diagnosis process. It uses information no model
has (true rate and biological state), so it is an **approximate upper bound**, not a model.

### Leakage design (tested)

* Each patient has three independent random streams (observation, future, oracle). Observation rows are
  generated first and cannot depend on the future. Test: changing future-only parameters leaves every
  observation row identical; a mutation that made one observation value depend on the label was caught.
* Scheduling and missingness use only the past (MNAR: the current true value), never the future.
* Truth columns never appear in the model-facing table, the loaded CSV, or the cohort's observation data.

### Cohort change: unascertained outcomes (audit item ML-8 / I-7)

`CohortConfig.require_outcome_measurement` (config `cohort.require_outcome_measurement`, default **false**).
When true, a patient whose horizon has neither a measured HbA1c nor a recorded diagnosis is excluded
(`audit.excluded_unmeasured_outcome`) instead of being labelled 0. The default keeps the original behaviour,
so legacy runs are unchanged. Note: adding this field changes the Optuna study identity hash of every config
(the identity includes the cohort section), so previously persisted studies cannot be resumed.

Not changed: a missing observation-window HbA1c still cannot trigger the baseline-diabetes exclusion.

## Phase 13E: unified evaluation runner

Module: `dialong_automl/experiments/nested_cv.py` (`NestedCVRunner`), statistics in
`dialong_automl/experiments/statistics.py`, CLI `scripts/run_nested_cv.py`,
config `configs/research_v4_nested_cv.yaml` (section `nested_cv`).

### Protocol (per repeat r, outer fold k)

1. Outer stratified K-fold over eligible patients. The test fold (visits and labels) is held in
   `HeldOutFold`; the selection code receives only outer-training patients and the fold is read exactly once
   (count recorded; the run fails otherwise).
2. Inner stratified K-fold of the outer-training patients. Representations, vocabulary and preprocessors are
   fitted on each inner-training set only.
3. Optuna (third-party) searches the existing space; the objective is the **mean inner-fold validation AUROC**
   (`CVAutoMLOptimizer`). Studies are in-memory and unique per (repeat, fold). Pruning is inactive because
   intermediate values are not reported across inner folds.
4. Threshold: `training.threshold_strategy` on the best trial's inner out-of-fold probabilities.
5. Final fit on all outer-training patients with the selected configuration (GRU: internal stratified holdout
   for early stopping, as in `TrainingPipeline`), then one prediction of the test fold.
6. Baselines B0-B3: thresholds from their own inner out-of-fold scores, refit on outer-train, test once.
7. Optional oracle row (v2 only) on the same patients, reported as a ceiling.

### Statistics

* Pooled outer-test predictions: every eligible patient exactly once per repeat; point estimate = mean over repeats.
* Patient-level (cluster) bootstrap; the same resample is used for every predictor, so comparisons are paired.
* Comparisons are `automl_selected - comparator` for AUROC, AUPRC and Brier, with bootstrap CIs and two-sided
  bootstrap p-values. Primary comparison (pre-specified): AUROC vs `b3_logreg_all`, unadjusted. The other
  baseline AUROC comparisons are Holm-adjusted. The oracle is excluded from the Holm family (`role: ceiling`).
* Threshold metrics (F1, recall, precision) use each fold's own threshold.

Caveats: the CIs resample patients of **one** fixed synthetic dataset and do not cover variation across datasets;
pooling scores across folds mixes fold-specific models (standard for pooled out-of-fold AUROC, but a fold-level
summary is also available in `folds.json`); bootstrap p-values are approximate.

### Outputs and provenance

`<outputs_dir>/nested_cv/<run>/` (created with `exist_ok=False`, never overwritten): `run_result.json`
(`RunResult`: config hash, seeds, dataset id, split hash, flat metrics; metadata with git commit, dirty flag,
branch, Python, platform, package versions, dataset SHA-256, generator parameter hash, cohort audit,
selected models per fold, measured test access per fold, comparisons, disclaimer), `predictions.csv`,
`summary.csv`, `comparisons.csv`, `folds.json`, `resolved_config.json`, `generation_metadata.json`.

### Commands (PowerShell, repo root)

```powershell
# Main run: v2_standard, 5,000 patients, 2 x 5 outer x 3 inner folds, 20 trials per fold (~25 min on CPU)
.\.venv\Scripts\python.exe scripts/run_nested_cv.py --config configs/research_v4_nested_cv.yaml --run-name <new-name>

# Corrected legacy reference (the historical dataset), ~7 min
.\.venv\Scripts\python.exe scripts/run_nested_cv.py --config configs/research_v4_nested_cv.yaml `
    --scenario legacy_v1 --patients 1000 --dataset-seed 42 --run-name <new-name>
```

Both write new folders under `outputs/research_v4/nested_cv/` and do not read or write anything under `data/`.

## Results of the first runs (synthetic, 22 Nov 2026)

Both runs: `configs/research_v4_nested_cv.yaml`, all five model families, 2 repeats x 5 outer x 3 inner folds,
20 Optuna trials per outer fold, `validation_f1` thresholds, 1,000 bootstrap resamples. Every outer test fold
was read exactly once; no trial failed. Provenance caveat: both runs record `git_dirty: true` (commit
`22b6e95` plus the then-uncommitted Phase 13D/13E code), and GRU training is not bit-reproducible
(`deterministic_torch: false`).

### `v2_standard`, 5,000 patients, seed 42 (`outputs/research_v4/nested_cv/v2_standard_n5000_s42`)

Eligible 2,778 patients, 286 positive (10.3 %); exclusions: 485 prior diabetes, 809 too few visits,
536 no follow-up, 392 unmeasured outcome. Selected per fold: random forest 7, GRU 2, logistic regression 1.
Runtime 23 min.

| Predictor | AUROC [95 % CI] | AUPRC | Brier |
|---|---|---|---|
| AutoML selected | 0.883 [0.867, 0.898] | 0.427 | 0.077 |
| B0 prevalence | 0.498 [0.474, 0.519] | 0.103 | 0.092 |
| B1 last HbA1c | 0.897 [0.879, 0.914] | 0.494 | n/a |
| B2 trend extrapolation | 0.852 [0.831, 0.873] | 0.392 | n/a |
| **B3 logistic regression, all aggregates** | **0.901 [0.886, 0.916]** | 0.520 | 0.065 |
| B3 HbA1c only | 0.896 [0.878, 0.914] | 0.519 | 0.065 |
| B3 no HbA1c | 0.875 [0.857, 0.893] | 0.431 | 0.072 |
| Oracle (latent truth, ceiling) | 0.938 [0.923, 0.953] | 0.731 | 0.048 |

Paired AUROC differences (AutoML minus comparator): **primary vs B3 all: -0.018 [-0.027, -0.010], p = 0.002**;
vs B1 -0.014 (Holm p = 0.066); vs B3 HbA1c-only -0.013 (Holm p = 0.066); vs B3 no-HbA1c +0.007 (Holm p = 0.28);
vs B2 +0.031 (Holm p = 0.010); vs oracle -0.055 (ceiling, not adjusted).
Mean best inner-CV AUROC across folds was 0.895 against 0.883 on the outer folds (selection optimism about 0.01).

### `legacy_v1`, 1,000 patients, seed 42: the historical dataset (`.../legacy_v1_n1000_s42`)

Eligible 786, 46 positive (9-10 per outer test fold). AutoML 0.957 [0.912, 0.983]; B3 all 0.968 [0.953, 0.980];
B3 HbA1c-only 0.981 [0.969, 0.991]; B2 0.976; B1 0.972. Primary AutoML - B3 all: -0.011 [-0.051, +0.013], p = 0.65.
AutoML - B3 HbA1c-only: -0.024 [-0.060, -0.004], Holm p = 0.012. This reproduces the Phase 13A finding with the
corrected protocol.

### What these results show, and what they do not

* On both scenarios the AutoML pipeline does **not** outperform a fixed, untuned logistic regression on simple
  aggregates. On `v2_standard` it is significantly worse on the pre-specified primary comparison.
  **Correction (Phase 14B):** about half of that gap comes from pooling outer-fold scores, not from ranking
  within folds. See "Post-hoc methodological correction" below.
* `v2_standard` is harder than `legacy_v1` (last-HbA1c AUROC 0.90 vs 0.97) and leaves measurable headroom
  (oracle 0.94), so a failure to beat the baseline is informative rather than a ceiling effect.
* The GRU was selected in only 2 of 10 folds; there is no evidence here that sequence modelling helps.
* These are results on one synthetic dataset per scenario. They establish nothing about clinical performance.
* Plausible reasons (not tested): Optuna selects on a noisy inner-CV AUROC among 5 families in 20 trials;
  tree models on ~1,800 training patients with ~190 positives overfit; the wide features already encode the
  signal a linear model needs. Investigating this is future work, not part of Phase 13E.

## Post-hoc methodological correction: pooled vs fold-averaged AUROC (added in Phase 14B)

This analysis was added **after** the results above were seen. It does not replace the pre-specified primary
endpoint (pooled out-of-fold AUROC, AutoML vs B3 all), which stands as reported. It explains part of the gap.

**Mechanism.** Pooled AUROC ranks outer-test patients from all 5 folds together, although each fold's scores come
from a separately selected and fitted model. If the score scale differs between folds (for example a different
model family, or a different degree of overconfidence), patients are ranked against other folds' scales, and
AUROC drops even when every fold ranks its own patients well. The fixed logistic-regression baselines have
nearly the same scale in every fold. The AutoML model selected different families in different folds. So pooling
penalises AutoML more than the baselines.

**Metric.** `auroc_fold_mean` is the mean of the 5 per-fold AUROCs, averaged over the 2 repeats. Uncertainty
comes from the same patient-level paired bootstrap (1,000 resamples, seed 2026), with each patient keeping
its fold. It is reported and compared but is never the primary comparison and is not in the Holm family.
Implemented in `dialong_automl/experiments/statistics.py` (`fold_ids` argument of `bootstrap_summary`).
Computed read-only from the existing `predictions.csv` with `scripts/posthoc_fold_mean_auroc.py`; output in
`outputs/research_v4/posthoc_fold_mean/`. The recomputed pooled AUROCs match the original `summary.csv` to 1e-16.

**`v2_standard`, 5,000 patients** (AUROC, 95 % bootstrap CI)

| Predictor | Pooled | Fold mean | Pooling loss |
|---|---|---|---|
| AutoML selected | 0.883 [0.867, 0.898] | 0.894 [0.878, 0.909] | 0.011 |
| B3 logistic, all features | 0.901 [0.886, 0.916] | 0.901 [0.886, 0.917] | 0.000 |
| B1 last HbA1c | 0.897 | 0.897 | 0.000 |
| B3 HbA1c only | 0.896 | 0.897 | 0.001 |
| Oracle (ceiling) | 0.938 | 0.938 | 0.000 |

| AutoML minus | Pooled difference | Fold-mean difference |
|---|---|---|
| B3 all (primary) | −0.018 [−0.027, −0.010], p = 0.002 | −0.008 [−0.016, +0.000], p = 0.07 |
| B1 last HbA1c | −0.014 [−0.025, −0.003] | −0.004 [−0.014, +0.007] |
| B3 HbA1c only | −0.013 [−0.024, −0.002] | −0.003 [−0.013, +0.007] |
| B3 no HbA1c | +0.007 [−0.005, +0.022] | +0.018 [+0.006, +0.032] |

**`legacy_v1`, 1,000 patients:** AutoML pooled 0.957 vs fold mean 0.969. Against B3 all, the fold-mean
difference is +0.000 [−0.022, +0.018]. HbA1c-only logistic regression is still higher: −0.014 [−0.033, −0.001].

**Corrected reading.**
* The pre-specified result stands: on pooled AUROC, AutoML is worse than B3 all on `v2_standard`.
* About 0.011 of the 0.018 gap comes from AutoML's fold-to-fold scale variation. Judged within folds, AutoML
  ranks patients about as well as the simple baselines: −0.008 vs B3 all, with a CI that touches 0. It is not
  better than them on either metric. The earlier claim that AutoML is "significantly worse" therefore says more
  about how consistent its score scale is across folds than about how well it ranks patients.
* Scale inconsistency matters in practice. A served score is read on an absolute scale, so this finding is
  part of the motivation for post-hoc calibration (Phase 14B, R1).
* Calibration does not change `auroc_fold_mean`: each fold has a monotone calibrator. It can change pooled
  AUROC, because it puts the folds on a more common scale. This is expected, not a contradiction.
* All results remain synthetic, from one dataset per scenario. Not clinical evidence.
