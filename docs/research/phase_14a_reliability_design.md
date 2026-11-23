# Phase 14A: Prediction Reliability, Design

Date: 2026-11-22 · Status: **design only (read-only phase)**. No production code was changed.
Repository state: branch `phase/13-synthetic-validity-baselines`, HEAD `22b6e95`, with the Phase 13D/13E
work uncommitted.

> All data and results are **synthetic**. Nothing here is clinical evidence. The feature described below
> describes how far the *model's own output* can be trusted. It is not a medical judgement.

**Evidence labels**
- **[M]** measured for this design (read-only diagnostics on the existing run
  `outputs/research_v4/nested_cv/v2_standard_n5000_s42`, repeat 0, and an in-memory regeneration of its dataset).
- **[R]** read from code.
- **[P]** proposal.

---

## 1. Goal and scope

Every prediction should come with an answer to three questions:

1. **Can the percentage be taken at face value?** This is calibration: "among patients given 20 %, about 20 % develop onset".
2. **How sure is the model about this patient?** This is per-patient uncertainty: a variability range and a
   statement when the model cannot decide.
3. **Is this patient's input good enough?** This covers data-quality and out-of-distribution checks.

These combine into a **reliability level** (High / Moderate / Low) with the reasons listed.

**Audience.** The audience is a researcher or clinician reviewing a synthetic demonstration. Patient-facing
wording is out of scope until real-data validation exists (section 7).

**Non-goals.**
- No change to the dataset generator, the cohort or model selection.
- No real-world data.
- No claim of clinical reliability.

---

## 2. Current state

### 2.1 What exists [R]

| Item | Status |
|---|---|
| `evaluation/calibration.py` | ECE, MCE and reliability bins, implemented and tested, **not used anywhere**. No calibration slope or intercept, and no recalibration. |
| `predict.py` / `POST /predict` / `POST /patients/{id}/predict` | Return `probability`, `prediction`, `threshold`, `model`, `representation`, `n_visits`, `disclaimer`. No uncertainty. |
| `frontend/src/components/PredictionResult.tsx` | Shows the probability to 1 decimal, a bar, the class and the model name. No reliability information. |
| Nested-CV runner (13E) | Produces inner-CV out-of-fold (OOF) probabilities for every outer-training patient. These are the natural leak-free data for fitting calibration and conformal sets. |
| v2 ground truth (13D) | The oracle probability allows reliability claims to be checked against the latent truth. |

### 2.2 Measured evidence [M]

Pooled outer-test predictions, repeat 0, 2,778 patients, observed onset rate 10.3 %.

| Predictor | Mean predicted | ECE (10 equal-width bins) | Calibration slope | Calibration-in-the-large |
|---|---|---|---|---|
| AutoML selected | **17.3 %** | **0.071** | 0.92 | **-0.88** |
| B3 logistic regression (all) | 10.3 % | 0.019 | 0.91 | 0.00 |
| B3 HbA1c-only | 10.3 % | 0.011 | 1.00 | 0.00 |
| Oracle (latent truth) | 10.2 % | 0.012 | 0.51 | 0.02 |

- **The AutoML model over-predicts.** Its displayed percentages are on average about 1.7 times the observed
  rate. A likely cause is that the search space allows `class_weight="balanced"`, which inflates
  probabilities. This was not tested. Calibration is therefore a correctness issue for the UI, not a
  cosmetic one.
- **The oracle is miscalibrated in slope (0.51)**, because a 32-draw Monte Carlo estimate is coarse and
  extreme. Use it only as a discrimination ceiling and for ranking checks, never as a calibration
  reference.
- **Model instability is large for some patients.** The two repeats train different models on different
  folds. Between them, the absolute difference in a patient's AutoML probability has a median of 0.03, a
  90th percentile of 0.26 and a 99th percentile of 0.44. A per-patient variability measure therefore
  carries information.
- **Disagreement between AutoML and B3 tracks error strongly.** By quartile of |p_AutoML − p_B3|, the
  AutoML Brier score is 0.011, 0.033, 0.077 and 0.207. This is **confounded with the risk level**:
  high-risk patients have higher Brier scores regardless. Any reliability signal must therefore be judged
  *within* risk strata (section 6).
- **Input quality in the v2 cohort.** Of 2,778 patients:
  - 46 (1.7 %) have no HbA1c in the observation window;
  - 272 have exactly one;
  - 566 have 3 instead of 4 visits.

  Days from the last measured HbA1c to the index visit: median 0, 75th percentile 65, 90th percentile 216,
  95th percentile 322.

---

## 3. Components [P]

Every learned quantity is fitted on data the evaluated patients are not part of. The table states the
partition explicitly.

| Fitted in | Nested CV (13E) | Single-split `TrainingPipeline` |
|---|---|---|
| Calibrator, conformal quantiles, width cut-offs | inner-CV OOF predictions of the selected configuration | validation-split predictions of the best trial (pre-retrain), the same partition as the threshold |
| OOD reference statistics | outer-training patients | train split (refit on train + validation with the model) |
| Ensemble members | bootstrap resamples of outer-training patients | bootstrap resamples of train + validation |
| Evaluated on | outer test folds only | test split only |

The single-split caveat from 13B.1 still applies: the validation split is also used for selection, so
reliability estimates fitted there are optimistic. The nested runner is the reference evaluation.

### R1 Calibrated probability

- **Method:** Platt scaling, a logistic regression of y on logit(p), is the default. Isotonic regression is
  an option for sensitivity analysis only, because with about 190 positives per outer-training set it is
  prone to step artefacts and ties.
- **Order of operations:** fit the calibrator on OOF probabilities, then choose the decision threshold on
  the *calibrated* OOF probabilities. Platt is strictly monotone, so AUROC is unchanged. Isotonic may
  create ties and lower AUROC slightly; report this if used.
  - **Changed in 14B:** the threshold is *not* re-selected on calibrated probabilities. The decision stays
    `raw probability >= threshold`, chosen as before. The threshold is mapped through the calibrator
    (`calibrated_threshold`) for display and for thresholded metrics on calibrated scores. Because the
    calibrator is monotone, this decision is identical to `calibrated >= calibrated_threshold`, and served
    predictions do not change. Re-selecting the threshold would have changed them.
  - **Clarified in 14B:** AUROC invariance holds for scores from one model with one calibrator, i.e. within a
    fold. Pooled out-of-fold AUROC can change, because each fold has its own calibrator. The fold-averaged
    AUROC (`auroc_fold_mean`) is exactly invariant.
  - **Edge cases in 14B:** a non-positive fitted slope falls back to the identity. Constant fit input falls back
    to an intercept-only shift with slope 1, so ranking is kept and nothing collapses to one value. The logit
    clip is 1e-15: a 1e-6 clip tied very small probabilities and changed AUROC in a test run.
- **Output:** `calibrated_probability`, plus the calibrator's coefficients in the metadata.
- **Persisted as** `calibration.json` (method, coefficients, fit partition, n, n_positive). This is plain
  JSON, not pickle.
- **Evaluation:**
  - ECE with 10 equal-width bins (existing function) and equal-mass bins (new);
  - calibration slope and calibration-in-the-large with bootstrap CIs;
  - reliability diagram data.

### R2 Model-variability range (ensemble)

- **Method:** refit the selected configuration on `M` patient-level bootstrap resamples (default
  `M = 10`). Apply the same R1 calibrator to each member's probability.
- **Output:** a range from the 10th to the 90th percentile of the member probabilities, plus its width.
  The point estimate stays the calibrated prediction of the main model.
- **Wording:** "model variability range". It is **not** a confidence interval for the patient's true risk.
- **Cost:** `M` extra fits per final model. For the GRU this is roughly `M` times the retrain time.
  `M` is configurable, and `0` disables R2.

### R3 Conformal prediction set

- **Method:** split conformal classification with **class-conditional (Mondrian)** calibration. At about
  10 % prevalence, marginal coverage could be met by predicting "no onset" almost everywhere. Per-class
  coverage forces the method to cover positives too.
- **Details:**
  - nonconformity score s = 1 − p̂(true class), on calibrated probabilities;
  - per class c, q̂_c is the ⌈(n_c + 1)(1 − α)⌉ / n_c empirical quantile of the scores of OOF patients of
    class c;
  - a class is in the set if its nonconformity score is ≤ q̂_c;
  - default α = 0.10.
- **Output:** a set, `{no onset}`, `{onset}`, `{no onset, onset}` (undecided) or `{}` (rare: the input
  is unlike any calibration case).
- **Guarantee:** with exchangeable calibration and test patients, coverage is at least 1 − α per class.
  In nested CV the OOF scores come from inner-fold models, not the final model, so the guarantee is
  **approximate** (a cross-conformal / CV+ style approximation). We must verify it empirically
  (section 6) and say so in the output.
- **Persisted as** `conformal.json` (α, mode, q̂ per class, n per class).

### R4 Input-quality flags

Rules are fixed in advance and computed from the submitted visits only. Thresholds come from section 2.2.

| Flag | Rule | Severity |
|---|---|---|
| `no_hba1c` | no measured HbA1c in the observation window | severe |
| `single_hba1c` | exactly one measured HbA1c (no trend available) | moderate |
| `stale_hba1c` | last HbA1c more than 365 days before the latest visit | severe |
| `old_hba1c` | last HbA1c 181 to 365 days before the latest visit (about the 90th percentile in v2) | moderate |
| `short_history` | fewer visits than the model's observation window | info |
| `implausible_value` | any value outside the schema bounds (the frontend `NUMERIC_RANGES`, to be mirrored in the backend) | severe |
| `no_glucose` | no fasting glucose in the window | moderate |

### R5 Out-of-distribution score

- **Method:** compute the Mahalanobis distance on the standardised numeric aggregate features used by B3
  (`b3_columns("all")`, numeric part). Use a shrinkage covariance (Ledoit-Wolf) fitted on the training
  patients, and compute the same features whatever model is served.
- **Output:** the distance percentile relative to the training patients' own distances.
  - `ood_far` (severe): above the 99th percentile.
  - `ood_unusual` (moderate): above the 95th percentile.
- **Persisted as** `ood.json` (feature list, mean, precision matrix, reference percentiles).

### R6 Reliability level

The rules are fixed before any evaluation. The width cut-offs are percentiles of the R2 widths on the
fit partition, so they are a procedure and not tuned numbers.

| Level | Rule |
|---|---|
| **Low** | any severe flag (R4/R5); or the conformal set is undecided or empty (R3); or the R2 width exceeds the 95th percentile of fit-partition widths |
| **Moderate** | not Low, and any moderate flag; or the R2 width exceeds the 80th percentile |
| **High** | otherwise |

The output lists every triggered reason in plain words. Info-level flags are shown but never change the
level.

### Optional R7: disagreement with the baseline

The measured signal is strong (section 2.2) but confounded with risk. **Not part of R6 by default.** Evaluate
it in 14F as a candidate, and add it only if it improves the within-stratum separation.

---

## 4. Software design [P]

### 4.1 New package `dialong_automl/reliability/`

All additive. No existing module is rewritten.

| Module | Responsibility |
|---|---|
| `calibration.py` | `PlattCalibrator` / `IsotonicCalibrator`: `fit(p, y)`, `transform(p)`, `to_dict()` / `from_dict()`. Also calibration slope, intercept and equal-mass ECE. |
| `conformal.py` | `ClassConditionalConformal`: `fit(p_cal, y_cal, alpha)`, `predict_set(p)`, `to_dict()` / `from_dict()`. |
| `ensemble.py` | Builds `M` bootstrap members of a configuration; returns the calibrated member probabilities and the percentile range. |
| `quality.py` | R4 rules on a visit DataFrame; returns flags with severity and message. |
| `ood.py` | `MahalanobisOOD`: `fit(wide_X)`, `score(wide_X)`, `to_dict()` / `from_dict()`. |
| `report.py` | `ReliabilityReport` dataclass and the R6 rules, `to_dict()`. |

### 4.2 Integration points

| Where | Change |
|---|---|
| `config/schema.py` | New `ReliabilityConfig`. It is **not** part of the Optuna study identity. Defaults: `enabled: true`, `calibration_method: "platt"`, `conformal_alpha: 0.10`, `conformal_mode: "class_conditional"`, `ensemble_members: 10`, `ood_percentiles: [95, 99]`, `hba1c_age_days: [180, 365]`, `width_percentiles: [80, 95]`. |
| `experiments/nested_cv.py` | Per outer fold: fit R1 and R3 on the inner OOF, R5 on outer-train, R2 members on outer-train. Score the test fold once and add reliability columns to `predictions.csv`. Add reliability metrics to the summary and the `RunResult`. |
| `training/pipeline.py` | Optional block after threshold selection. Fit on the validation predictions, write `calibration.json`, `conformal.json`, `ood.json`, the members and a `reliability` metadata section. |
| `predict.py` | If the reliability artifacts exist, return a `reliability` object; if not, return `null`. Existing fields stay **unchanged**, so old artifacts keep working. The served `probability` stays the raw model output. The new `calibrated_probability` is shown in the UI. |
| `api/schemas.py`, `api/app.py`, `api/patients.py` | Optional `reliability: ReliabilityInfo \| None` on `PredictResponse` and `SelectedPredictResponse`. |
| `frontend` | `types.ts` gains `ReliabilityInfo`. A new `ReliabilityPanel.tsx` inside `PredictionResult` replaces the 1-decimal figure with a rounded calibrated percentage and shows a range bar, the level badge, the conformal statement and the reasons. |

**Decision for review:** keep `probability` raw for backward compatibility, or replace it with the
calibrated value. I propose keeping it raw and adding `calibrated_probability`, so no existing consumer
silently changes meaning.

### 4.3 Example API payload

```json
"reliability": {
  "calibrated_probability": 0.22,
  "variability_range": [0.15, 0.31],
  "conformal_set": ["no_onset", "onset"],
  "conformal_alpha": 0.10,
  "level": "Low",
  "reasons": [
    {"code": "conformal_undecided", "severity": "severe", "message": "The model cannot decide between onset and no onset at 90 % confidence."},
    {"code": "old_hba1c", "severity": "moderate", "message": "Latest HbA1c is 214 days older than the latest visit."}
  ],
  "ood_percentile": 71.0,
  "method": {"calibration": "platt", "fit_partition": "validation split (pre-retrain)", "ensemble_members": 10},
  "disclaimer": "Reliability describes the model output on synthetic training data, not clinical certainty."
}
```

---

## 5. Leakage and integrity rules [P]

1. No reliability parameter is fitted on the patients being evaluated: no calibrator, no conformal
   quantile, no OOD statistic, no width cut-off. In nested CV the outer test fold is read once, as in 13E.
2. All rules and cut-offs (R4 thresholds, R6 logic, α, `M`, percentiles) are fixed in this document before
   any reliability result is seen. Changing them later means a new, documented version.
3. The oracle is used only for evaluation checks (section 6), never for fitting.
4. Reliability artifacts are JSON where possible. Ensemble members reuse the existing pickle or torch
   formats under the same trusted-artifact assumption as today.

---

## 6. Evaluation plan (Phase 14F) and acceptance criteria [P]

All of this is computed on outer test folds of the nested runner, for `v2_standard` (primary) and
`legacy_v1`. Results are reported with patient-level bootstrap CIs. **Failure to meet a criterion is
reported, not hidden.**

| Claim | Metric | Pass criterion |
|---|---|---|
| R1 improves calibration | ECE (equal-width and equal-mass), slope, calibration-in-the-large: raw vs calibrated | calibrated ECE < raw ECE, and the calibration-in-the-large CI contains 0. Fold-mean AUROC is unchanged for Platt (to 1e-12). Pooled AUROC may change, and is reported. |
| R3 coverage holds | empirical per-class coverage on outer test folds | ≥ 1 − α − 0.03 for both classes; set-size distribution reported |
| R6 separates reliability | Brier score and misclassification rate by level | Low > High with a bootstrap CI excluding 0, **within** risk-decile strata (stratified difference) |
| R2 ranks uncertainty | Spearman correlation of range width with \|calibrated p − oracle p\| (v2 only) | positive, CI excluding 0 |
| Selective prediction | risk–coverage curve: error when abstaining on Low, then Moderate | monotone decrease; reported |
| R7 (optional) | same within-stratum test | included in R6 only if it improves separation |

---

## 7. Presentation rules [P]

- **Wording:**
  - "Estimated risk 22 %" instead of "probability 21.8 %";
  - "model variability range 15–31 %" instead of "confidence interval";
  - "the model is undecided" instead of a probability statement.
- **Never imply diagnosis or certainty.** Keep the synthetic-data disclaimer next to the reliability
  panel.
- **Level badges** use text and colour, not colour alone (accessibility). The reasons are always shown
  for Moderate and Low.
- **Rounding:** whole percent, or "< 1 %". Showing more precision than the variability range supports
  would contradict the feature.
- **Patient-facing use is out of scope.** It would need real-data validation and a separate review.

---

## 8. Implementation plan

Each phase is reviewable and stops for approval.

| Phase | Scope | Key tests | Done when |
|---|---|---|---|
| 14B | R1 module; nested runner and pipeline integration; calibration metrics | hand-computed Platt fit; monotonicity / AUROC invariance; calibrator fitted only on OOF (a label-flip test on the outer fold changes nothing) | tests pass; calibration metrics appear in run outputs |
| 14C | R2 ensemble, R3 conformal | conformal coverage on simulated exchangeable data within tolerance; per-class quantile hand example; ensemble deterministic by seed | tests pass; coverage measured in a small nested run |
| 14D | R4 flags, R5 OOD, R6 levels, `ReliabilityReport` | each flag on hand-built visits; OOD on an injected outlier; level-rule truth table | tests pass |
| 14E | `predict.py`, API schemas and the React panel | old artifacts still predict (`reliability: null`); API contract tests; vitest for the panel and wording | backend and frontend suites pass |
| 14F | full nested runs (v2, legacy) and a report against section 6 | — | report written with every criterion marked pass or fail |

Estimated extra runtime for the full v2 nested run with `M = 10`: about 1.5 to 3 times the current
23 minutes, depending on how often the GRU is selected. Not measured yet.

---

## 9. Decisions (approved 2026-11-22; project owner delegated the choice to the recommendation)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Keep `probability` raw; add `calibrated_probability`.** The UI shows the calibrated value. | No existing consumer (CLI, tests, old artifacts, API clients) silently changes meaning. |
| 2 | **Ensemble `M = 10`.** `M = 0` stays available to disable R2. | Enough members for a 10th-90th percentile range at acceptable cost. Revisit only if 14F shows unstable ranges. |
| 3 | **α = 0.10 (90 % conformal sets).** 14F also reports α = 0.20 as a sensitivity analysis. The R6 rule is not changed by that result. | 90 % is the conventional level. A high "undecided" rate is an honest finding, not a reason to weaken the guarantee. |
| 4 | **Demo model: the AutoML-selected model**, with reliability built model-agnostically. 14F also computes reliability for B3 and reports both. | The framework is the project's subject. The UI already names the served model, and calibration (R1) fixes the measured over-prediction. Serving B3 remains a one-line config change if preferred later. |
| 5 | **Fit reliability in the single-split `TrainingPipeline` too**, labelled optimistic in metadata and in the API `method.fit_partition`. Nested CV is the reference evaluation. | The API serves single-split artifacts, so the served model needs a fitted calibrator, conformal quantiles and OOD statistics. Without them the feature could not be shown. |

---

## 10. Risks

- **Conformal sets may be mostly "undecided".** At 10 % prevalence with an AUROC near 0.88 this is
  plausible, which would make the level "Low" for many patients. That is honest, but could look unhelpful.
  14C must report the set-size distribution before 14D fixes the rule. The rule itself is not tuned to the
  result: if most sets are undecided, we report it.
- **Uninformative levels.** Reliability levels may separate error mainly through risk level (the confound
  in section 2.2). The stratified test in section 6 exists to detect this.
- **GRU ensemble cost.**
- **Generalisation.** Everything is validated on synthetic data only. Real-data behaviour is unknown and
  must not be claimed.
