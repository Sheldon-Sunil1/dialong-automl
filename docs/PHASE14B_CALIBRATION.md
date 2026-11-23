# Phase 14B: post-hoc probability calibration (reliability item R1)

> Synthetic data only. Nothing here is clinical evidence.

Design: `docs/research/phase_14a_reliability_design.md` (R1, decisions in section 9, 14B notes in R1).

## What was added

| Where | Change |
|---|---|
| `dialong_automl/reliability/calibration.py` | `PlattCalibrator` (default), `IsotonicCalibrator` (sensitivity only), `IdentityCalibrator`; JSON round trip (`to_dict`, `calibrator_from_dict`); metrics `calibration_slope_intercept`, `ece_equal_mass`, `calibration_summary`. Pure NumPy, deterministic. |
| `config/schema.py` | `ReliabilityConfig`: `reliability.enabled` (default true), `reliability.calibration_method` (default `platt`). Not part of the Optuna study identity. |
| `training/pipeline.py` | Calibrator fitted on the best trial's pre-retrain validation predictions, the same partition as the threshold. It writes `calibration.json`, and adds `reliability` to metadata, run summary and `PipelineResult`. Raw vs calibrated test calibration metrics. Labelled optimistic. |
| `experiments/nested_cv.py` | Per outer fold, the calibrator is fitted on the inner-CV OOF probabilities (never the outer test fold), and a new `automl_calibrated` predictor row is added (role `calibrated_variant`, outside the Holm family). Coefficients go in `folds.json`. |
| `experiments/statistics.py` | Calibration metrics `ece`, `ece_equal_mass`, `cal_slope`, `cal_intercept` with bootstrap CIs. Post-hoc `auroc_fold_mean` (see the 13D/13E doc correction). |
| `scripts/posthoc_fold_mean_auroc.py` | Read-only re-summary of an existing run with fold-mean AUROC. |

**Unchanged:** the served `probability` and the decision `raw probability >= threshold`. The threshold is mapped
through the calibrator (`calibrated_threshold`) only for display and metrics. Because the calibrator is monotone,
predicted classes are identical.

**Defects found and fixed during 14B:**
* With constant fit input, the Platt fallback collapsed all predictions to the prevalence; in one test fold,
  AUROC dropped from 0.946 to 0.5. It is now an intercept-only shift with slope 1, which keeps ranking.
* A 1e-6 logit clip tied very small probabilities. The clip is now 1e-15, and the sigmoid is overflow-free.

## Results: `v2_standard`, 5,000 patients, seed 42

Run `outputs/research_v4/nested_cv/v2_standard_n5000_s42_phase14b` (new folder; 24 min). Same config as 13E;
the raw AutoML predictions reproduce the 13E run (pooled AUROC 0.8828). Mean over 2 repeats, 95 % patient
bootstrap CI (1,000 resamples).

| Predictor | ECE | Calibration-in-the-large | Slope | Brier | Pooled AUROC | Fold-mean AUROC |
|---|---|---|---|---|---|---|
| AutoML raw | 0.044 [0.039, 0.054] | −0.55 [−0.70, −0.40] | 0.97 | 0.077 | 0.883 | 0.894 |
| AutoML calibrated (Platt) | 0.013 [0.011, 0.024] | +0.02 [−0.12, +0.17] | 0.99 | 0.069 | 0.893 | 0.894 |
| B3 logistic, all features | 0.015 [0.012, 0.025] | 0.00 [−0.16, +0.15] | 0.90 | 0.065 | 0.901 | 0.901 |

Paired differences (raw − calibrated): ECE +0.031 [+0.019, +0.039]; Brier +0.008 [+0.006, +0.010];
fold-mean AUROC exactly 0.

**R1 acceptance criteria (design section 6):**
* Calibrated ECE is below raw ECE: **pass**.
* The calibration-in-the-large CI contains 0: **pass**.
* Fold-mean AUROC is unchanged: **pass** (difference 0).

Pooled AUROC rose by 0.010 [0.005, 0.015]. Calibration puts the separately fitted fold models on a common scale,
which removes most of the pooling loss described in the 13D/13E correction. This is not a gain in ranking ability.

**Reading.**
* Raw AutoML probabilities over-predict risk; a negative calibration-in-the-large means the predictions are too
  high. After Platt scaling, AutoML is about as well calibrated as the B3 logistic regression (ECE 0.013 vs
  0.015).
* Its Brier score is still worse than B3's (0.069 vs 0.065), because calibration does not improve ranking.
* The 14A measurement (ECE 0.071) used repeat 0 only. Here the value is the mean over both repeats, so the
  numbers differ.
* Per-fold Platt slopes ranged from 0.96 to 1.40, and no fallback was triggered. Random forest was selected in
  6 of 10 folds, GRU in 2, logistic regression in 2.

## Tests

* `tests/test_reliability_calibration.py` (21 tests) covers:
  * agreement with unpenalised sklearn logistic regression;
  * identity on already calibrated scores;
  * correction of over-prediction;
  * strict monotonicity with exact AUROC invariance, including for extreme probabilities;
  * edge cases, isotonic against sklearn plus a hand example, and the JSON round trip;
  * the metric hand examples;
  * that the pipeline fits once on the validation split, writes the artifact and keeps the prediction rule;
  * that the nested calibrator is fitted on the inner OOF and unchanged when test labels are flipped, plus the
    disable switch.
* New tests in `tests/test_nested_cv.py` check the fold-mean AUROC: scale shift, absent without fold ids,
  invariance under calibration.
* Full backend suite: **726 passed, 8 skipped**.
* Mutation check: refitting the nested calibrator on outer-test labels made two tests fail (calibrator fitted on
  inner OOF; flipped test labels). The mutation was reverted.
* Ruff: no new findings in modified tracked files compared with HEAD. The only findings in new files are
  `X` naming (N806), matching the existing repo convention.

## Limitations

* One synthetic dataset; calibration quality on other populations is unknown.
* Single-split pipeline calibration is fitted on one validation split (about 20 % of patients) and is optimistic.
  Nested CV is the reference evaluation.
* Not yet exposed through `predict.py`, the API or the UI. Per the design, that comes with the `ReliabilityReport`
  in 14D/14E.
