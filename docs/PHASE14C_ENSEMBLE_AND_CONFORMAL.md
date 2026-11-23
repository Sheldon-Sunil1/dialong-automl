# Phase 14C: model-variability range (R2) and conformal prediction sets (R3)

> Synthetic data only. Nothing here is clinical evidence.

Design: `docs/research/phase_14a_reliability_design.md`, items R2 and R3.

## What was added

| Where | Change |
|---|---|
| `dialong_automl/reliability/conformal.py` | `MondrianConformal`: class-conditional split conformal. The score is 1 − calibrated p(candidate label); q̂_c is the ⌈(n_c+1)(1−α)⌉-th smallest score of class c, or +∞ (class always included) if that index exceeds n_c. JSON round trip; `coverage_summary`. |
| `dialong_automl/reliability/ensemble.py` | `member_subsamples` (stratified, without replacement, seeded) and `variability_range` (10th–90th percentile of member probabilities). |
| `config/schema.py` | `reliability.conformal_alpha` (0.10), `reliability.ensemble_members` (10; 0 disables; 1 rejected), `reliability.ensemble_subsample_fraction` (0.8). |
| `experiments/nested_cv.py` | Per outer fold, conformal sets are fitted on the calibrated inner-CV OOF probabilities. Members are the selected configuration refitted on subsamples of outer-train and calibrated with the fold's calibrator. The `automl_calibrated` rows gain `set_no_onset`, `set_onset`, `range_low`, `range_high` and `range_width`. Per-fold `conformal` and `ensemble` records. Run-level `reliability_summary.json`, also stored in `RunResult.metadata`. The test fold is still read once. |
| `training/pipeline.py` | Conformal sets fitted on the calibrated validation predictions (labelled optimistic); `conformal.json`; test coverage in `reliability`. |

**Deviations from the design**
* Members use 80 % subsamples without replacement, not bootstrap resamples. Representations and preprocessors are
  keyed by patient id, so duplicated patients cannot be represented without renaming them.
* Ensemble members for the single-split pipeline (the served model) are deferred to 14E, where `predict.py`
  needs them. 14C builds them in nested CV only.

## Result: `v2_standard`, 2,000 patients, seed 42 (small run)

Run `outputs/research_v4/nested_cv/v2_standard_n2000_s42_phase14c` (22 min, M = 10). It has 1,105 eligible
patients, about 115 positive; 2 repeats × 5 outer folds.

| Quantity | Value | Criterion (design §6) |
|---|---|---|
| Coverage, no onset | 0.895 | ≥ 0.87: **pass** |
| Coverage, onset | 0.878 | ≥ 0.87: **pass (narrowly)** |
| Lowest single-fold coverage | no onset 0.84, onset 0.74 | not a criterion (about 11 positives per fold) |
| Set types | 66 % "no onset", 16 % "onset", 18 % "undecided", 0 % empty | reported |
| Variability-range width | median 0.019, mean 0.050, 95th percentile 0.21 | reported |

* With about 115 positives, coverage for the onset class has a standard error of about 0.03, so this run
  cannot distinguish 0.88 from the 0.90 target. The 14F full run (286 positives) is the real check.
* The main model's calibrated probability lies inside its own members' 10–90 % range for 78 % of patients.
  The members are trained on 80 % of the data and the main model on 100 %, so this is expected and is not
  an error.
* The R1 results were repeated: ECE fell from 0.067 to 0.013, and fold-mean AUROC was unchanged (0.878).

## Tests

* `tests/test_reliability_conformal.py` (10 tests) covers:
  * a hand-computed quantile and set example;
  * the +∞ case and JSON safety;
  * per-class coverage on simulated exchangeable data at α = 0.10 and 0.20 (20 seeds);
  * the round trip and validation;
  * member subsamples being stratified, deterministic and distinct;
  * a range hand example and the config validation;
  * nested-run outputs;
  * that flipping test labels leaves the conformal thresholds, the sets and the member ranges unchanged.
* Full backend suite: **736 passed, 8 skipped**.
* Mutation check: fitting the conformal sets on outer-test labels fails the label-flip test. The mutation was
  reverted.
* Ruff on the new files: only the repository's existing `X` naming convention (N806) is flagged.

## Limitations

* The nested-CV guarantee is approximate: calibration scores come from inner-fold models, and the calibrator
  and conformal sets are fitted on the same OOF patients.
* One small synthetic dataset. The full-size check belongs to 14F.
* Ensemble members add about M final fits per fold, which is expensive when the GRU is selected.
