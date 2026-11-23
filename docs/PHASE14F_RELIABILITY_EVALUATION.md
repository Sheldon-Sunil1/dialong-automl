# Phase 14F: evaluation of the reliability layer

> All data are **synthetic**. Nothing here is clinical evidence or shows clinical usefulness.

Acceptance criteria: `docs/research/phase_14a_reliability_design.md`, section 6, fixed before any reliability
result was seen. Failures are reported, not hidden.

## Runs

| Scenario | Run folder (`outputs/research_v4/nested_cv/`) | Eligible / positive | Runtime |
|---|---|---|---|
| `v2_standard`, 5,000 patients, seed 42 (primary) | `v2_standard_n5000_s42_phase14f` | 2,778 / 286 | 94 min |
| `legacy_v1`, 1,000 patients, seed 42 | `legacy_v1_n1000_s42_phase14f` | 786 / 46 | 19 min |

* Config: `configs/research_v4_nested_cv.yaml`, 2 repeats × 5 outer × 3 inner folds, 20 Optuna trials per
  fold, Platt calibration, α = 0.10, M = 10 members.
* The only code change before the runs was recording α = 0.20 conformal sets for the sensitivity analysis.
  They are never used by the reliability level.
* Evaluation: `scripts/evaluate_reliability.py`, which reads each run folder and does not modify it. Output is
  in `outputs/research_v4/reliability_eval/`. The bootstrap uses 1,000 patient-level resamples (seed 2026),
  with the same patients in both repeats. Estimates are means over repeats.
* The v2 runtime was 3.9 times the 14B run (24 min). The design estimated 1.5–3 times, and the extra cost
  comes mostly from refitting members when the GRU or random forest is selected.

## Results against the criteria

| Claim | Criterion | `v2_standard` | `legacy_v1` |
|---|---|---|---|
| **R1** calibration | calibrated ECE < raw; CITL CI contains 0; fold-mean AUROC unchanged | **Pass.** ECE 0.044 → 0.013 (difference 0.031 [0.019, 0.039]); CITL +0.02 [−0.12, +0.17]; fold-mean AUROC difference 0 | **Pass.** ECE 0.022 → 0.014; CITL −0.09 [−0.55, +0.35]; difference 0 |
| **R3** coverage, α = 0.10 | each class ≥ 0.87 | **Pass.** No onset 0.904 [0.894, 0.916]; onset 0.895 [0.862, 0.929] | **Pass.** 0.905 [0.884, 0.925]; 0.924 [0.843, 0.990] |
| R3 sensitivity, α = 0.20 | each class ≥ 0.77 | Pass: 0.807; 0.802 | Pass: 0.832; 0.848 |
| **R6** levels separate reliability | Low − High within risk deciles, CI > 0, for **Brier and misclassification** | **Fail (partial).** Brier +0.032 [+0.010, +0.054] passes. Misclassification +0.012 [−0.023, +0.045] does not | **Fail.** Brier +0.012 [−0.003, +0.049]; misclassification +0.012 [−0.004, +0.054] |
| **R2** range width ranks error | Spearman(width, \|calibrated p − oracle p\|) > 0, CI excluding 0 | **Pass.** ρ = 0.86 [0.85, 0.88] | not applicable (no oracle) |
| Selective prediction | error falls as Low, then Moderate, are dropped | **Monotone.** Brier 0.069 → 0.053 → 0.019; misclassification 0.128 → 0.114 → 0.034 (retained 100 % → 75 % → 50 %) | **Monotone.** Brier 0.029 → 0.014 → 0.002 (100 % → 91 % → 73 %) |
| R7 candidate (disagreement with B3) | adopt only if it improves separation | Not adopted. Brier difference 0.034 vs 0.032 without it; the CIs overlap almost entirely | Not adopted (0.010 vs 0.012) |

### Prediction sets and levels on `v2_standard`

* α = 0.10 sets: 67 % "no onset", 15 % "onset", 18 % "undecided", 0 % empty.
* α = 0.20 sets: 74 % "no onset", 22 % "onset", 0 % undecided, 4 % empty.
* Levels: 50 % High, 25 % Moderate, 25 % Low. 73 % of Low predictions are Low because of an undecided set.
* Unstratified error by level:

  | Level | Brier | Misclassification | Onset rate |
  |---|---|---|---|
  | High | 0.019 | 0.034 | 2.7 % |
  | Moderate | 0.119 | 0.268 | 20.7 % |
  | Low | 0.116 | 0.173 | 14.9 % |

* Most frequent flags:
  * `short_history` 20 % (info only);
  * `single_hba1c` 10 %, `old_hba1c` 9 %, `ood_unusual` 4 %;
  * `stale_hba1c` 3.5 %, `no_hba1c` 1.7 %, `ood_far` 1.5 %.

## Reading

1. **Calibration (R1) works and should be shown.** The displayed probabilities are now about as well calibrated as
   the logistic-regression baseline, and ranking is unchanged.
2. **The 90 % sets keep their coverage for both classes in both scenarios.** The guarantee in nested CV is
   approximate, but it held empirically. 18 % of v2 patients get an "undecided" set. This is the honest cost of
   a 90 % guarantee at an AUROC of about 0.89.
3. **The reliability level is useful, but it met the strict pre-specified criterion only partly.**
   * Within the same risk decile, Low predictions have a higher Brier score than High ones (v2: +0.032, CI
     excluding 0).
   * The misclassification difference is not significant. Within a narrow risk band, the yes/no decision is
     dominated by whether the patient lies near the threshold, which the level does not target.
   * On legacy (46 positives) there is too little data to show a difference.
   * Unstratified, the separation is large (Brier 0.019 for High vs 0.116 for Low). Much of that reflects risk
     level, which is why the stratified test was pre-specified.
   * **The pre-specified R6 criterion is therefore not met.** The levels must not be described as validated
     separation. They are a transparent, rule-based summary with a measured within-stratum Brier benefit on the
     primary scenario.
4. **Range width tracks the distance from the latent truth strongly (ρ = 0.86).** This is partly confounded with
   risk: both the width and the error are near 0 for very low-risk patients. The design did not pre-specify a
   stratified version of this test, so the result should be read as supportive, not conclusive.
5. **Abstaining helps.** Dropping Low predictions lowers error on the rest. Keeping only High predictions keeps 50 %
   of patients with a misclassification rate of 3.4 % (v2). Those retained patients are also mostly low-risk.
6. **R7 is not added.** Disagreement with B3 did not improve separation.

## What can be claimed

On synthetic longitudinal diabetes data, the framework produces calibrated probabilities, class-conditional 90 %
prediction sets whose coverage held in nested CV, and a model-variability range that tracks error. The rule-based
reliability level separated prediction quality within risk strata on the Brier score in the primary scenario,
but not on misclassification and not on the smaller legacy scenario. None of this establishes clinical
validity.

## Limitations

* One synthetic dataset per scenario. The legacy scenario has only 46 positives.
* Conformal validity in nested CV is approximate. The calibrator and conformal thresholds share the same OOF
  patients.
* The oracle is a coarse Monte Carlo estimate (calibration slope well below 1). It is used only as a ranking
  reference.
* The thresholds for flags and levels come from the v2 distribution, not from clinical guidance.
* Reliability for the B3 baseline (design decision 4) was not computed, to limit runtime.
