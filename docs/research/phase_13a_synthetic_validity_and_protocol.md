# Phase 13A: Synthetic Data Validity and Research Protocol Review

Date: 2026-11-12
Branch / commit audited: `phase/12-real-dataset` @ `fead672`
Status: **read-only investigation. No production code, generator, configuration, or experiment output was modified.** Nothing was committed.

> All data and results discussed here are **synthetic**. They are not clinical evidence and must not be presented as such.

## How to read this document

- **[M] Measured** = produced by a lightweight diagnostic I ran for this phase (in-memory generation with the existing generator code, no files written to the repo, no Optuna, no model training other than small scikit-learn logistic regressions).
- **[R] Read** = established by reading code or saved metadata.
- **[P] Proposal** = my recommendation. Not yet validated.

Diagnostic method (so it can be repeated): regenerate 1,000-patient datasets with the existing `SyntheticGenerator` for seeds 42 to 46 using the v3 data settings; build cohorts with `CohortBuilder` using the v3 cohort settings; derive per-patient aggregate features from observation rows only; score simple rules (no fitting) on the pooled eligible cohort (3,898 patients, 214 positives) and fit logistic regressions with repeated stratified 5-fold CV. Seed-42 data was also compared with the existing CSV, and the seed-42 split was reproduced with `PatientSplitter` to evaluate baselines on the exact v3 validation and test patients. The scripts lived in the OS temp folder and have been removed. They can be recreated on request, or committed under `scripts/` if you approve.

---

## 1. Current synthetic generator assessment

### 1.1 What the generator does [R]

| Aspect | Implementation |
|---|---|
| Patient characteristics | Age ~ U(35, 75); sex, family history, smoking, activity drawn independently from fixed categorical distributions. A single latent `risk_bias ~ U(0,1)` shifts all baseline clinical values (HbA1c, glucose, BMI, BP, lipids). |
| HbA1c trajectory | `hba1c = base + slope * t + N(0, 0.1)` where **`t` is the visit index (0,1,2,...)**, not elapsed time. Linear, clipped to [4, 14]. |
| Slope | A hidden binary `will_progress ~ Bernoulli(sigmoid(logit))`. Progressors: slope ~ N(0.12 + modifier, 0.03). Non-progressors: slope ~ N(-0.02 + 0.3 * modifier, 0.03). `modifier` is a sum of fixed effects of activity, smoking, family history and metformin. |
| Glucose, BMI, BP | Baseline + own linear slope + noise. Glucose slope = N(1.0 * risk_bias + 3 * modifier, 0.5), **independent of `will_progress`**. BP slope is small. |
| LDL, HDL, TG, heart rate | Baseline + independent per-visit noise. **No trend.** |
| Medication | `metformin_use` is a **static** yes/no drawn once per patient with p = 0.05 + 0.20 * risk_bias. Its only effects: slope modifier -0.15 and logit -0.3. No start, stop, dose, adherence, or response heterogeneity. |
| Hypertension | `yes` if baseline systolic >= 130; becomes `yes` permanently if any later systolic >= 140. Past-dependent only. |
| Visit schedule | 4 to 6 observation visits + 2 to 4 future visits. Gaps ~ U(90, 240) days, independent of everything else. |
| Missingness | 5% of numeric cells in **observation** rows set to NaN, completely at random. **Future rows are never missing.** Applied after the label is computed. |
| `diabetes_diagnosis` | Always 0 in observation rows. In future rows equals `hba1c >= 6.5`. |

### 1.2 Measured properties [M]

| Property | Result |
|---|---|
| Reproducibility | The existing `data/synthetic/diabetes_progression_full.csv` (8,004 rows) is **identical** to regenerating with seed 42. |
| Quadratic term in HbA1c vs visit index | Mean coefficient 0.0005 (SD 0.0099): trajectories are linear to measurement precision. |
| Residual SD of per-patient linear fit | 0.084 vs visit index, 0.088 vs calendar days. Trajectories are effectively linear in **visit index**; visit spacing carries no information about the HbA1c process. |
| Visit gaps | Mean 165 d, SD 44, range 90 to 240 d (uniform). |
| Observation HbA1c missing rate | 5.4% (MCAR). Future HbA1c missing: **0%**. |
| Eligible patients with last HbA1c missing | 0.0%. With fewer than 2 valid HbA1c readings: 0.03%. |
| Observation visits per eligible patient | **Exactly 4 for all 786 eligible patients (seed 42).** Generated 4 to 6, cohort keeps the last 4. The sequence mask is therefore all ones in every experiment so far. |
| Full-trajectory slope distribution (diagnostic only; uses future visits) | Clearly bimodal: a large mass around -0.05 to 0.05 and a second mass from about 0.20 to 0.35. 33% of patients have a slope above 0.06 per visit. |

### 1.3 Assessment

Strengths: seeded and reproducible; observation and future rows are clearly separated; the outcome rule is identical in generator and cohort builder (tested); no obvious feature leakage (section 3).

Weaknesses relevant to scientific defensibility:

1. **Heterogeneity is thin.** Two latent regimes (flat or rising), each with a narrow Gaussian slope (SD 0.03). There are no phenotypes, no change-points, no remitters, no accelerating trajectories.
2. **Irregular visit intervals are cosmetic.** Because trajectories are linear in visit index, spacing does not affect values. The only role of spacing is how many future visits land inside the 365-day horizon.
3. **Measurement noise is small and the only noise source.** SD 0.1 on HbA1c, with no separation of biological variability from assay error.
4. **Glucose is physiologically decoupled from HbA1c.** Glucose slope does not depend on `will_progress`. Fasting glucose therefore adds little (single-feature AUROC 0.58 [M]).
5. **Medication is a static covariate**, not a time-varying exposure.
6. **Missingness is only MCAR and only in the observation window.** The "outcome unmeasured" situation never occurs, so the future-HbA1c handling in the cohort builder is never exercised.
7. **No variable-length histories reach the models** (all eligible patients have 4 visits).
8. **Prevalence control is unreliable.** The config says `progression_prevalence: 0.30`. Measured: label prevalence before exclusion is 23.7% to 28.3%; **after the cohort's baseline-diabetes exclusion it is 4.2% to 6.7%** (seed 42: 46 of 786 = 5.85%). Roughly 21% to 24% of generated patients are excluded as already diabetic by HbA1c.
9. **Statistical power is poor by construction.** One 1,000-patient dataset yields 32 to 52 positives in total. A 15% test split holds about 5 to 8.

---

## 2. Current target-generation methodology

### 2.1 Definition [R]

- Index date = date of the patient's last observation visit.
- Observation window = last up to 4 observation visits (`max_observation_visits: 4`).
- Horizon = 365 days after the index date.
- Eligibility: at least 3 observation visits; at least one future visit within the horizon; **no observation-window `diabetes_diagnosis == 1` and no observation HbA1c >= 6.5**.
- Label = 1 if any future visit within the horizon has `diabetes_diagnosis == 1` or `hba1c >= 6.5`.
- The observation / future split comes from the generator's `is_future_visit` flag, not from an independent index-date rule.

### 2.2 Properties [M]

| Observation | Result |
|---|---|
| Cohort funnel, seeds 42 to 46 | 764 to 803 eligible of 1,000; 197 to 236 excluded for prior diabetes; **0 excluded for insufficient visits; 0 for insufficient follow-up**. Two of the three exclusion rules never fire on synthetic data. |
| Eligible prevalence | 5.85%, 4.15%, 4.97%, 6.72%, 5.73% (pooled 5.49%, 214 of 3,898). |
| Label depends on follow-up density | Positive rate by number of future visits inside the horizon (seeds 42, 43): 1 visit 2.7% (n=452); 2 visits 6.1% (n=1,054); 3 visits 3.9% (n=51). The label is a maximum over visits, so more follow-up gives more chances to cross. Follow-up density is **not** a feature, so it acts as extra label noise but is also a design artefact. |
| Where the positives sit | Last observed HbA1c: positives median 6.36 (90th percentile 6.46), negatives median 5.54 (90th percentile 6.11). Most positives are patients who are already within about 0.15 of the 6.5 threshold at the index date. |
| Outcome ascertainment | Always observed; no missing or unmeasured outcome. |

### 2.3 Methodological observations

- The task is "will a currently non-diabetic patient's measured HbA1c cross 6.5 within a year". The input HbA1c is the same variable that defines the outcome, with a small amount of independent noise (0.1) and a linear trend. This is a legitimate prediction task, but with this noise level it is close to a deterministic extrapolation problem.
- The exclusion rule is applied to **noisy measurements**: a baseline patient with one high reading is excluded and one with a low reading is retained. This is reasonable, but it means roughly a fifth of the cohort is removed and the remaining positives are concentrated near the threshold.
- The 0.5 decision threshold on a 5.5% prevalence task makes accuracy meaningless (a constant "no progression" predictor scores 94.5% [M, arithmetic]).

---

## 3. Evidence of target predictability

### 3.1 Simple rules and logistic regression [M]

Pooled eligible cohort over seeds 42 to 46 (n = 3,898; 214 positives; prevalence 5.49%). Rules have no fitted parameters. Logistic regressions use 5-fold CV repeated 3 times, with imputation and scaling fitted inside each training fold.

| Predictor | AUROC | Notes |
|---|---|---|
| Constant (majority class) | 0.500 | AUPRC floor = prevalence 0.055 |
| Number of observation visits | 0.500 | Constant 4: no information, as expected |
| Mean visit gap | 0.481 | No leakage through schedule |
| Age | 0.549 | |
| Last BMI | 0.550 | |
| Last fasting glucose | 0.579 | |
| Mean HbA1c | 0.882 | |
| OLS slope of HbA1c (per visit) | 0.890 | |
| OLS slope of HbA1c (per day) | 0.889 | |
| Max HbA1c | 0.940 | |
| **Last observed HbA1c** | **0.961** | AUPRC 0.559 |
| Last HbA1c + 3 visits of slope | 0.970 | |
| Last HbA1c + 365 days of per-day slope | 0.973 | |
| **Last HbA1c + 2 visits of slope** | **0.974** | AUPRC 0.734 |
| LR on last HbA1c only | 0.961 +/- 0.009 | |
| **LR on last HbA1c + slope** | **0.974 +/- 0.008** | |
| LR on all HbA1c aggregates | 0.976 +/- 0.008 | |
| LR on everything (aggregates + other labs + demographics) | 0.976 +/- 0.008 | Other features add nothing beyond HbA1c |
| **LR with no HbA1c at all** (other labs + demographics) | **0.745 +/- 0.025** | Signal that flows through `risk_bias` |

Threshold rules (pooled): last HbA1c >= 5.7 gives recall 1.00, precision 0.13; >= 6.0 gives recall 0.96, precision 0.24; >= 6.2 gives recall 0.81, precision 0.42; "last + 2 slopes >= 6.5" gives recall 0.70, precision 0.60.

Per-seed (n about 770 to 800, 32 to 52 positives) LR on last HbA1c + slope: AUROC 0.981, 0.965, 0.982, 0.978, 0.961. The seed-to-seed range of 0.02 reflects sampling noise in a single dataset.

### 3.2 Comparison with the reported GRU result on the exact same patients [M]

I reproduced the v3 split (`PatientSplitter`, seed 42, stratified 70/15/15). It gives 550 / 118 / 118 patients with 32 / 7 / 7 positives, matching `model_metadata.json`.

| On the v3 **test** split (7 positives) | AUROC |
|---|---|
| Reported best model (GRU, chosen from 40 Optuna trials) | 0.950 |
| Rule: last HbA1c | 0.977 |
| Rule: last HbA1c + 2 slopes | 0.983 |
| LR (last HbA1c, slope) fitted on the v3 train split only | **0.985** |

On the v3 validation split the rules score 0.941 (last HbA1c) and 0.965 (extrapolation); the reported validation figures were 0.955 (search) and 0.991 (post-retrain).
A patient-level bootstrap of the extrapolation rule on the v3 test split gives a 95% CI of roughly 0.955 to 1.000 [M]; the GRU's 0.950 lies at or below the bottom of that interval, but with 7 positives this is not a reliable ranking.

**Conclusion [M + interpretation]:** the AutoML pipeline's selected model did not outperform a two-feature logistic regression. This is evidence that (a) the task is close to saturated for a trend-extrapolating model and (b) the current protocol cannot demonstrate any benefit of sequence models or AutoML on this data.

### 3.3 Is the label recoverable from observed HbA1c trends?

**Yes, almost.** Evidence: a single feature reaches 0.96 and two features 0.974 on 3,898 patients. What I could **not** measure is the Bayes-optimal ceiling, because the hidden slope, noise draws and number of horizon visits are not stored. The residual 2% to 3% comes from the independent future noise (SD 0.1), the slope spread (SD 0.03), and the number of horizon visits. The generator could emit its latent truth in a sidecar file so that this ceiling can be computed rather than guessed (proposal P-1 below).

Some of the predictability is legitimate: patients whose HbA1c is close to 6.5 and rising are truly more likely to cross it. The concern is that the noise level and the strictly linear, regime-based dynamics make the problem easier than any real clinical cohort. Real biological variability in HbA1c between visits is generally larger than 0.1, and real trajectories are not linear in visit count. I did not verify specific published figures in this phase, so treat that as an expectation to confirm against literature before it is cited.

### 3.4 Leakage assessment [R + M]

| Candidate | Finding |
|---|---|
| `target`, `is_future_visit`, future rows | Stripped by the cohort builder; builders raise on these names; runtime assertion that no observation row postdates the index date. |
| `diabetes_diagnosis` | Constant 0 in observation rows and not in the feature lists. |
| `visit_number` | Not in the feature lists. |
| Visit count | AUROC 0.500 (constant). |
| Visit spacing | AUROC 0.481. No relationship to the label via features. |
| `hypertension_diagnosis` | Updated only from current or earlier visits. |
| Missingness | Applied after the label and only to observation cells, MCAR. |
| Shared latent `risk_bias` | Legitimate confounding (drives baselines and progression), not leakage. Explains why non-HbA1c features reach 0.745. |
| Cohort exclusion | Uses observation data only. |

No feature leaks the target. The problem is **easy predictability from legitimate features**, not leakage.

### 3.5 Does this mean we should make the task harder?

Not for its own sake. The right question is whether the data-generating process is **plausible and documented** and whether the evaluation can distinguish models. At present neither the plausibility nor the discriminating power is established. The proposals in section 5 target plausibility and statistical power. Task difficulty should then be an *outcome* of the specification, reported with its oracle ceiling, rather than a tuned target number.

---

## 4. Recommended baselines

All baselines use the same cohort, the same patient-level splits, and the same metrics and bootstrap procedure as the models. None may use test labels to choose parameters.

| # | Baseline | What it measures | Evaluation notes | Measured here [M] |
|---|---|---|---|---|
| B0 | **Majority class / prevalence predictor** | The floor: what a model achieves with no information. Gives reference values for AUPRC (= prevalence), Brier (= p(1-p) = 0.052 at 5.5%), and log loss. Exposes why accuracy is uninformative. | Predict the training-fold prevalence for every patient. Report Brier skill score and AUPRC lift for all other models relative to it. AUROC is 0.5 by definition. | AUROC 0.500; accuracy 94.5% |
| B1 | **Last-HbA1c rule** ("previous HbA1c threshold") | How much of the task is simply "who is near 6.5 now". Clinically the nearest analogue to current practice (prediabetes thresholds). | Use the continuous last HbA1c as the score for AUROC, AUPRC and calibration (after a monotone mapping fitted on training folds). For threshold metrics use a threshold chosen on training/inner folds only (candidates 5.7, 6.0, or Youden). Do not select the threshold on test. Missing last value: use the last non-missing value, and define the rule when none exists. | AUROC 0.961; AUPRC 0.559 |
| B2 | **Trend-extrapolation rule** | Whether the longitudinal structure beyond the last value matters. Score = last HbA1c + slope * horizon. | Fix the slope estimator (OLS on calendar days over valid readings) and the extrapolation horizon (365 days) **in advance**, to avoid tuning on results. Report a robust variant (Theil-Sen) as a sensitivity check. Treat patients with fewer than 2 valid readings explicitly. **Warning:** on the current data this baseline matches the generating process (linear trend), so it is an unusually strong competitor; it should degrade on a nonlinear generator. | AUROC 0.973 to 0.974; AUPRC 0.734 |
| B3 | **Logistic regression on simple aggregates** | Whether flexible models add value over the simplest fitted model. | Features: last, mean, min, max, first HbA1c, slope, last glucose and other labs, age, categoricals. Fit on training folds only, with imputation and scaling in-fold. Report the HbA1c-only and "no HbA1c" variants as ablations (the latter isolates signal carried by correlated covariates). Regularisation fixed or tuned only by inner CV. | HbA1c aggregates 0.976; no-HbA1c 0.745 |

How baselines are used: the **primary comparison** is "best AutoML model vs B3" (and vs B2 as a sanity bound), using a paired bootstrap on the difference in AUROC and Brier. A model that does not beat B3 with a CI excluding zero should be reported as not demonstrably better.

Not recommended yet: the ADA 5.7% screening rule alone (covered by B1 thresholds), and deep baselines such as Cox models, until a time-to-event outcome exists.

---

## 5. Proposed generator improvements

Principles [P]:

- **Do not touch `v1`.** Keep the current generator as `legacy_v1` so all existing artefacts stay reproducible. Add new scenarios behind a version/scenario flag.
- **Specify the DGP a priori from clinical plausibility, document it, then freeze it before any model comparison.** Do not iterate generator parameters until models reach a desired score. Report the oracle ceiling of each scenario.
- **Emit ground-truth latent variables in a separate sidecar file** (never in the model-facing CSV) so we can compute the oracle AUROC and validate explanations against true drivers.
- Add only what addresses a stated problem. Items are tiered below.

### Summary

| ID | Change | Tier | Main problem addressed |
|---|---|---|---|
| G1 | Larger cohort and eligible-prevalence control | 1 (must) | Statistical power (5 to 8 test positives) |
| G2 | Ground-truth sidecar and oracle ceiling | 1 | Cannot quantify difficulty or validate explanations |
| G3 | Continuous-time trajectories with irregular intervals | 1 | Visit spacing is cosmetic; linear in index |
| G4 | Patient-specific continuous progression rates + subgroup mixture | 1 | Two narrow regimes, thin heterogeneity |
| G5 | Measurement noise model | 1 | Single small noise source |
| G6 | Outcome uncertainty (diagnosis process, confirmation, follow-up) | 1 | Label is a deterministic maximum of future values |
| G7 | Controlled missingness (MCAR / MAR / MNAR; future visits too) | 1 | Never exercises missing-outcome handling or informative missingness |
| G8 | Coupled glucose-HbA1c and correlated labs | 2 | Glucose carries no progression signal |
| G9 | Variable-length histories reaching the models | 2 | Mask is always all ones |
| G10 | Nonlinear progression (acceleration, plateau, change-point) | 2 | Linear extrapolation matches the DGP |
| G11 | Time-varying treatment (start, stop, response heterogeneity) | 2 | Medication is a static flag |
| G12 | Named clinical phenotypes | 3 (defer) | Realism; only after G3 to G7 are validated |

### G1. Larger cohort and eligible-prevalence control (Tier 1)

- *Problem:* 1,000 generated patients give about 786 eligible and 46 positives. A 15% test split holds about 7. Hanley-McNeil 95% half-width for AUROC at prevalence 5.5% [M, formula]: 7 positives gives +/-0.09 (A=0.97) to +/-0.15 (A=0.90); 30 positives +/-0.04 to +/-0.07; 100 positives +/-0.02 to +/-0.04.
- *Implementation:* generate about 5,000 patients (about 215 positives pooled, based on the measured 78% eligibility and 5.5% prevalence). Separately, re-parameterise so that incidence *among eligible* patients is a controlled, documented target (for example 10% to 20%, reflecting a higher-risk screened population) rather than an accident of exclusion. Report both generated and eligible prevalence. The existing `progression_prevalence` knob is not a reliable control (realised eligible prevalence is about a fifth of the setting).
- *Effect on target:* changes prevalence and sample size, not the mechanism.
- *Test:* a unit test asserting eligible prevalence lies within a tolerance of the configured target across seeds; a power check that pooled positives exceed the protocol minimum.
- *Leakage risk:* none.

### G2. Ground-truth sidecar and oracle ceiling (Tier 1)

- *Problem:* we cannot say how much of the remaining error is irreducible, nor whether SHAP or occlusion find the true drivers.
- *Implementation:* write `generation_truth.csv` (per patient: latent class, true rate, true noise draws summary, true outcome time, number of horizon visits; per visit: noise-free HbA1c). Add a function that computes the Bayes-style oracle AUROC given the true latent state at the index date.
- *Effect on target:* none.
- *Test:* the sidecar is never loaded by the loader, builders or cohort builder (canary test: removing it changes nothing); oracle AUROC is reproducible by seed.
- *Leakage risk:* only if the sidecar is accidentally joined to features. Mitigate by a separate path, a distinct file name pattern excluded from the loader, and the canary test.

### G3. Continuous-time trajectories with irregular intervals (Tier 1)

- *Problem:* trajectories are linear in visit index, so visit spacing has no effect on values.
- *Implementation:* define HbA1c as a function of elapsed days, `h(t_days)`. Draw inter-visit gaps from a right-skewed distribution (for example log-normal, median about 4 to 6 months, with occasional long gaps) so the horizon contains a variable number of follow-up visits. Optionally make scheduling depend on **past** state only (more frequent visits after a high reading).
- *Effect on target:* the number of observed future points in the horizon varies realistically; the per-visit slope baseline becomes biased, while the per-day one does not.
- *Test:* residual SD of a per-patient linear fit is lower versus calendar days than versus index (the reverse of the current measurement); visit gaps match the configured distribution; scheduling covariates never use future values.
- *Leakage risk:* low if scheduling depends only on earlier values. Scheduling that depends on the *future* outcome would leak; forbid it and test.

### G4. Patient-specific progression rates and subgroups (Tier 1)

- *Problem:* a binary regime with narrow slopes makes the future almost a deterministic function of the current slope.
- *Implementation:* draw a continuous patient-level rate from a mixture: for example stable / slow / fast progressors / improvers, with overlapping component distributions (overlap is a design choice justified a priori, not tuned), plus a random within-patient rate drift. Retain the existing covariate effects (activity, smoking, family history, treatment) as shifts.
- *Effect on target:* the label becomes a probabilistic function of observed history; AUROC falls only as a consequence of overlap.
- *Test:* the sidecar gives the class; check that rate distributions overlap as specified; oracle AUROC is below 1; marginal covariate effects recoverable by a correctly specified regression.
- *Leakage risk:* none; latent class stays in the sidecar.

### G5. Measurement noise model (Tier 1)

- *Problem:* a single noise SD of 0.1.
- *Implementation:* separate within-person biological variability (autocorrelated, SD configurable) from assay error (SD configurable, with rounding to 0.1 as in many reports), optional lab or site offset. Defaults chosen from literature that must be cited in the config comments; I have not verified specific values in this phase.
- *Effect on target:* observed trends are noisier; the true latent value still defines "true progression", while the label uses measured values (G6).
- *Test:* empirical noise SD recovered from repeated stable patients; rounding reflected in the data.
- *Leakage risk:* none.

### G6. Outcome uncertainty (Tier 1)

- *Problem:* the label is `max over future visits of (hba1c >= 6.5)`, a deterministic function of future measurements, with follow-up density affecting prevalence.
- *Implementation:* define the outcome via a documented clinical process: incident diabetes requires either a recorded diagnosis (assigned with a probability that depends on HbA1c and a lag) or two confirming readings, within the horizon. Optionally add a minimum-follow-up requirement. Document and test the rule in both generator and cohort builder (they must remain identical; the existing consistency test pattern applies).
- *Effect on target:* removes the dependence on a single noisy reading; adds irreducible outcome noise.
- *Test:* target-consistency test between generator and cohort builder; sensitivity of prevalence to follow-up density is reported and bounded.
- *Leakage risk:* the diagnosis event may only appear in future rows; observation rows must never contain a diagnosis flag.

### G7. Controlled missingness (Tier 1)

- *Problem:* only MCAR in the observation window. The pipeline's missing-future-HbA1c logic and informative missingness are never exercised.
- *Implementation:* a `missingness_mode` setting: MCAR; MAR (HbA1c measured less often when previous values were normal); MNAR (value-dependent). HbA1c measured less frequently than glucose, as in practice. Apply also to **future** visits. The cohort builder must then treat an unmeasured outcome window as *unknown* (exclude or flag), not negative.
- *Effect on target:* label ascertainment changes. If a future reading is missing, the label is unknown. This is an **intentional** change that must be paired with the cohort-builder fix (see section 6, issue I-7).
- *Test:* realised missing rate per column and mode; no patient labelled negative solely because the outcome was unmeasured; missingness indicators in features computed from observation rows only.
- *Leakage risk:* medium if MAR/MNAR mechanisms depend on future values. Restrict mechanisms to current or past values and test.

### G8. Coupled glucose-HbA1c and correlated labs (Tier 2)

- *Problem:* glucose does not track HbA1c; lipids and heart rate have no trend.
- *Implementation:* generate a latent glycaemic state and derive HbA1c (lagged, smoothed) and fasting glucose from it with separate noise; couple BMI and triglycerides to insulin resistance.
- *Effect:* additional correlated, redundant signal; robustness to missing HbA1c becomes testable (glucose as a surrogate).
- *Test:* correlation structure matches the specification; ablation shows glucose helps when HbA1c is missing.
- *Leakage:* none.

### G9. Variable-length histories (Tier 2)

- *Problem:* every eligible patient has exactly 4 observation visits, so the GRU never sees padding.
- *Implementation:* allow 2 to 6 observation visits and let the cohort keep up to 4 (min 3 stays as eligibility). Alternatively relax min to 2.
- *Effect on target:* none directly; exposes padding and masking behaviour.
- *Test:* distribution of lengths in the eligible cohort; a masking regression test; the occlusion fix (section 6) tested on padded inputs.
- *Leakage:* none.

### G10. Nonlinear progression (Tier 2)

- *Problem:* B2 matches the linear DGP. Sequence models have no opportunity to add value.
- *Implementation:* include a minority of patients with accelerating, plateauing, or change-point trajectories (for example a slope increase after an event), with parameters from the latent class.
- *Effect on target:* trend extrapolation underperforms on these patients; sequence and flexible models may gain.
- *Test:* the extrapolation baseline's advantage over LR shrinks as the nonlinear share grows (a *diagnostic* of the DGP, not an optimisation target); oracle ceiling reported.
- *Leakage:* none.

### G11. Time-varying treatment (Tier 2)

- *Problem:* medication is a patient-constant flag with a fixed effect.
- *Implementation:* treatment initiation as a hazard depending on **past** observed HbA1c and covariates (confounding by indication); stop and switch events; response heterogeneity; adherence.
- *Effect on target:* observed history now includes treatment-induced slope changes; a model must learn that treated patients flatten.
- *Test:* known average treatment effect recoverable from the sidecar; initiation hazard uses only earlier values; a treatment started after the index date must not appear in observation rows.
- *Leakage risk:* medium. Treatment triggered by *future* latent state would leak the outcome. Generate strictly in time order and test with a canary.

### G12. Named clinical phenotypes (Tier 3, defer)

Insulin-resistant obesity, lean beta-cell failure, remitting lifestyle responders, and so on. These are largely compositions of G4, G8, G10 and G11 and should only be added after those are validated. Not recommended now, to avoid arbitrary complexity.

### Scenario plan

| Scenario | Content | Purpose |
|---|---|---|
| `legacy_v1` | Existing generator, unchanged | Reproduce all prior artefacts; useful as a sanity or "easy" benchmark |
| `v2_standard` | G1 to G7 | Primary benchmark |
| `v2_extended` | v2_standard + G8 to G11 | Robustness / stress benchmark |

Each scenario is pre-registered (parameters, rationale, and oracle ceiling written down) **before** model results are inspected.

---

## 6. Review of existing issues: what affects synthetic experiments now

Classification: **Now** = distorts results on the current synthetic experiments or blocks the runner. **Runner** = must be fixed before the multi-seed runner is trustworthy. **Real** = mainly matters for future real-data integration.

| ID | Issue | Affects synthetic now? | Evidence | Priority |
|---|---|---|---|---|
| I-1 | **Validation metrics computed after train+validation retrain** | **Now** | [R] `pipeline.py` computes `val_metrics` on the retrained model; saved as `validation_metrics`. v3 reports val AUROC 0.991 against the search's held-out 0.955; v2 reports 1.0. Any comparison, plot or claim using these is contaminated. | Critical. Fix first (small). |
| I-2 | **Single validation split for Optuna and GRU early stopping** | **Now** | [M] Validation set has 7 positives (v3). Selection among 40 trials on 7 positives is dominated by noise. The selected GRU scored below a two-feature LR on the test split (0.950 vs 0.985). GRU early stopping, checkpoint choice and trial score all share this set. | Critical for the runner: replace with inner CV on the training fold. |
| I-3 | **Fixed 0.5 threshold despite `validation_f1` config** | **Now** | [M] Prevalence 5.5%. [R] The `threshold_strategy` field is validated but never read. v3 recall is 3 of 7. Threshold metrics (precision, recall, F1, accuracy, specificity) are therefore uninformative. | High. Fix in the runner: threshold from inner out-of-fold predictions; report decision curves. |
| I-4 | **Optuna study reuse** | **Now for any repeated run** | [R] fixed `study_name`, `load_if_exists=True`, SQLite in the artefact folder; in-memory best result vs persisted study can disagree. A multi-seed runner reusing a study name would silently share trials across seeds/folds, invalidating independence. | High (blocking for the runner). Unique study per run and fold, or in-memory storage with an explicit saved trial log. |
| I-5 | **GRU "pruning" does not prune; trial exceptions become pruned trials** | **Partly now** | [R] The GRU path breaks the loop but returns a value; the median pruner is therefore ineffective; any exception turns into `TrialPruned`. Affects compute cost and could hide failures, not metric correctness. | Medium. Fix with the runner (cost matters when multiplying by folds and seeds). Log pruned-by-exception counts. |
| I-6 | **Temporal occlusion and sequence masking** | **Now for explanation experiments only** | [R] Occluding step t zeroes it and sets `mask[t]=0`, but the GRU uses `mask.sum()` as length and packs the first `length` steps. [M] Every eligible patient has a full mask (4 of 4), so occluding any visit except the last drops the final visit and substitutes a zero step. Rankings for earlier visits are confounded; with the final visit being the most informative (last HbA1c AUROC 0.96), influence is mis-attributed. Prediction metrics are unaffected. | High for any explanation claim. Fix before explanation-stability results. |
| I-7 | **Missing future HbA1c treated as negative** | **Not now; Real** | [M] Future HbA1c missing rate in synthetic data is 0%, and cohort exclusions for follow-up never fire. [R] `_derive_future_target` drops NaN values and returns 0 if none remain. Becomes immediate **if** G7 (future missingness) is adopted, and decisive for any real dataset with sparse labs. | High for real data. Must be fixed together with G7. |
| I-8 | Metric implementation (custom AUROC/AUPRC; tie handling; trapezoidal AUPRC) | Now (minor) | [R] ties are not grouped. With saturated probabilities (v2) results could deviate from scikit-learn. | Medium. Add a parity test; prefer scikit-learn if environment allows. |
| I-9 | Prevalence config mismatch (0.30 vs 5.5% eligible) | **Now** | [M] section 2.2 | Fix via G1. |
| I-10 | Post-hoc statements of leakage controls (hard-coded `test_set_accessed_for_selection: false`; unused `HeldOutTestSet`) | Now (integrity) | [R] | Medium. Enforce in the runner, record the access counter. |
| I-11 | Wide/sequence patient-order inconsistency; `DataValidator` not run; unit handling; schema coupling | Real | [R] Pass on sorted synthetic IDs only. | Medium, with the adapter work. |
| I-12 | Missing-value indicators / time-delta features absent | Real (and G3, G7) | [R] | Medium. Becomes necessary once G3/G7 exist. |

Immediately affecting current synthetic experiments: **I-1, I-2, I-3, I-4, I-9** (and I-6 for explanations). Primarily real-data: **I-7, I-11, I-12**.

---

## 7. Experimental protocol and leakage prevention

### 7.1 Design [P]

**Data.**
- One primary dataset per scenario with at least 5,000 generated patients (about 215 or more positives among eligible patients at 5.5% prevalence; more if G1 raises prevalence). Pooled out-of-fold predictions should contain at least 150 positives.
- Independent replicates: 3 dataset seeds per scenario to quantify population-sampling variability.

**Resampling.**
- **Outer loop:** repeated stratified patient-level K-fold (for example 5-fold x 3 repeats) per dataset. Each patient appears in the outer test set once per repeat. The outer test fold is never used for selection.
- **Inner loop (per outer training set):** stratified patient-level 3- or 5-fold CV for hyperparameter search, early-stopping epoch determination, model-family selection, threshold selection and calibration fitting.
- Variance components reported separately: dataset seed, split seed, and model/training seed.
- If compute is prohibitive, a fallback is 5 dataset seeds x 1 stratified 60/20/20 split with the same inner CV; this is weaker because test sets are small.

**Model selection.**
- Fixed search budget per model family (for example 25 trials), same inner folds for all families.
- Primary selection metric: mean inner-CV AUROC, with Brier score or log loss as a tie-breaker (AUROC alone is noisy with few positives). Choose the metric before running.
- GRU: the epoch count is chosen from inner CV (median best epoch), then the final model is refit on the whole outer-training set with that fixed epoch count. This removes the different-recipe problem (ML-10 in the audit).
- Unique Optuna study per (run id, outer fold), never reused across runs.

**Final fit and test.**
- Refit the chosen configuration on the whole outer-training set; fit preprocessors on it; predict the outer test fold once.
- Threshold: from inner out-of-fold predictions (for example Youden index, or the threshold giving a pre-specified sensitivity), applied unchanged to the outer test fold. Report threshold-free metrics as primary and threshold metrics as secondary; add decision-curve analysis.

**Metrics.**
- Primary: AUROC, AUPRC (with lift over prevalence), Brier score and Brier skill score against B0.
- Calibration: pooled out-of-fold predictions; calibration slope and intercept; equal-mass (quantile) bin ECE in addition to the existing equal-width ECE; reliability plot. Optional post-hoc calibration (Platt/isotonic) fitted on inner out-of-fold predictions only and evaluated on outer test folds.

**Uncertainty.**
- Patient-level (cluster) bootstrap, at least 2,000 resamples, on pooled outer-test predictions; percentile or BCa intervals. Also report mean and SD across folds. Because folds share training data, use paired comparisons (below) rather than treating folds as independent.

**Comparisons.**
- Baselines B0 to B3 plus all model families evaluated on identical folds.
- Primary comparison pre-specified: best AutoML model vs B3 (ΔAUROC and ΔBrier, paired bootstrap on the same patients). Secondary comparisons use Holm correction. DeLong may be reported as a cross-check on pooled predictions. With the Nadeau-Bengio correction if a corrected resampled t-test is used.
- Ablations: wide vs sequence under the same budget; with and without HbA1c; with and without other labs.

**Robustness (perturbation).**
- Apply perturbations to **outer-test inputs only**, never to training data, before the training-fit preprocessor. Levels: measurement removal 5%, 10%, 20%, 30% (MCAR and MAR); removal of 1 visit; truncation to the most recent k visits; date jitter. At least 3 perturbation seeds per level. Labels come from the unperturbed future.
- Report ΔAUROC, ΔBrier and ΔECE with CIs; compare against baselines under the same perturbations.

**Explanation stability.**
- For outer-test patients: SHAP (tree models) and fixed temporal occlusion (GRU), with background data drawn from the outer training set only. Compute top-k Jaccard and Spearman (using absolute importance consistently) between clean and perturbed explanations; aggregate per patient then across patients with bootstrap CIs. Include a random-ranking reference. With the generator sidecar, also report agreement between explanations and true drivers.

**Reproducibility metadata** (extend `RunResult`):
- git commit and dirty flag, Python and OS, package versions (numpy, pandas, scikit-learn, xgboost, lightgbm, torch, shap, optuna), hardware and device, determinism flags.
- config hash; scenario name and generator version; dataset SHA-256; split-ID hash and the list of patient IDs per fold; seeds (data, split, model, Optuna, bootstrap, perturbation).
- test-set access counter, protocol version, wall-clock times, and a synthetic-data flag/disclaimer in every record.
- Run directories named with a UUID, never overwritten; results append-only.

### 7.2 Preventing test-set leakage during model selection [P]

| Layer | Control |
|---|---|
| Data | Model-facing files contain only observation features plus labels; latent truth only in a separate sidecar that no loader reads. |
| Cohort | Index-date rule and outcome rule use future data only to derive the label; leakage columns stripped by name and by a whitelist of allowed features rather than only a blacklist. |
| Split | Patient-level, stratified, deterministic; assertion of disjointness at every level; split IDs stored. |
| Preprocessing | Imputers, scalers, encoders and categorical vocabularies fitted **within** each (inner or outer) training fold only; refit for each inner fold. |
| Selection | Optuna objective, model-family choice, early-stopping epoch, threshold and calibrator use inner out-of-fold data only. |
| Test access | Instantiate and enforce the existing `HeldOutTestSet`-style guard: outer test predictions are produced only after the final configuration is frozen; the access count is written into the run record (measured, not hard-coded). |
| Perturbation | Applied only to outer-test inputs. |
| Explanations | Background and reference data from the training fold. |
| Protocol | Hyperparameter ranges, selection metric, primary comparison, and generator scenarios are frozen and recorded before the first outer-test evaluation; no generator parameters are altered in response to model results. |
| Automated tests | (a) Label-shuffle test: AUROC about 0.5 with CI covering 0.5. (b) Canary feature: inject a column equal to the future HbA1c and assert builders reject it. (c) Preprocessor statistics equal those of the training fold only. (d) Patient-order permutation invariance. (e) No patient ID in two outer folds' test sets within a repeat. (f) Optuna study names unique across folds. |

---

## 8. Prioritised implementation plan

Each step should be a separate, reviewable change with tests. Steps within a phase can be reordered, but dependencies are noted.

| Phase | Work | Depends on | Notes |
|---|---|---|---|
| **13B: Correctness fixes needed by any evaluation** | (1) Report held-out validation metrics and relabel or remove post-retrain ones (I-1). (2) Unique Optuna study per run (I-4). (3) Make GRU pruning real and stop turning exceptions into silent pruning, with counts logged (I-5). (4) Fix temporal occlusion to be mask-aware (I-6) and test on padded and unpadded inputs. (5) Add scikit-learn parity test for metrics (I-8). (6) CORS PUT/DELETE and repo hygiene from the earlier audit, if still pending. | none | Small. Preserves existing behaviour except where it was wrong. |
| **13C: Baselines module** | Implement B0 to B3 as evaluable models with the same interface; unit tests with hand-computed cases; fixed pre-specified parameters. | 13B | Allows an immediate re-assessment of existing v3 results on the corrected scale. |
| **13D: Generator v2 scenarios** | Scenario flag; `legacy_v1` unchanged and regression-tested to reproduce the existing CSV hash; G1 to G7; sidecar and oracle ceiling (G2); documentation of the DGP with citations to be verified; plausibility and leakage tests; pre-registration note. Then Tier 2 items (G8 to G11) as a second scenario. | 13B | Cohort-builder change for unmeasured outcome (I-7) lands with G7. |
| **13E: Unified evaluation runner** | Nested resampling, inner CV selection, threshold and calibration on inner OOF, bootstrap CIs, paired comparisons, `RunResult` with full provenance, never-overwrite outputs, enforced test-access guard. | 13B, 13C, 13D | The original "highest priority" item from the audit. |
| **13F: Calibration, robustness and stability integration** | Wire existing Phase 11 modules into the runner; add slope/intercept and quantile ECE; robustness curves; stability aggregation with ground-truth validation. | 13E, occlusion fix | |
| **13G: Reporting** | Automatic tables and figures with the synthetic-data disclaimer; run manifest; a one-command reproduction script. | 13E, 13F | |
| Later | Real-data adapter, schema-driven configuration, API/UI explanation endpoints | 13E | Deferred by decision. |

Suggested first deliverable after approval: **13B items (1) to (3)** plus **13C**, because they are small, remove known distortions, and immediately show how the existing models compare with honest baselines.

---

## 9. Definition of done for Phase 13A-driven work

The synthetic-data and protocol work is complete when:

1. **Generator**
   - `legacy_v1` reproduces the current CSV byte-for-byte or hash-for-hash (regression test).
   - `v2_standard` implements G1 to G7 with every parameter documented, with its clinical rationale and a cited source where a literature value is used.
   - A ground-truth sidecar exists, is isolated from loaders, and an oracle ceiling is reported per scenario.
   - Eligible prevalence is controlled and within tolerance of the configured target across at least 5 seeds; pooled positives meet the protocol minimum.
   - Tests cover reproducibility, leakage canaries, schedule independence from future values, missingness mechanisms, and generator/cohort outcome consistency.
   - Scenario parameters are frozen and recorded before model comparison.

2. **Baselines**: B0 to B3 implemented with fixed, pre-specified parameters, tested, and evaluated through the same pipeline as the models.

3. **Correctness fixes**: I-1 to I-6 resolved and tested; I-7 fixed together with future-outcome missingness; I-8 parity test in place.

4. **Runner and protocol**
   - Nested resampling with inner-CV selection, threshold and calibration on inner out-of-fold data only.
   - Bootstrap CIs, paired comparisons with the pre-specified primary comparison, and correction for secondary comparisons.
   - Calibration, perturbation robustness and explanation stability reported for the selected model and the baselines.
   - Every run emits a complete provenance record (section 7.1), never overwrites previous runs, and records the measured test-access count.

5. **Leakage controls**: all automated tests in section 7.2 pass, including label-shuffle (AUROC about 0.5) and the canary feature.

6. **Reporting integrity**: every result artefact carries the synthetic-data disclaimer; the oracle ceiling and baseline results are shown next to model results; no claim is made that the task is representative of real clinical performance; limitations (single DGP family, assumptions of the generator) are stated.

7. **Verification**: the full backend and frontend test suites pass; a complete protocol run is reproduced from a clean checkout using the recorded commit, config and seeds, and yields identical metrics (or documented non-determinism within stated tolerances).

---

## Appendix A: What I did not verify

- **Oracle ceiling.** Latent truth is not stored, so the Bayes-optimal AUROC was not computed. Statements that the remaining error is "mostly future noise, slope spread and horizon-visit count" are inferences from the generator code.
- **Literature values.** I did not look up published HbA1c variability, progression rates or incidence figures. Statements that real variability exceeds 0.1 and real trajectories are nonlinear are expectations to confirm before citation.
- **Models.** I did not train or re-evaluate GRU, XGBoost, LightGBM or Random Forest. The only fitted models were logistic regressions on aggregate features. Model-specific claims rely on the saved metadata.
- **Baseline comparability.** Pooled baseline numbers use all eligible patients across five seeds; the v3 split numbers use the single seed-42 split. Rule AUROCs involve no fitting. The LR on the v3 split was fitted on the v3 training patients only. Preprocessing details (median imputation, standardisation) differ from the project pipeline.
- **Hanley-McNeil intervals** are an analytic approximation for AUROC with independent samples; the bootstrap interval on the v3 test split is based on 7 positives and is itself unstable.
- The `progression_prevalence` realisation before exclusion (23.7% to 28.3%) was measured from generated label prevalence; I did not decompose it into `will_progress` rates.
- No real-data diagnostics were run and no real clinical dataset was touched.
