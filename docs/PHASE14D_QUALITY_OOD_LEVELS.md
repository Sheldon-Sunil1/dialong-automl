# Phase 14D: input-quality flags (R4), OOD score (R5), reliability level (R6)

> Synthetic data only. Nothing here is clinical evidence.

Design: `docs/research/phase_14a_reliability_design.md`, items R4–R6. The rules are as fixed there, before any
evaluation.

## What was added

| Where | Change |
|---|---|
| `reliability/quality.py` | `quality_flags(visits, max_visits, hba1c_age_days)` returns `Flag(code, severity, message)` for one patient. Flags: `no_hba1c` (severe), `single_hba1c` (moderate), `stale_hba1c` (> 365 days, severe), `old_hba1c` (181–365 days, moderate), `no_glucose` (moderate), `short_history` (info), `implausible_value` (severe; bounds mirrored from the frontend `NUMERIC_RANGES`), `no_visits` (severe). Uses only the submitted visits. |
| `reliability/ood.py` | `MahalanobisOOD`: the numeric B3 aggregate features are median-imputed and standardised with training statistics; Ledoit-Wolf covariance. Output: the percentile of the distance among the training patients' distances. Above the 95th percentile is `unusual`; above the 99th is `far`. JSON round trip. |
| `reliability/report.py` | `reliability_level` (R6 rule), `build_report` returns `ReliabilityReport(level, reasons, flags, prediction_set, range, ood_percentile)`, `width_cutoffs`. Every triggered reason is listed in plain words. Info flags never change the level. |
| `config/schema.py` | `reliability.ood_percentiles` (95, 99), `reliability.hba1c_age_days` (180, 365), `reliability.width_percentiles` (80, 95), all validated. |
| `experiments/nested_cv.py` | Per outer fold: OOD fitted on outer-train, and width cut-offs taken from the members' range widths on outer-train patients. `automl_calibrated` rows gain `ood_percentile`, `ood_category`, `flags` and `reliability_level`. The fold record gains `ood` and `ensemble.width_cutoffs`. `reliability_summary.json` gains the share, Brier score, misclassification and onset rate per level, and the flag shares. |

**Choice made here (the design left it open):** the design calls for width cut-offs from range widths "on the fit
partition". The members never score inner-OOF patients, so the cut-offs use their widths on the outer-training
patients. This uses no labels. Each member saw about 80 % of those patients, so the widths may be slightly
narrower than on new patients, and more test patients may cross the cut-offs. 14F will show how many.

**Not in 14D:** the single-split pipeline does not yet save OOD statistics or ensemble members, and `predict.py`
does not yet return a report. Both are 14E, which also serves the reliability output.

## Tests

* `tests/test_reliability_report.py` (20 tests) covers:
  * a clean patient with no flags;
  * each flag on hand-built visits, including the boundary at exactly 365 days (`old`, not `stale`);
  * OOD: an injected outlier is `far`, about 5 % of training patients exceed the 95th-percentile cut-off,
    missing values are imputed, and the JSON round trip;
  * the R6 truth table (10 cases);
  * that reports list every reason;
  * width cut-offs and config validation.
* In `tests/test_reliability_conformal.py`:
  * the nested-run test now checks the new columns, that every undecided set is `Low`, that level shares
    sum to 1, and that OOD is fitted on exactly the outer-training patients (`n_fit == n_train`);
  * the label-flip test also covers the OOD record, width cut-offs, percentiles and levels.
* Full backend suite: **756 passed, 8 skipped**.
* Ruff on the new files: only the repository's existing `X` naming convention (N806) is flagged.

## Limitations

* No evaluation yet. Whether `Low` predictions are worse than `High` ones within risk strata is the 14F
  acceptance test (design §6).
* The flag thresholds come from the v2 synthetic cohort's distribution (design §2.2), not from clinical
  guidance.
* OOD uses only the numeric aggregate features. It cannot detect unusual visit timing or unusual categorical
  values.
