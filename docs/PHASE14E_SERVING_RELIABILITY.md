# Phase 14E: reliability in predict.py, the API and the UI

> Synthetic data only. Nothing here is clinical evidence.

## What was added

| Where | Change |
|---|---|
| `training/pipeline.py` | Components for the served model, all fitted on train+validation (the test set is untouched): OOD reference (`ood.json`); `ensemble_members` refitted models (`reliability_members/member_XX.joblib\|pt`) on stratified 80 % subsamples of the final training rows, sharing the final preprocessor; width cut-offs from the members' range widths on those patients. The manifest `reliability.json` lists every component and is deleted at the start of saving, so a stale manifest can never describe a new model. The manifest is also recorded in `model_metadata.json` under `reliability.served_components`. |
| `predict.py` | If `reliability.json` exists, the response gets a `reliability` object: `level`, `reasons`, `flags`, `calibrated_probability`, `calibrated_threshold`, `prediction_set`, `conformal_coverage_target`, `range_low`/`range_high`, `ood_percentile`/`ood_category`, `method` and plain-language notes. Without the manifest (old artifacts) it is `null`. If the manifest exists but a component is missing, a clear `PredictionError` is raised. `probability`, `prediction` and `threshold` are unchanged. |
| `api/schemas.py` | `ReliabilityInfo` and `FlagModel`; optional `reliability` on `PredictResponse`, inherited by `SelectedPredictResponse`. |
| `frontend` | `ReliabilityPanel.tsx`, shown inside `PredictionResult`. It displays the level with a one-line meaning, the calibrated probability, the model-variability range, the 90 % prediction set and the reasons. Info flags are listed separately. It states that the range is not a confidence interval. Older models show "Reliability information is not available for this model version." Types are in `api/types.ts`; styles in `index.css`. |

**Difference from nested CV:** the served members share the final preprocessor instead of refitting it per
member. This is cheaper and avoids saving 10 preprocessors. The preprocessor uses no labels, and the nested-CV
evaluation (refit per member) remains the reference.

## To see it in the demo

The existing `artifacts/` folder predates 14E, so the UI shows "not available" until a model is retrained:

```powershell
.\.venv\Scripts\python.exe -m dialong_automl.train --config configs/demo_fast.yaml
```

## Tests

* `tests/test_reliability_serving.py` (6 tests) covers:
  * that the pipeline writes the manifest, members and components;
  * that predict returns a valid reliability object and keeps the raw decision rule: the calibrated
    probability crosses the calibrated threshold exactly when the raw prediction is 1;
  * that the info flag is never a reason;
  * that a missing HbA1c gives `Low`;
  * old artifacts returning `null` with the same probability, and a missing component raising an error;
  * the API contract for new and old artifacts;
  * GRU members (`.pt`) saved and scored.
* Two existing tests that count final-model fits now set `ensemble_members = 0`, because members add fits by
  design.
* Frontend: `tests/reliabilityPanel.test.tsx` (3 tests) checks the content, the wording, that info flags are
  not reasons, the `High` case without a range, and the old-artifact fallback. Vitest: **44 passed**;
  `tsc --noEmit` is clean.
* Backend: the full suite ran with 759 passed and 2 failed. The 2 were the fit-count tests above, which were
  then fixed. After the fixes, the affected files plus serving and predict tests pass (71). The full suite was
  not re-run after these test-only edits.

## Limitations

* Serving reliability is fitted on one validation split and is labelled optimistic. 14F reports the nested-CV
  evaluation.
* Members add about M model files and M predictions per request. For the GRU this adds latency.
