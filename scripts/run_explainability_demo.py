"""Phase 7 explainability demo — runs against the real saved artifacts.

DISCLAIMER: All data is synthetic demonstration data. Not clinical evidence.
Not suitable for medical validation.

Usage::

    python scripts/run_explainability_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)

from dialong_automl.models.lightgbm_model import LightGBMModel
from dialong_automl.features.preprocessing import WidePreprocessor
from dialong_automl.explainability import (
    TreeExplainerWrapper,
    EXPLANATION_DISCLAIMER,
)


def sep(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print("=" * 65)


def main() -> None:
    print("DiaLong-AutoML — Phase 7 Explainability Demo")
    print("DISCLAIMER:", EXPLANATION_DISCLAIMER)

    # ------------------------------------------------------------------ #
    # 1. Load real artifacts
    # ------------------------------------------------------------------ #
    sep("1. Loading real artifacts")
    model_path = Path("artifacts/best_model.joblib")
    prep_path  = Path("artifacts/preprocessor.joblib")

    if not model_path.exists() or not prep_path.exists():
        print("  ERROR: artifacts not found. Run training first.")
        sys.exit(1)

    model = LightGBMModel.load(model_path)
    prep  = WidePreprocessor.load(prep_path)
    print(f"  Model    : {model.name} (fitted={model.is_fitted})")
    print(f"  Prep     : n_output_features={len(prep.feature_names_out)}")

    # ------------------------------------------------------------------ #
    # 2. Build a synthetic validation-equivalent DataFrame
    # ------------------------------------------------------------------ #
    sep("2. Building synthetic validation DataFrame")
    # Load the wide feature mapping from seq vocabulary (if available)
    # For the demo we build a DataFrame matching the preprocessor's columns
    nc = prep.numeric_columns or []
    cc = prep.categorical_columns or []
    rng = np.random.default_rng(42)
    n = 75  # same size as original val set
    import pandas as pd
    data = {}
    for col in nc:
        data[col] = rng.standard_normal(n).astype("float32")
    for col in cc:
        cats = [c for c in prep._cat_categories.get(col, ["F", "M"])
                if c not in ("<MISSING>",)]
        if not cats:
            cats = ["unknown"]
        data[col] = [cats[i % len(cats)] for i in range(n)]
    df = pd.DataFrame(data)
    y_dummy = rng.integers(0, 2, n).astype(np.int32)
    print(f"  DataFrame shape: {df.shape}")
    print(f"  NOTE: Using synthetic data — not real patient data.")

    # ------------------------------------------------------------------ #
    # 3. Build explainer
    # ------------------------------------------------------------------ #
    sep("3. Building TreeExplainerWrapper")
    explainer = TreeExplainerWrapper(model, prep, threshold=0.5)
    print(f"  Explanation method: {explainer.explanation_method}")
    print(f"  SHAP available   : {explainer.explanation_method == 'shap_tree'}")

    # ------------------------------------------------------------------ #
    # 4. Global explanation
    # ------------------------------------------------------------------ #
    sep("4. Global feature importance (top 15)")
    global_exp = explainer.global_explanation(df, y=y_dummy, n_top=15)
    print(f"  n_samples   : {global_exp.n_samples}")
    print(f"  method      : {global_exp.explanation_method}")
    print(f"  disclaimer  : {global_exp.disclaimer}")
    print()
    print(f"  {'Rank':>4}  {'Column':40s}  {'Readable Name':45s}  {'Importance':>10}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*45}  {'-'*10}")
    for e in global_exp.features[:15]:
        print(f"  {e.rank:>4}  {e.column_name:40s}  {e.readable_name:45s}  {e.importance:10.4f}")

    # ------------------------------------------------------------------ #
    # 5. Patient-level explanation (first sample)
    # ------------------------------------------------------------------ #
    sep("5. Patient-level explanation (sample 0)")
    patient_exp = explainer.patient_explanation(
        df, row_index=0, patient_id="SYN_DEMO_00001"
    )
    print(f"  patient_id           : {patient_exp.patient_id}")
    print(f"  prediction_prob      : {patient_exp.prediction_probability:.4f}")
    print(f"  predicted_class      : {patient_exp.predicted_class}")
    print(f"  threshold            : {patient_exp.threshold}")
    print(f"  explanation_method   : {patient_exp.explanation_method}")
    print(f"  disclaimer           : {patient_exp.disclaimer}")
    if patient_exp.top_positive:
        print("\n  Top risk-increasing contributions:")
        for c in patient_exp.top_positive[:5]:
            print(f"    {c.rank:>3}. {c.readable_name:45s}  shap={c.shap_value:+.4f}")
    if patient_exp.top_negative:
        print("\n  Top risk-decreasing contributions:")
        for c in patient_exp.top_negative[:5]:
            print(f"    {c.rank:>3}. {c.readable_name:45s}  shap={c.shap_value:+.4f}")
    if patient_exp.explanation_method == "not_available":
        print("  (SHAP not available — only prediction probability provided)")

    # ------------------------------------------------------------------ #
    # 6. Persist artifacts
    # ------------------------------------------------------------------ #
    sep("6. Persisting explanation artifacts")
    out_dir = Path("artifacts/explanations")
    out_dir.mkdir(parents=True, exist_ok=True)

    global_path  = out_dir / "global_importance.json"
    patient_path = out_dir / "patient_SYN_DEMO_00001.json"

    global_exp.save(global_path)
    patient_exp.save(patient_path)

    print(f"  global_importance.json : {global_path.stat().st_size:,} bytes")
    print(f"  patient_SYN_DEMO_00001 : {patient_path.stat().st_size:,} bytes")

    # Verify disclaimer in both
    with open(global_path) as fh:
        assert json.load(fh)["disclaimer"] == EXPLANATION_DISCLAIMER
    with open(patient_path) as fh:
        assert json.load(fh)["disclaimer"] == EXPLANATION_DISCLAIMER
    print("  Disclaimers verified in both artifacts.")

    # ------------------------------------------------------------------ #
    # 7. Summary
    # ------------------------------------------------------------------ #
    sep("SUMMARY")
    print(f"  Model            : {model.name}")
    print(f"  Explanation      : {global_exp.explanation_method}")
    print(f"  n_features       : {global_exp.metadata['n_features_total']}")
    print(f"  Top feature      : {global_exp.features[0].readable_name}")
    print(f"  Top importance   : {global_exp.features[0].importance:.4f}")
    print(f"  Global artifact  : {global_path}")
    print(f"  Patient artifact : {patient_path}")
    print()
    print("  Phase 7 explainability demo COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
