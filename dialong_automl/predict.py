"""DiaLong-AutoML prediction CLI entry point.

Loads the artifacts produced by a training run (see
:class:`dialong_automl.training.pipeline.TrainingPipeline`) and scores a
single patient trajectory supplied as JSON.

Usage::

    python -m dialong_automl.predict \\
        --artifact-dir artifacts \\
        --input examples/patient_trajectory.json

The command:
1. Loads ``model_metadata.json`` to learn which model/representation and
   observation-window size were used during training.
2. Loads the saved model and the EXACT preprocessing state fitted during
   training.
3. Reads and validates the patient-trajectory JSON (observation visits
   only — any future/leakage column is rejected).
4. Rebuilds the same feature representation used in training and applies
   the saved preprocessor (never refits).
5. Prints a probability for the positive class and a binary prediction.

This module NEVER trains a model. If the artifacts are missing or
incompatible it fails with a clear error.

DISCLAIMER: Operates on synthetic demonstration data only. Predictions are
not clinical evidence and are not suitable for medical decisions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dialong_automl.data.loader import STRING_COLUMNS
from dialong_automl.features.wide import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _REPRESENTATION_LEAKAGE_COLUMNS,
)
from dialong_automl.models.base import REPRESENTATION_SEQUENCE, REPRESENTATION_WIDE

logger = logging.getLogger("dialong_automl.predict")

# Columns that must be present on every visit record.
_REQUIRED_VISIT_COLUMNS: tuple[str, ...] = ("visit_date",)

# Default artifact filenames written by the training pipeline.
_METADATA_FILENAME = "model_metadata.json"
_WIDE_MODEL_FILENAME = "best_model.joblib"
_SEQ_MODEL_FILENAME = "best_model.pt"
_WIDE_PREP_FILENAME = "preprocessor.joblib"
_SEQ_PREP_FILENAME = "preprocessor_seq.joblib"
_SEQ_VOCAB_FILENAME = "seq_vocabulary.json"


class PredictionError(RuntimeError):
    """Raised when prediction cannot be performed (bad input or artifacts)."""


# ---------------------------------------------------------------------------
# Input parsing & validation
# ---------------------------------------------------------------------------

def _load_patient_json(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Load and validate a patient-trajectory JSON file.

    The accepted JSON shapes are::

        {"patient_id": "SYN_00001", "visits": [ {visit}, {visit}, ... ]}

    or a bare list of visit objects::

        [ {visit}, {visit}, ... ]

    Each visit must contain ``visit_date`` plus any subset of the known
    clinical columns.  Future/leakage columns are rejected.

    Returns
    -------
    (patient_id, visits)
    """
    if not path.exists():
        raise PredictionError(f"Input file not found: {path}")
    if path.stat().st_size == 0:
        raise PredictionError(f"Input file is empty: {path}")

    try:
        # utf-8-sig tolerates an optional BOM that some editors/tools emit.
        with path.open("r", encoding="utf-8-sig") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise PredictionError(f"Input file is not valid JSON ({path}): {exc}") from exc

    patient_id = "UNKNOWN"
    if isinstance(payload, dict):
        patient_id = str(payload.get("patient_id", "UNKNOWN"))
        visits = payload.get("visits")
        if visits is None:
            raise PredictionError(
                "Input JSON object must contain a 'visits' list. "
                "Got keys: " + ", ".join(sorted(map(str, payload.keys())))
            )
    elif isinstance(payload, list):
        visits = payload
    else:
        raise PredictionError(
            "Input JSON must be an object with a 'visits' list or a list of "
            f"visit objects, got {type(payload).__name__}."
        )

    if not isinstance(visits, list) or len(visits) == 0:
        raise PredictionError("Patient trajectory must contain at least one visit.")

    for i, visit in enumerate(visits):
        if not isinstance(visit, dict):
            raise PredictionError(f"Visit #{i} is not a JSON object.")
        for col in _REQUIRED_VISIT_COLUMNS:
            if col not in visit:
                raise PredictionError(
                    f"Visit #{i} is missing the required field {col!r}."
                )
        leakage = sorted(set(visit) & _REPRESENTATION_LEAKAGE_COLUMNS)
        if leakage:
            raise PredictionError(
                f"Visit #{i} contains forbidden leakage column(s) {leakage}. "
                "Prediction input must contain observation data only — never "
                "future outcomes or the target."
            )

    return patient_id, visits


def _visits_to_frame(patient_id: str, visits: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a long-format observation DataFrame from parsed visits.

    Applies the same type conventions as the CSV loader: ``visit_date`` is
    parsed to datetime, categorical columns are strings, numeric columns are
    coerced to float.
    """
    df = pd.DataFrame(visits)
    df["patient_id"] = str(patient_id)

    # Parse visit_date.
    try:
        df["visit_date"] = pd.to_datetime(df["visit_date"])
    except Exception as exc:
        raise PredictionError(
            f"Could not parse visit_date values in the trajectory: {exc}"
        ) from exc
    if df["visit_date"].isna().any():
        raise PredictionError("One or more visit_date values are unparseable.")

    # Type coercion matching the training-time loader.
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in STRING_COLUMNS:
        if col in df.columns and col != "patient_id":
            df[col] = df[col].astype(str).str.strip()

    df = df.sort_values("visit_date").reset_index(drop=True)
    return df


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee every canonical feature column exists on *df*.

    Representation builders emit a feature layout based on the columns present
    in their input. At training time all canonical features are present, so the
    fitted preprocessor expects the full, fixed layout. At inference a partial
    trajectory may omit some features, which previously produced a truncated
    layout and an ``IndexError`` in ``SequencePreprocessor.transform``.

    Adding the missing columns as NaN (numeric) / None (categorical) restores
    the exact training layout. These are genuine "missing" markers — the fitted
    preprocessor imputes numeric NaNs with the training median and maps absent
    categoricals to the ``<MISSING>`` one-hot slot, identical to training-time
    missing-value handling. No clinical values are fabricated and feature order
    is unchanged.
    """
    out = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = None
    return out


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_metadata(artifact_dir: Path) -> dict[str, Any]:
    meta_path = artifact_dir / _METADATA_FILENAME
    if not meta_path.exists():
        raise PredictionError(
            f"Model metadata not found: {meta_path}. "
            "Train a model first with: "
            "python -m dialong_automl.train --config configs/demo_fast.yaml"
        )
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise PredictionError(f"model_metadata.json is corrupt: {exc}") from exc


def _load_wide_artifacts(artifact_dir: Path, model_name: str) -> tuple[Any, Any]:
    """Load a wide-representation model + its preprocessor."""
    from dialong_automl.features.preprocessing import WidePreprocessor
    from dialong_automl.models import build_model

    model_path = artifact_dir / _WIDE_MODEL_FILENAME
    prep_path = artifact_dir / _WIDE_PREP_FILENAME
    if not model_path.exists():
        raise PredictionError(f"Model artifact not found: {model_path}")
    if not prep_path.exists():
        raise PredictionError(f"Preprocessor artifact not found: {prep_path}")

    model_cls = type(build_model(model_name))
    model = model_cls.load(model_path)
    preprocessor = WidePreprocessor.load(prep_path)
    if not preprocessor.is_fitted:
        raise PredictionError("Loaded wide preprocessor is not fitted.")
    return model, preprocessor


def _load_sequence_artifacts(artifact_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load a sequence (GRU) model, its preprocessor, and the vocabulary."""
    from dialong_automl.features.preprocessing import SequencePreprocessor
    from dialong_automl.models.gru import GRUModel

    model_path = artifact_dir / _SEQ_MODEL_FILENAME
    prep_path = artifact_dir / _SEQ_PREP_FILENAME
    vocab_path = artifact_dir / _SEQ_VOCAB_FILENAME
    for p, what in (
        (model_path, "model"),
        (prep_path, "preprocessor"),
        (vocab_path, "sequence vocabulary"),
    ):
        if not p.exists():
            raise PredictionError(f"Sequence {what} artifact not found: {p}")

    model = GRUModel.load(model_path)
    preprocessor = SequencePreprocessor.load(prep_path)
    if not preprocessor.is_fitted:
        raise PredictionError("Loaded sequence preprocessor is not fitted.")
    with vocab_path.open("r", encoding="utf-8") as fh:
        vocab = json.load(fh)
    return model, preprocessor, vocab


# ---------------------------------------------------------------------------
# Prediction core
# ---------------------------------------------------------------------------

def predict_from_artifacts(
    artifact_dir: str | Path,
    patient_id: str,
    visits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score one patient trajectory using saved artifacts.

    Parameters
    ----------
    artifact_dir:
        Directory containing the training artifacts.
    patient_id:
        Patient identifier (used only for the output record).
    visits:
        List of observation-visit dicts (already leakage-checked).

    Returns
    -------
    dict with keys: ``patient_id``, ``probability``, ``prediction``,
    ``threshold``, ``model``, ``representation``, ``n_visits``, ``disclaimer``.
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        raise PredictionError(f"Artifact directory not found: {artifact_dir}")

    metadata = _load_metadata(artifact_dir)
    model_name = metadata.get("selected_model")
    representation = metadata.get("representation")
    threshold = float(metadata.get("threshold", 0.5))
    # Observation-window size used at training time == wide max_waves / seq len.
    max_waves = int(metadata.get("observation_visits", 4))
    if not model_name or not representation:
        raise PredictionError(
            "model_metadata.json is missing 'selected_model'/'representation'."
        )

    obs_df = _visits_to_frame(patient_id, visits)
    # Ensure the observation frame has EVERY canonical feature column so the
    # representation builders emit the full, fixed feature layout that the
    # fitted preprocessor expects. Absent columns become NaN (numeric) or None
    # (categorical) — genuine "missing" values that the fitted preprocessor
    # imputes exactly as it does for missing values during training. This does
    # NOT fabricate clinical values and does NOT change feature order; it only
    # prevents a truncated layout (which caused an IndexError in the sequence
    # SequencePreprocessor.transform when a trajectory omitted some features).
    obs_df = _ensure_feature_columns(obs_df)

    if representation == REPRESENTATION_WIDE:
        from dialong_automl.features.wide import WideRepresentationBuilder

        model, preprocessor = _load_wide_artifacts(artifact_dir, model_name)
        wide = WideRepresentationBuilder(max_waves=max_waves).build(obs_df)
        X = preprocessor.transform(wide.X)
        proba = float(np.asarray(model.predict_proba(X)).ravel()[0])

    elif representation == REPRESENTATION_SEQUENCE:
        from dialong_automl.features.sequence import SequenceRepresentationBuilder

        model, preprocessor, vocab = _load_sequence_artifacts(artifact_dir)
        seq = SequenceRepresentationBuilder(
            max_seq_len=max_waves,
            categorical_vocabularies=dict(vocab),
        ).build(obs_df)
        X = preprocessor.transform(seq.X, seq.mask)
        proba = float(
            np.asarray(model.predict_proba_with_mask(X, seq.mask)).ravel()[0]
        )
    else:
        raise PredictionError(
            f"Unknown representation {representation!r} in model metadata."
        )

    prediction = int(proba >= threshold)
    return {
        "patient_id": patient_id,
        "probability": proba,
        "prediction": prediction,
        "threshold": threshold,
        "model": model_name,
        "representation": representation,
        "n_visits": len(visits),
        "disclaimer": (
            "Synthetic demonstration model. Prediction is not clinical "
            "evidence and must not be used for medical decisions."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DiaLong-AutoML — score a patient trajectory from saved artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts",
        dest="artifact_dir",
        help="Directory containing the trained model artifacts.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the patient-trajectory JSON file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        dest="log_level",
        help="Log level (DEBUG/INFO/WARNING/ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    try:
        patient_id, visits = _load_patient_json(Path(args.input))
        result = predict_from_artifacts(args.artifact_dir, patient_id, visits)
    except PredictionError as exc:
        logger.error("Prediction failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
