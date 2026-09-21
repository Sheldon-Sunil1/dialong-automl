"""Long-format CSV loader for DiaLong-AutoML longitudinal datasets.

Responsibilities
----------------
- Read a CSV file from disk using :func:`pathlib.Path`.
- Normalise and validate expected column names.
- Parse ``visit_date`` as :class:`pandas.Timestamp`.
- Preserve patient IDs as strings (no numeric coercion).
- Preserve categorical values without silent coercion.
- Return a predictable :class:`pandas.DataFrame`.
- Raise clear exceptions on serious data problems.

Usage::

    from dialong_automl.data.loader import load_longitudinal_csv

    df = load_longitudinal_csv("data/synthetic/diabetes_progression_demo.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that MUST be present for any longitudinal dataset
REQUIRED_COLUMNS: list[str] = ["patient_id", "visit_date"]

# Columns expected in the DiaLong synthetic dataset
EXPECTED_COLUMNS: list[str] = [
    "patient_id",
    "visit_date",
    "visit_number",
    "hba1c",
    "fasting_glucose",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
    "heart_rate",
    "age_at_visit",
    "sex",
    "smoking_status",
    "physical_activity_level",
    "family_history_diabetes",
    "metformin_use",
    "hypertension_diagnosis",
    "diabetes_diagnosis",
]

# These columns are kept as strings
STRING_COLUMNS: set[str] = {
    "patient_id",
    "sex",
    "smoking_status",
    "physical_activity_level",
    "family_history_diabetes",
    "metformin_use",
    "hypertension_diagnosis",
}


class LoadError(ValueError):
    """Raised when the CSV cannot be loaded due to a data problem."""


def load_longitudinal_csv(
    path: str | Path,
    *,
    expected_columns: list[str] | None = None,
    warn_missing_columns: bool = True,
    date_format: str | None = None,
) -> pd.DataFrame:
    """Load a long-format longitudinal dataset from CSV.

    Parameters
    ----------
    path:
        Path to the CSV file.
    expected_columns:
        List of column names to check for.  Defaults to
        :data:`EXPECTED_COLUMNS`.  Missing columns produce a warning (or
        error when ``warn_missing_columns=False``).
    warn_missing_columns:
        When ``True`` (default), missing expected columns produce a log
        warning.  When ``False``, they raise :class:`LoadError`.
    date_format:
        ``strftime`` format string for parsing ``visit_date``.
        ``None`` lets pandas infer the format.

    Returns
    -------
    pandas.DataFrame
        Loaded and normalised dataframe with:
        - ``visit_date`` as ``datetime64[ns]``
        - ``patient_id`` as ``str``
        - categorical columns as ``str``
        - numeric columns as ``float64`` where parseable

    Raises
    ------
    LoadError
        If the file does not exist, is empty, is missing required columns,
        or has an unparseable ``visit_date`` column.
    """
    path = Path(path)
    expected_columns = expected_columns or EXPECTED_COLUMNS

    # --- File existence ---
    if not path.exists():
        raise LoadError(f"Data file not found: {path}")
    if path.stat().st_size == 0:
        raise LoadError(f"Data file is empty: {path}")

    logger.info("Loading longitudinal dataset from %s", path)

    # --- Read CSV ---
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception as exc:
        raise LoadError(f"Failed to read CSV {path}: {exc}") from exc

    if df.empty:
        raise LoadError(f"CSV file loaded but contains no rows: {path}")

    logger.debug("Raw load: %d rows × %d columns", len(df), len(df.columns))

    # --- Normalise column names (strip whitespace, lowercase) ---
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Required columns ---
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise LoadError(
            f"Missing required columns {missing_required} in {path}. "
            f"Found columns: {list(df.columns)}"
        )

    # --- Expected columns ---
    missing_expected = [c for c in expected_columns if c not in df.columns]
    if missing_expected:
        msg = (
            f"Missing expected columns {missing_expected} in {path}. "
            "These columns will not be available for downstream processing."
        )
        if warn_missing_columns:
            logger.warning(msg)
        else:
            raise LoadError(msg)

    # --- patient_id: keep as string, strip whitespace ---
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    empty_ids = df["patient_id"].isin(["", "nan", "None", "NaN"])
    if empty_ids.any():
        raise LoadError(
            f"Found {empty_ids.sum()} rows with empty/null patient_id in {path}."
        )

    # --- Parse visit_date ---
    raw_dates = df["visit_date"].copy()
    try:
        if date_format:
            df["visit_date"] = pd.to_datetime(df["visit_date"], format=date_format)
        else:
            df["visit_date"] = pd.to_datetime(df["visit_date"])
    except Exception as exc:
        # Surface a clear error with examples of unparseable values
        bad_samples = (
            raw_dates[pd.to_datetime(raw_dates, errors="coerce").isna()]
            .dropna()
            .head(5)
            .tolist()
        )
        raise LoadError(
            f"Could not parse visit_date column in {path}. "
            f"Sample bad values: {bad_samples}. Original error: {exc}"
        ) from exc

    unparseable = df["visit_date"].isna().sum()
    if unparseable > 0:
        raise LoadError(
            f"{unparseable} rows have unparseable visit_date in {path}."
        )

    # --- Type coercions ---
    # String columns: ensure str, strip
    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Numeric columns: coerce to float, preserving NaN
    numeric_candidates = set(expected_columns) - STRING_COLUMNS - {"visit_date", "visit_number"}
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # visit_number: int if present
    if "visit_number" in df.columns:
        df["visit_number"] = pd.to_numeric(df["visit_number"], errors="coerce").astype("Int64")

    # target: int if present
    if "target" in df.columns:
        df["target"] = pd.to_numeric(df["target"], errors="coerce").astype("Int64")

    # diabetes_diagnosis: int if present
    if "diabetes_diagnosis" in df.columns:
        df["diabetes_diagnosis"] = pd.to_numeric(
            df["diabetes_diagnosis"], errors="coerce"
        ).astype("Int64")

    logger.info(
        "Loaded %d rows, %d unique patients from %s",
        len(df),
        df["patient_id"].nunique(),
        path,
    )
    return df
