"""Dataset validator for DiaLong-AutoML longitudinal datasets.

Produces structured :class:`ValidationResult` objects rather than
printing arbitrary text so callers can act on individual checks
programmatically.

Usage::

    from dialong_automl.data.validator import DataValidator, ValidationConfig

    cfg = ValidationConfig()
    validator = DataValidator(cfg)
    result = validator.validate(df)
    print(result.summary())
    if not result.passed:
        raise RuntimeError("Dataset failed validation")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known leakage-candidate column name substrings
# ---------------------------------------------------------------------------

LEAKAGE_PATTERNS: list[str] = [
    "target",
    "diabetes_progression",
    "future_diabetes",
    "outcome",
    "diagnosis_after",
    "post_outcome",
    "future_hba1c",
    "future_glucose",
]

BINARY_CATEGORICAL_COLUMNS = {
    "family_history_diabetes": {"yes", "no"},
    "metformin_use": {"yes", "no"},
    "hypertension_diagnosis": {"yes", "no"},
}

VALID_TARGET_VALUES: set[int] = {0, 1}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ValidationConfig:
    """Configuration that controls validator behaviour.

    Parameters
    ----------
    allow_duplicate_patient_date:
        When ``False`` (default), duplicate ``patient_id + visit_date``
        pairs are reported as errors.
    target_inconsistency_as_error:
        When ``True`` (default), patients whose ``target`` value changes
        across visits raise an error-level finding.  When ``False``,
        it is a warning.
    require_chronological_order:
        When ``True`` (default), out-of-order visit dates within a patient
        are reported (but not silently corrected).
    expected_clinical_columns:
        Columns that should be present.  Missing columns are warnings.
    leakage_blacklist:
        Additional column name substrings to flag as leakage candidates.
    """

    allow_duplicate_patient_date: bool = False
    target_inconsistency_as_error: bool = True
    require_chronological_order: bool = True
    expected_clinical_columns: list[str] = field(default_factory=lambda: [
        "hba1c", "fasting_glucose", "bmi", "systolic_bp", "diastolic_bp",
        "ldl_cholesterol", "hdl_cholesterol", "triglycerides",
        "heart_rate", "age_at_visit",
        "sex", "smoking_status", "physical_activity_level",
        "family_history_diabetes", "metformin_use", "hypertension_diagnosis",
        "diabetes_diagnosis",
    ])
    leakage_blacklist: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single validation finding."""

    level: str       # "error" | "warning" | "info"
    code: str        # short machine-readable code
    message: str
    details: Any = None


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Structured outcome of a validation run.

    Attributes
    ----------
    passed:
        ``True`` if no error-level findings were produced.
    findings:
        List of all :class:`Finding` objects.
    profile:
        Basic dataset profile produced alongside validation.
    """

    passed: bool
    findings: list[Finding]
    profile: dict[str, Any]

    # ------------------------------------------------------------------

    def errors(self) -> list[Finding]:
        """Return only error-level findings."""
        return [f for f in self.findings if f.level == "error"]

    def warnings(self) -> list[Finding]:
        """Return only warning-level findings."""
        return [f for f in self.findings if f.level == "warning"]

    def summary(self) -> str:
        """Return a human-readable summary string."""
        n_err = len(self.errors())
        n_warn = len(self.warnings())
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Validation {status}: {n_err} error(s), {n_warn} warning(s)",
            f"  Rows: {self.profile.get('row_count', '?')}",
            f"  Patients: {self.profile.get('unique_patient_count', '?')}",
        ]
        if self.profile.get("target_present"):
            lines.append(
                f"  Positive patients: {self.profile.get('positive_patient_count', '?')}"
            )
            lines.append(
                f"  Negative patients: {self.profile.get('negative_patient_count', '?')}"
            )
        for f in self.findings:
            prefix = {"error": "  ✗ ERROR", "warning": "  ⚠ WARN ", "info": "  ✓ INFO "}.get(
                f.level, "  ? "
            )
            lines.append(f"{prefix} [{f.code}] {f.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class DataValidator:
    """Validate a longitudinal DataFrame against DiaLong-AutoML expectations.

    Parameters
    ----------
    config:
        Validation configuration.  Uses defaults when omitted.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.cfg = config or ValidationConfig()

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Run all validation checks and return a structured result.

        Parameters
        ----------
        df:
            Loaded dataframe (output of :func:`~dialong_automl.data.loader.load_longitudinal_csv`).

        Returns
        -------
        ValidationResult
        """
        findings: list[Finding] = []

        # --- Basic structural checks ---
        self._check_required_columns(df, findings)
        self._check_patient_id_not_empty(df, findings)
        self._check_visit_date_valid(df, findings)
        self._check_duplicate_patient_date(df, findings)
        self._check_chronological_order(df, findings)

        # --- Expected clinical columns ---
        self._check_expected_columns(df, findings)

        # --- Target checks (if present) ---
        if "target" in df.columns:
            self._check_target_binary(df, findings)
            self._check_target_consistency(df, findings)

        # --- Binary categorical fields ---
        self._check_binary_categorical(df, findings)

        # --- Leakage detection ---
        self._detect_leakage_candidates(df, findings)

        # --- Build profile ---
        profile = self._build_profile(df)

        passed = not any(f.level == "error" for f in findings)

        logger.info(
            "Validation %s: %d errors, %d warnings",
            "PASSED" if passed else "FAILED",
            sum(1 for f in findings if f.level == "error"),
            sum(1 for f in findings if f.level == "warning"),
        )

        return ValidationResult(passed=passed, findings=findings, profile=profile)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_required_columns(
        self, df: pd.DataFrame, findings: list[Finding]
    ) -> None:
        for col in ["patient_id", "visit_date"]:
            if col not in df.columns:
                findings.append(
                    Finding(
                        level="error",
                        code="MISSING_REQUIRED_COL",
                        message=f"Required column '{col}' is absent.",
                    )
                )

    def _check_patient_id_not_empty(
        self, df: pd.DataFrame, findings: list[Finding]
    ) -> None:
        if "patient_id" not in df.columns:
            return
        empty_mask = df["patient_id"].astype(str).str.strip().isin(
            ["", "nan", "None", "NaN", "<NA>"]
        )
        n = int(empty_mask.sum())
        if n > 0:
            findings.append(
                Finding(
                    level="error",
                    code="EMPTY_PATIENT_ID",
                    message=f"{n} rows have empty or null patient_id.",
                    details={"count": n},
                )
            )

    def _check_visit_date_valid(
        self, df: pd.DataFrame, findings: list[Finding]
    ) -> None:
        if "visit_date" not in df.columns:
            return
        if df["visit_date"].dtype == object:
            parsed = pd.to_datetime(df["visit_date"], errors="coerce")
        else:
            parsed = df["visit_date"]
        n_invalid = int(parsed.isna().sum())
        if n_invalid > 0:
            findings.append(
                Finding(
                    level="error",
                    code="INVALID_VISIT_DATE",
                    message=f"{n_invalid} rows have invalid or unparseable visit_date.",
                    details={"count": n_invalid},
                )
            )

    def _check_duplicate_patient_date(
        self, df: pd.DataFrame, findings: list[Finding]
    ) -> None:
        if not all(c in df.columns for c in ["patient_id", "visit_date"]):
            return
        dup_mask = df.duplicated(subset=["patient_id", "visit_date"], keep=False)
        n_dups = int(dup_mask.sum())
        if n_dups > 0:
            level = "warning" if self.cfg.allow_duplicate_patient_date else "error"
            findings.append(
                Finding(
                    level=level,
                    code="DUPLICATE_PATIENT_DATE",
                    message=(
                        f"{n_dups} rows are duplicates of patient_id + visit_date. "
                        f"({'allowed by config' if self.cfg.allow_duplicate_patient_date else 'not allowed'})"
                    ),
                    details={"count": n_dups},
                )
            )

    def _check_chronological_order(
        self, df: pd.DataFrame, findings: list[Finding]
    ) -> None:
        if not self.cfg.require_chronological_order:
            return
        if not all(c in df.columns for c in ["patient_id", "visit_date"]):
            return
        out_of_order_patients: list[str] = []
        for pid, grp in df.groupby("patient_id", sort=False):
            dates = grp["visit_date"].values
            if not all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1)):
                out_of_order_patients.append(str(pid))
        if out_of_order_patients:
            findings.append(
                Finding(
                    level="warning",
                    code="NON_CHRONOLOGICAL_ORDER",
                    message=(
                        f"{len(out_of_order_patients)} patient(s) have visits not in "
                        "chronological order.  Data was NOT silently sorted."
                    ),
                    details={"patient_sample": out_of_order_patients[:10]},
                )
            )

    def _check_expected_columns(
        self, df: pd.DataFrame, findings: list[Finding]
    ) -> None:
        missing = [c for c in self.cfg.expected_clinical_columns if c not in df.columns]
        if missing:
            findings.append(
                Finding(
                    level="warning",
                    code="MISSING_EXPECTED_COL",
                    message=f"Expected clinical columns not found: {missing}",
                    details={"missing": missing},
                )
            )

    def _check_target_binary(
        self, df: pd.DataFrame, findings: list[Finding]
    ) -> None:
        if "target" not in df.columns:
            return
        non_null = df["target"].dropna()
        unique_vals = set(non_null.astype(int).unique()) if len(non_null) > 0 else set()
        invalid = unique_vals - VALID_TARGET_VALUES
        if invalid:
            findings.append(
                Finding(
                    level="error",
                    code="NON_BINARY_TARGET",
                    message=(
                        f"'target' column contains non-binary values: {invalid}. "
                        "Expected only 0 and 1."
                    ),
                    details={"invalid_values": sorted(invalid)},
                )
            )

    def _check_target_consistency(
        self, df: pd.DataFrame, findings: list[Finding]
    ) -> None:
        if "target" not in df.columns or "patient_id" not in df.columns:
            return
        inconsistent: list[str] = []
        for pid, grp in df.groupby("patient_id", sort=False):
            non_null_targets = grp["target"].dropna().unique()
            if len(non_null_targets) > 1:
                inconsistent.append(str(pid))
        if inconsistent:
            level = "error" if self.cfg.target_inconsistency_as_error else "warning"
            findings.append(
                Finding(
                    level=level,
                    code="INCONSISTENT_TARGET",
                    message=(
                        f"{len(inconsistent)} patient(s) have inconsistent target values "
                        "across visits."
                    ),
                    details={"patient_sample": inconsistent[:10]},
                )
            )

    def _check_binary_categorical(
        self, df: pd.DataFrame, findings: list[Finding]
    ) -> None:
        for col, valid_set in BINARY_CATEGORICAL_COLUMNS.items():
            if col not in df.columns:
                continue
            col_vals = set(df[col].dropna().astype(str).str.strip().str.lower().unique())
            invalid = col_vals - valid_set - {"nan", "none", "<na>"}
            if invalid:
                findings.append(
                    Finding(
                        level="warning",
                        code="INVALID_BINARY_CATEGORICAL",
                        message=(
                            f"Column '{col}' has unexpected values: {invalid}. "
                            f"Expected one of {valid_set}."
                        ),
                        details={"column": col, "invalid_values": sorted(invalid)},
                    )
                )

    def _detect_leakage_candidates(
        self, df: pd.DataFrame, findings: list[Finding]
    ) -> None:
        all_patterns = LEAKAGE_PATTERNS + self.cfg.leakage_blacklist
        leakage_cols: list[str] = []
        for col in df.columns:
            col_lower = col.lower()
            for pattern in all_patterns:
                if pattern.lower() in col_lower:
                    leakage_cols.append(col)
                    break
        if leakage_cols:
            findings.append(
                Finding(
                    level="warning",
                    code="LEAKAGE_CANDIDATE_COLUMNS",
                    message=(
                        f"Columns {leakage_cols} match leakage-candidate patterns. "
                        "Ensure these are excluded from model observation features."
                    ),
                    details={"columns": leakage_cols},
                )
            )

    # ------------------------------------------------------------------
    # Profile builder
    # ------------------------------------------------------------------

    def _build_profile(self, df: pd.DataFrame) -> dict[str, Any]:
        """Build a compact profile of the dataset."""
        profile: dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
        }

        if "patient_id" in df.columns:
            profile["unique_patient_count"] = int(df["patient_id"].nunique())
            visits_per_patient = df.groupby("patient_id").size()
            profile["visits_per_patient"] = {
                "min": int(visits_per_patient.min()),
                "max": int(visits_per_patient.max()),
                "median": float(visits_per_patient.median()),
                "mean": round(float(visits_per_patient.mean()), 2),
            }
        else:
            profile["unique_patient_count"] = None

        if "visit_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["visit_date"]):
            profile["date_range"] = {
                "min": str(df["visit_date"].min().date()),
                "max": str(df["visit_date"].max().date()),
            }

        # Target distribution
        profile["target_present"] = "target" in df.columns
        if profile["target_present"] and "patient_id" in df.columns:
            pt = df.drop_duplicates("patient_id")[["patient_id", "target"]]
            n_pos = int((pt["target"] == 1).sum())
            n_neg = int((pt["target"] == 0).sum())
            n_null = int(pt["target"].isna().sum())
            profile["positive_patient_count"] = n_pos
            profile["negative_patient_count"] = n_neg
            profile["unknown_target_count"] = n_null
            total_labeled = n_pos + n_neg
            profile["positive_rate"] = round(n_pos / total_labeled, 4) if total_labeled > 0 else None

        # Missingness per feature
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        missingness: dict[str, float] = {}
        for col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                missingness[col] = round(n_null / len(df), 4)
        profile["missingness"] = missingness

        # Duplicate count
        if "patient_id" in df.columns and "visit_date" in df.columns:
            profile["duplicate_patient_date_count"] = int(
                df.duplicated(subset=["patient_id", "visit_date"]).sum()
            )

        # Leakage candidate columns
        all_patterns = LEAKAGE_PATTERNS + []
        profile["leakage_candidate_columns"] = [
            c for c in df.columns
            if any(p.lower() in c.lower() for p in all_patterns)
        ]

        return profile
