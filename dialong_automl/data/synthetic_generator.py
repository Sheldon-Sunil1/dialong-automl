"""Synthetic longitudinal diabetes progression data generator.

DISCLAIMER
----------
All data produced by this module is **synthetic demonstration data**.
It is NOT based on real patients, NOT clinical evidence, and NOT suitable
for medical validation, diagnosis, or treatment decisions.

Design
------
For each patient the generator creates:
  1. A static baseline risk profile (age, sex, family history, …).
  2. A temporal trajectory covering observation visits PLUS hidden future
     visits that are used only to derive the outcome label.

Observation visits ← model may see these
Future visits      ← used ONLY for target construction, never as features

The full timeline is written to ``diabetes_progression_full.csv``.
A trimmed version (observation visits only) is written to
``diabetes_progression_demo.csv``.  Both files carry the disclaimer header.

Target definition (canonical)
------------------------------
The single canonical rule, shared by BOTH the generator and the cohort
builder, is:

    target = 1  if, in any future visit (after the observation cutoff and
                within the configured prediction horizon):
                    diabetes_diagnosis == 1
                OR  hba1c >= diabetes_onset_hba1c_threshold (default 6.5 %)

    target = 0  otherwise (when sufficient future follow-up exists and
                neither criterion is met in any future visit)

``diabetes_diagnosis`` in the generated data is set to 1 at the first
future visit where HbA1c crosses the threshold (for a progressor) or
remains 0 for non-progressors.  It is NOT gated by a separate hidden
``will_progress`` coin-flip — the observed future data is the ground truth.

The hidden ``will_progress`` variable controls only the HbA1c *trajectory
slope* (progressors get a steeper positive slope) so that the two groups
have different observable patterns.  Whether a patient's future HbA1c
actually crosses 6.5 during the prediction horizon determines the label.

This guarantees that:
    generator pre-labeled target  ==  cohort builder future-outcome target
for the same patient and horizon configuration.

The ``progression_prevalence`` config parameter tunes the fraction of
patients whose trajectory slope is set to a rising pattern; the actual
realised prevalence will be close but not identical due to individual noise.

Usage::

    from dialong_automl.data.synthetic_generator import SyntheticGenerator

    gen = SyntheticGenerator(n_patients=500, seed=42)
    result = gen.generate()
    result.save(output_dir=Path("data/synthetic"))
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENERATOR_VERSION = "1.1.0"  # bumped: target definition aligned with cohort builder
DISCLAIMER = (
    "Synthetic demonstration data. "
    "Not clinical evidence. "
    "Not suitable for medical validation."
)

NUMERIC_FEATURES = [
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
]

CATEGORICAL_FEATURES = [
    "sex",
    "smoking_status",
    "physical_activity_level",
    "family_history_diabetes",
    "metformin_use",
    "hypertension_diagnosis",
]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Canonical diabetes-onset HbA1c threshold (matches CohortConfig default)
DIABETES_HBA1C_THRESHOLD: float = 6.5

# Columns included in the full output
FULL_COLUMNS = (
    ["patient_id", "visit_date", "visit_number", "is_future_visit"]
    + ALL_FEATURE_COLUMNS
    + ["diabetes_diagnosis", "target"]
)

# Columns included in the observation-only output (future visits stripped)
OBS_COLUMNS = (
    ["patient_id", "visit_date", "visit_number"]
    + ALL_FEATURE_COLUMNS
    + ["diabetes_diagnosis", "target"]
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    """Configuration for :class:`SyntheticGenerator`.

    Parameters
    ----------
    n_patients:
        Number of synthetic patients to generate.
    min_obs_visits:
        Minimum number of observation-window visits per patient.
    max_obs_visits:
        Maximum number of observation-window visits per patient.
    min_future_visits:
        Minimum number of future visits generated (used for outcome only).
    max_future_visits:
        Maximum number of future visits generated (used for outcome only).
    min_days_between_visits:
        Lower bound of the inter-visit interval in days.
    max_days_between_visits:
        Upper bound of the inter-visit interval in days.
    progression_prevalence:
        Approximate fraction of patients assigned a *rising* HbA1c
        trajectory (i.e., progressors).  The realised label prevalence
        will be close to but not necessarily equal to this value, because
        whether the HbA1c actually crosses the threshold within the
        prediction horizon depends on the individual trajectory noise.
    missingness_rate:
        Fraction of individual numeric feature values set to NaN in the
        observation window (missingness is NOT applied to future visits).
    seed:
        Random seed for reproducibility.
    diabetes_onset_hba1c_threshold:
        HbA1c threshold used for both ``diabetes_diagnosis`` generation
        and pre-labeled target derivation.  Must match the value used in
        :class:`~dialong_automl.data.cohort.CohortConfig` for the two
        targets to be consistent.
    """

    n_patients: int = 1000
    min_obs_visits: int = 4
    max_obs_visits: int = 6
    min_future_visits: int = 2
    max_future_visits: int = 4
    min_days_between_visits: int = 90
    max_days_between_visits: int = 240
    progression_prevalence: float = 0.30
    missingness_rate: float = 0.05
    seed: int = 42
    diabetes_onset_hba1c_threshold: float = DIABETES_HBA1C_THRESHOLD


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Container for generated data and metadata.

    Attributes
    ----------
    full_df:
        Complete timeline including future visits (``is_future_visit=True``).
    obs_df:
        Observation-window visits only (``is_future_visit=False``).
        The ``target`` column here is the pre-labeled target, which is
        derived from future observations using the same rule as the cohort
        builder's future-outcome mode.
    metadata:
        Dict with generation parameters, counts, and disclaimer.
    """

    full_df: pd.DataFrame
    obs_df: pd.DataFrame
    metadata: dict[str, Any]

    def save(self, output_dir: Path | str) -> dict[str, Path]:
        """Save both CSVs and metadata JSON to *output_dir*.

        Parameters
        ----------
        output_dir:
            Directory to write files into.  Created if absent.

        Returns
        -------
        dict[str, Path]
            Mapping of ``"full_csv"``, ``"obs_csv"``, ``"metadata_json"``
            to the paths written.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        full_path = out / "diabetes_progression_full.csv"
        obs_path = out / "diabetes_progression_demo.csv"
        meta_path = out / "generation_metadata.json"

        self.full_df.to_csv(full_path, index=False)
        self.obs_df.to_csv(obs_path, index=False)
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2, default=str)

        logger.info("Saved full CSV (%d rows) -> %s", len(self.full_df), full_path)
        logger.info("Saved observation CSV (%d rows) -> %s", len(self.obs_df), obs_path)
        logger.info("Saved metadata -> %s", meta_path)
        return {"full_csv": full_path, "obs_csv": obs_path, "metadata_json": meta_path}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


class SyntheticGenerator:
    """Generate synthetic longitudinal diabetes progression data.

    Parameters
    ----------
    n_patients:
        Number of patients.  Overrides ``config.n_patients`` when provided.
    seed:
        Random seed.  Overrides ``config.seed`` when provided.
    config:
        Full :class:`GeneratorConfig`; keyword arguments take precedence.
    **kwargs:
        Additional :class:`GeneratorConfig` fields to override.
    """

    def __init__(
        self,
        n_patients: int | None = None,
        seed: int | None = None,
        config: GeneratorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.cfg = config or GeneratorConfig()
        if n_patients is not None:
            self.cfg.n_patients = n_patients
        if seed is not None:
            self.cfg.seed = seed
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
            else:
                raise ValueError(f"Unknown GeneratorConfig field: {k!r}")

        self._rng = np.random.default_rng(self.cfg.seed)
        random.seed(self.cfg.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> GenerationResult:
        """Generate the full synthetic dataset.

        Returns
        -------
        GenerationResult
            Contains ``full_df``, ``obs_df``, and ``metadata``.
        """
        logger.info(
            "Generating %d synthetic patients (seed=%d, target_prevalence~=%.2f)",
            self.cfg.n_patients,
            self.cfg.seed,
            self.cfg.progression_prevalence,
        )

        all_rows: list[dict[str, Any]] = []

        for i in range(self.cfg.n_patients):
            patient_id = f"SYN_{i + 1:05d}"
            patient_rows = self._generate_patient(patient_id)
            all_rows.extend(patient_rows)

        full_df = pd.DataFrame(all_rows)
        full_df["visit_date"] = pd.to_datetime(full_df["visit_date"])
        full_df = full_df.sort_values(["patient_id", "visit_date"]).reset_index(drop=True)

        # Observation-only subset
        obs_df = (
            full_df[~full_df["is_future_visit"]]
            [OBS_COLUMNS]
            .reset_index(drop=True)
        )

        # Compute actual statistics
        n_obs_patients = obs_df["patient_id"].nunique()
        n_positive = int(obs_df.drop_duplicates("patient_id")["target"].sum())
        actual_prevalence = float(n_positive / n_obs_patients) if n_obs_patients > 0 else 0.0

        # Missingness
        numeric_cols_present = [c for c in NUMERIC_FEATURES if c in obs_df.columns]
        total_numeric_cells = len(obs_df) * len(numeric_cols_present)
        missing_cells = int(obs_df[numeric_cols_present].isna().sum().sum())
        actual_missingness = missing_cells / total_numeric_cells if total_numeric_cells > 0 else 0.0

        visits_per_patient = obs_df.groupby("patient_id").size()

        metadata: dict[str, Any] = {
            "disclaimer": DISCLAIMER,
            "generator_version": GENERATOR_VERSION,
            "seed": self.cfg.seed,
            "intended_n_patients": self.cfg.n_patients,
            "actual_n_patients": n_obs_patients,
            "total_observation_rows": len(obs_df),
            "total_full_rows": len(full_df),
            "intended_progression_prevalence": self.cfg.progression_prevalence,
            "actual_progression_prevalence": round(actual_prevalence, 4),
            "n_positive_patients": n_positive,
            "n_negative_patients": int(n_obs_patients - n_positive),
            "min_obs_visits": self.cfg.min_obs_visits,
            "max_obs_visits": self.cfg.max_obs_visits,
            "min_future_visits": self.cfg.min_future_visits,
            "max_future_visits": self.cfg.max_future_visits,
            "visits_per_patient_min": int(visits_per_patient.min()),
            "visits_per_patient_max": int(visits_per_patient.max()),
            "visits_per_patient_median": float(visits_per_patient.median()),
            "intended_missingness_rate": self.cfg.missingness_rate,
            "actual_missingness_rate": round(actual_missingness, 4),
            "diabetes_onset_hba1c_threshold": self.cfg.diabetes_onset_hba1c_threshold,
            "target_definition": (
                "target=1 iff any future visit has diabetes_diagnosis==1 "
                f"OR hba1c >= {self.cfg.diabetes_onset_hba1c_threshold}; "
                "target=0 otherwise. "
                "Identical to CohortBuilder FUTURE_OUTCOME rule."
            ),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        }

        logger.info(
            "Generation complete: %d patients, %d obs rows, "
            "actual_prevalence=%.3f, missingness=%.3f",
            n_obs_patients, len(obs_df), actual_prevalence, actual_missingness,
        )

        return GenerationResult(full_df=full_df, obs_df=obs_df, metadata=metadata)

    # ------------------------------------------------------------------
    # Patient-level generation
    # ------------------------------------------------------------------

    def _generate_patient(self, patient_id: str) -> list[dict[str, Any]]:
        """Generate all visits (observation + future) for one patient.

        Target derivation rule (canonical — must match CohortBuilder)
        -------------------------------------------------------------
        target = 1  iff any future visit satisfies:
                        diabetes_diagnosis == 1
                    OR  hba1c >= diabetes_onset_hba1c_threshold

        The hidden ``will_progress`` variable governs only the HbA1c
        *trajectory slope*, not whether ``diabetes_diagnosis`` is set or
        whether the target is 1.  This ensures the pre-labeled ``target``
        stamped on observation rows is identical to what the cohort
        builder's future-outcome mode would derive from the same data.
        """
        # --- Static baseline profile ---
        age_baseline = float(self._rng.uniform(35, 75))
        sex = self._rng.choice(["M", "F"])
        family_history = self._rng.choice(["yes", "no"], p=[0.35, 0.65])
        smoking = self._rng.choice(
            ["never", "former", "current"],
            p=[0.50, 0.30, 0.20],
        )
        activity = self._rng.choice(
            ["sedentary", "moderate", "active"],
            p=[0.35, 0.40, 0.25],
        )

        # --- Baseline clinical values ---
        risk_bias = float(self._rng.uniform(0, 1))  # 0 = low risk, 1 = high risk

        hba1c_base = float(np.clip(self._rng.normal(5.2 + 1.0 * risk_bias, 0.4), 4.0, 9.0))
        fasting_glucose_base = float(np.clip(self._rng.normal(88 + 30 * risk_bias, 8), 70, 160))
        bmi_base = float(np.clip(self._rng.normal(25 + 8 * risk_bias, 3), 18, 45))
        systolic_base = float(np.clip(self._rng.normal(118 + 22 * risk_bias, 10), 90, 180))
        diastolic_base = float(np.clip(self._rng.normal(76 + 10 * risk_bias, 7), 55, 110))
        ldl_base = float(np.clip(self._rng.normal(110 + 30 * risk_bias, 20), 50, 220))
        hdl_base = float(np.clip(self._rng.normal(55 - 15 * risk_bias, 10), 25, 100))
        triglycerides_base = float(np.clip(self._rng.normal(120 + 80 * risk_bias, 25), 50, 400))
        heart_rate_base = float(np.clip(self._rng.normal(72, 10), 45, 120))

        hypertension = "yes" if systolic_base >= 130 else "no"
        metformin_p = 0.05 + 0.20 * risk_bias
        metformin = "yes" if self._rng.random() < metformin_p else "no"

        # --- Determine whether this patient is a trajectory progressor ---
        # ``will_progress`` controls the HbA1c *slope* only.  It is calibrated
        # to hit ``progression_prevalence`` in expectation, but the actual label
        # is determined purely by whether future HbA1c crosses the threshold.
        progression_logit = (
            -3.5
            + 3.0 * risk_bias
            + 0.04 * (age_baseline - 50)
            + (0.5 if family_history == "yes" else 0.0)
            + (0.3 if smoking == "current" else 0.0)
            + (-0.4 if activity == "active" else 0.2 if activity == "sedentary" else 0.0)
            + (-0.3 if metformin == "yes" else 0.0)
            + float(self._rng.normal(0, 0.5))
        )
        calibration_offset = math.log(
            self.cfg.progression_prevalence / (1.0 - self.cfg.progression_prevalence + 1e-9)
        ) - (-3.5 + 1.5)
        progression_logit += calibration_offset * 0.4
        progression_prob = 1.0 / (1.0 + math.exp(-progression_logit))
        will_progress = bool(self._rng.random() < progression_prob)

        # --- Trajectory slopes ---
        lifestyle_modifier = (
            -0.3 if activity == "active" else (0.2 if activity == "sedentary" else 0.0)
        )
        smoking_modifier = 0.15 if smoking == "current" else 0.0
        fh_modifier = 0.1 if family_history == "yes" else 0.0
        metformin_modifier = -0.15 if metformin == "yes" else 0.0
        net_modifier = lifestyle_modifier + smoking_modifier + fh_modifier + metformin_modifier

        # Progressors get a steeper rising slope; non-progressors stay flat/decline
        if will_progress:
            hba1c_slope = float(self._rng.normal(0.12 + net_modifier, 0.03))
        else:
            hba1c_slope = float(self._rng.normal(-0.02 + net_modifier * 0.3, 0.03))

        glucose_slope = float(self._rng.normal(1.0 * risk_bias + net_modifier * 3, 0.5))
        bmi_slope = float(self._rng.normal(0.05 * risk_bias + lifestyle_modifier * 0.5, 0.05))
        bp_slope = float(self._rng.normal(0.2 * risk_bias, 0.1))

        # --- Visit counts and dates ---
        n_obs = int(self._rng.integers(self.cfg.min_obs_visits, self.cfg.max_obs_visits + 1))
        n_future = int(self._rng.integers(self.cfg.min_future_visits, self.cfg.max_future_visits + 1))
        n_total = n_obs + n_future

        start_date = pd.Timestamp("2018-01-01") + pd.DateOffset(
            days=int(self._rng.integers(0, 365 * 2))
        )
        visit_dates: list[pd.Timestamp] = [start_date]
        for _ in range(n_total - 1):
            gap = int(self._rng.integers(
                self.cfg.min_days_between_visits,
                self.cfg.max_days_between_visits + 1,
            ))
            visit_dates.append(visit_dates[-1] + pd.DateOffset(days=gap))

        # --- Generate visit rows ---
        rows: list[dict[str, Any]] = []
        threshold = self.cfg.diabetes_onset_hba1c_threshold

        for visit_idx in range(n_total):
            is_future = visit_idx >= n_obs
            visit_date = visit_dates[visit_idx]
            age_at_visit = age_baseline + (visit_date - start_date).days / 365.25
            t = visit_idx

            hba1c = float(np.clip(
                hba1c_base + hba1c_slope * t + self._rng.normal(0, 0.1),
                4.0, 14.0,
            ))
            fasting_glucose = float(np.clip(
                fasting_glucose_base + glucose_slope * t + self._rng.normal(0, 3),
                60, 400,
            ))
            bmi = float(np.clip(
                bmi_base + bmi_slope * t + self._rng.normal(0, 0.3),
                15, 60,
            ))
            systolic = float(np.clip(
                systolic_base + bp_slope * t + self._rng.normal(0, 3),
                85, 200,
            ))
            diastolic = float(np.clip(
                diastolic_base + bp_slope * 0.5 * t + self._rng.normal(0, 2),
                50, 120,
            ))
            ldl = float(np.clip(ldl_base + self._rng.normal(0, 5), 40, 250))
            hdl = float(np.clip(hdl_base + self._rng.normal(0, 3), 20, 110))
            triglycerides = float(np.clip(
                triglycerides_base + self._rng.normal(0, 10), 40, 500,
            ))
            heart_rate = float(np.clip(
                heart_rate_base + self._rng.normal(0, 4), 40, 130,
            ))

            # --- diabetes_diagnosis ---
            # Set to 1 when hba1c >= threshold, regardless of will_progress.
            # This makes the diagnosis observable and the target derivable
            # directly from the data, consistent with the cohort builder.
            # In the observation window, diagnosis remains 0 (the patient
            # has not yet been diagnosed at the time of observation).
            if is_future:
                diabetes_diagnosis = 1 if hba1c >= threshold else 0
            else:
                diabetes_diagnosis = 0

            if systolic >= 140:
                hypertension = "yes"

            rows.append({
                "patient_id": patient_id,
                "visit_date": visit_date,
                "visit_number": visit_idx + 1,
                "is_future_visit": is_future,
                "hba1c": round(hba1c, 2),
                "fasting_glucose": round(fasting_glucose, 1),
                "bmi": round(bmi, 2),
                "systolic_bp": round(systolic, 1),
                "diastolic_bp": round(diastolic, 1),
                "ldl_cholesterol": round(ldl, 1),
                "hdl_cholesterol": round(hdl, 1),
                "triglycerides": round(triglycerides, 1),
                "heart_rate": round(heart_rate, 1),
                "age_at_visit": round(age_at_visit, 2),
                "sex": sex,
                "smoking_status": smoking,
                "physical_activity_level": activity,
                "family_history_diabetes": family_history,
                "metformin_use": metformin,
                "hypertension_diagnosis": hypertension,
                "diabetes_diagnosis": diabetes_diagnosis,
                "target": None,  # filled after all rows are generated
            })

        # --- Derive pre-labeled target using the CANONICAL rule ---
        # Identical to CohortBuilder._derive_future_target():
        #   target = 1 iff any future visit has diabetes_diagnosis==1
        #            OR hba1c >= threshold
        future_rows = [r for r in rows if r["is_future_visit"]]
        if future_rows:
            patient_target: int | None = 1 if any(
                r["diabetes_diagnosis"] == 1 or r["hba1c"] >= threshold
                for r in future_rows
            ) else 0
        else:
            patient_target = None  # no future data — excluded by cohort builder

        for row in rows:
            row["target"] = patient_target

        # --- Apply missingness to OBSERVATION visits only ---
        for row in rows:
            if not row["is_future_visit"]:
                for col in NUMERIC_FEATURES:
                    if self._rng.random() < self.cfg.missingness_rate:
                        row[col] = float("nan")

        return rows
