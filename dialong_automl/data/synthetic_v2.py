"""Synthetic data scenarios for DiaLong-AutoML (Phase 13D).

Two scenarios are available through :func:`generate_scenario`:

``legacy_v1``
    The original generator (:mod:`dialong_automl.data.synthetic_generator`,
    version 1.1.0), called unchanged. Reproduces every historical dataset.

``v2_standard``
    A continuous-time generator designed in Phase 13A to remove the main
    weaknesses of ``legacy_v1`` (near-deterministic labels, cosmetic visit
    spacing, MCAR-only missingness, no unmeasured outcomes, a single latent
    regime). It implements items G1-G7 of the Phase 13A plan, plus coupled
    glucose (G8) and variable-length histories (G9):

    * G1 cohort size and eligible prevalence controlled by documented parameters.
    * G2 ground-truth sidecar (separate DataFrames / files, never part of the
      model-facing table) and an approximate latent-state oracle probability.
    * G3 HbA1c defined in calendar time; right-skewed, past-dependent visit gaps.
    * G4 patient-specific continuous progression rates from an overlapping
      mixture of subgroups (stable / slow / fast / improver).
    * G5 biological variability (Ornstein-Uhlenbeck) separate from assay error;
      HbA1c reported to 0.1 %.
    * G6 outcome uncertainty: diagnosis is recorded from a single measured
      HbA1c >= 6.5 % OR fasting glucose >= 126 mg/dL, both noisy.
    * G7 MCAR / MAR / MNAR missingness, applied to observation AND future visits.
    * G8 fasting glucose derived from the latent glycaemic level (ADAG relation).
    * G9 2-7 observation visits, so the cohort contains padded sequences.

    Not implemented (Phase 13A Tier 2/3): nonlinear progression (G10),
    time-varying treatment (G11), named phenotypes (G12).

All parameters of ``v2_standard`` live in the frozen dataclass :class:`V2Params`
and are recorded (with a hash) in the metadata. They were fixed before any model
comparison; change them only by defining a NEW scenario, never in place.

Leakage design
--------------
* Each patient gets three independent random streams (observation, future,
  oracle). Observation rows are generated before, and independently of, the
  future process, so no future quantity can influence an observation value.
* Visit scheduling and missingness depend only on the past and the present
  (MNAR depends on the current, unobserved true value), never on the future.
* Ground truth is returned separately and is written to a ``truth/`` subfolder
  that no loader reads.

DISCLAIMER: synthetic demonstration data only. Not clinical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dialong_automl.data.synthetic_generator import (
    DISCLAIMER,
    FULL_COLUMNS,
    GeneratorConfig,
    SyntheticGenerator,
)
from dialong_automl.data.synthetic_generator import (
    GENERATOR_VERSION as LEGACY_GENERATOR_VERSION,
)

GENERATOR_V2_VERSION = "2.0.0"
SCENARIOS: tuple[str, ...] = ("legacy_v1", "v2_standard")
MISSINGNESS_MODES: tuple[str, ...] = ("mcar", "mar", "mnar")

_LABS = ("ldl_cholesterol", "hdl_cholesterol", "triglycerides")
_VITALS = ("bmi", "systolic_bp", "diastolic_bp", "heart_rate")


@dataclass(frozen=True)
class V2Params:
    """Frozen, documented parameters of the ``v2_standard`` scenario.

    Units: HbA1c in %, glucose in mg/dL, rates in % per year, times in days
    unless stated. Values are illustrative design choices for a synthetic
    benchmark; they are not calibrated to any real population.
    """

    # --- population -------------------------------------------------------
    age_min: float = 30.0
    age_max: float = 75.0
    p_female: float = 0.5
    p_family_history: float = 0.30
    smoking: tuple[tuple[str, float], ...] = (("never", 0.55), ("former", 0.25), ("current", 0.20))
    activity: tuple[tuple[str, float], ...] = (
        ("sedentary", 0.35), ("moderate", 0.40), ("active", 0.25),
    )
    bmi_mean: float = 28.0
    bmi_sd: float = 4.5

    # --- baseline HbA1c: base = mean + risk_coef * risk + N(0, sd) ------
    hba1c_base_mean: float = 5.35
    hba1c_base_risk_coef: float = 0.25
    hba1c_base_sd: float = 0.25
    hba1c_base_min: float = 4.3
    hba1c_base_max: float = 7.5

    # --- G4 progression subgroups: (name, logit_intercept, logit_risk_slope,
    #     rate_mean, rate_sd) with rate in % HbA1c per year ----------------
    subgroups: tuple[tuple[str, float, float, float, float], ...] = (
        ("stable", 0.0, 0.0, 0.00, 0.05),
        ("slow", -0.3, 0.6, 0.15, 0.07),
        ("fast", -1.8, 0.8, 0.45, 0.15),
        ("improver", -1.5, -0.3, -0.15, 0.07),
    )
    rate_risk_coef: float = 0.05
    metformin_logit_intercept: float = -2.5
    metformin_logit_risk: float = 0.8
    metformin_rate_effect: float = -0.10

    # --- G5 noise -----------------------------------------------------------
    bio_sd: float = 0.15
    bio_tau_years: float = 0.5
    assay_sd: float = 0.10
    hba1c_resolution: float = 0.1

    # --- G8 glucose: mean = slope * HbA1c + intercept + fasting_offset ---
    glucose_slope: float = 28.7
    glucose_intercept: float = -46.7
    glucose_fasting_offset: float = -15.0
    glucose_bio_sd: float = 8.0
    glucose_assay_sd: float = 3.0

    # --- G6 diagnosis criteria (single measurement) ------------------------
    dx_hba1c: float = 6.5
    dx_glucose: float = 126.0

    # --- G3 / G9 visits ------------------------------------------------------
    n_obs_min: int = 2
    n_obs_max: int = 7
    gap_median_days: float = 150.0
    gap_sigma: float = 0.45
    gap_min_days: int = 30
    gap_max_days: int = 540
    gap_factor_if_high: float = 0.7
    gap_high_hba1c: float = 6.0
    followup_days: int = 730
    max_future_visits: int = 8
    dropout_prob: float = 0.12

    # --- G7 missingness -----------------------------------------------------
    missingness_mode: str = "mar"
    p_hba1c: float = 0.80
    p_hba1c_low: float = 0.65
    p_hba1c_high: float = 0.90
    mar_hba1c_cut: float = 5.7
    mnar_slope: float = 2.0
    mnar_center: float = 5.8
    p_glucose: float = 0.90
    p_lipids: float = 0.75
    p_vitals: float = 0.97

    # --- outcome label (pre-labeled ``target`` column and oracle) ---------
    label_horizon_days: int = 365

    def __post_init__(self) -> None:
        if self.missingness_mode not in MISSINGNESS_MODES:
            raise ValueError(
                f"missingness_mode must be one of {MISSINGNESS_MODES}, "
                f"got {self.missingness_mode!r}"
            )
        if not 2 <= self.n_obs_min <= self.n_obs_max:
            raise ValueError("need 2 <= n_obs_min <= n_obs_max")
        for name, p in (("p_hba1c", self.p_hba1c), ("p_glucose", self.p_glucose),
                        ("p_lipids", self.p_lipids), ("p_vitals", self.p_vitals),
                        ("dropout_prob", self.dropout_prob)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {p}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def params_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class ScenarioResult:
    """Generated data. ``full_df`` is the ONLY model-facing table."""

    full_df: pd.DataFrame
    metadata: dict[str, Any]
    truth_patients: pd.DataFrame | None = None
    truth_visits: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Write the scenario to a NEW directory; refuses to overwrite.

        Layout: ``diabetes_progression_full.csv``, ``generation_metadata.json``
        and, for v2, ``truth/patients_truth.csv`` + ``truth/visits_truth.csv``.
        """
        out = Path(output_dir)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty directory {out}")
        out.mkdir(parents=True, exist_ok=True)
        paths = {"full_csv": out / "diabetes_progression_full.csv",
                 "metadata_json": out / "generation_metadata.json"}
        self.full_df.to_csv(paths["full_csv"], index=False)
        with paths["metadata_json"].open("w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2, default=str)
        if self.truth_patients is not None:
            truth = out / "truth"
            truth.mkdir()
            paths["truth_patients"] = truth / "patients_truth.csv"
            paths["truth_visits"] = truth / "visits_truth.csv"
            self.truth_patients.to_csv(paths["truth_patients"], index=False)
            if self.truth_visits is not None:
                self.truth_visits.to_csv(paths["truth_visits"], index=False)
        return paths


def dataframe_sha256(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame (CSV serialisation, no index)."""
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_scenario(
    scenario: str,
    n_patients: int,
    seed: int,
    compute_oracle: bool = True,
    oracle_draws: int = 32,
    params: V2Params | None = None,
) -> ScenarioResult:
    """Generate a synthetic dataset for a named scenario (in memory)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}.")
    if n_patients < 1:
        raise ValueError("n_patients must be >= 1")
    if scenario == "legacy_v1":
        if params is not None:
            raise ValueError("legacy_v1 does not take V2Params.")
        legacy_cfg = GeneratorConfig(n_patients=n_patients, seed=seed)
        res = SyntheticGenerator(config=legacy_cfg).generate()
        meta = dict(res.metadata)
        meta.update({"scenario": "legacy_v1", "generator": "synthetic_generator.SyntheticGenerator",
                     "generator_version": LEGACY_GENERATOR_VERSION,
                     "dataset_sha256": dataframe_sha256(res.full_df),
                     "ground_truth_available": False})
        return ScenarioResult(full_df=res.full_df, metadata=meta,
                              notes=["legacy_v1 stores no latent truth; no oracle ceiling."])
    return _generate_v2(n_patients, seed, params or V2Params(), compute_oracle, oracle_draws)


# ---------------------------------------------------------------------------
# v2_standard implementation
# ---------------------------------------------------------------------------

@dataclass
class _Patient:
    pid: str
    age0: float
    sex: str
    family_history: str
    smoking: str
    activity: str
    metformin: str
    risk: float
    subgroup: str
    hba1c_base: float
    rate: float           # % per year
    bmi0: float
    bmi_rate: float
    sys0: float
    ldl0: float
    hdl0: float
    tg0: float
    hr0: float
    start: pd.Timestamp


@dataclass
class _State:
    t_days: float = 0.0
    bio: float = 0.0
    last_measured_hba1c: float | None = None
    diagnosed: bool = False
    hypertension: bool = False
    n_visits: int = 0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _choice(rng: np.random.Generator, options: tuple[tuple[str, float], ...]) -> str:
    names = [o[0] for o in options]
    probs = np.array([o[1] for o in options], dtype=float)
    return str(rng.choice(names, p=probs / probs.sum()))


def _draw_patient(pid: str, p: V2Params, rng: np.random.Generator) -> _Patient:
    age0 = float(rng.uniform(p.age_min, p.age_max))
    sex = "F" if rng.random() < p.p_female else "M"
    fh = "yes" if rng.random() < p.p_family_history else "no"
    smoking = _choice(rng, p.smoking)
    activity = _choice(rng, p.activity)
    z = float(rng.normal())
    bmi0 = float(np.clip(rng.normal(p.bmi_mean + 1.5 * z, p.bmi_sd), 17.0, 50.0))
    risk = (z + 0.5 * (fh == "yes") + 0.3 * (smoking == "current")
            + 0.3 * (activity == "sedentary") - 0.3 * (activity == "active")
            + 0.05 * (bmi0 - p.bmi_mean) + 0.02 * (age0 - 50.0))
    base = float(np.clip(
        p.hba1c_base_mean + p.hba1c_base_risk_coef * risk + rng.normal(0.0, p.hba1c_base_sd),
        p.hba1c_base_min, p.hba1c_base_max,
    ))
    logits = np.array([a + b * risk for _, a, b, _, _ in p.subgroups])
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    k = int(rng.choice(len(p.subgroups), p=probs))
    name, _, _, r_mean, r_sd = p.subgroups[k]
    metformin = "yes" if rng.random() < _sigmoid(
        p.metformin_logit_intercept + p.metformin_logit_risk * risk) else "no"
    rate = float(rng.normal(r_mean, r_sd)) + p.rate_risk_coef * risk
    if metformin == "yes":
        rate += p.metformin_rate_effect
    start = pd.Timestamp("2018-01-01") + pd.Timedelta(int(rng.integers(0, 730)), unit="D")
    return _Patient(
        pid=pid, age0=age0, sex=sex, family_history=fh, smoking=smoking, activity=activity,
        metformin=metformin, risk=risk, subgroup=name, hba1c_base=base, rate=rate,
        bmi0=bmi0, bmi_rate=float(rng.normal(0.1 * z, 0.2)),
        sys0=float(rng.normal(118 + 4 * risk + 0.4 * (age0 - 50), 10)),
        ldl0=float(rng.normal(115 + 8 * risk, 25)), hdl0=float(rng.normal(52 - 5 * risk, 10)),
        tg0=float(rng.normal(130 + 30 * risk, 40)), hr0=float(rng.normal(72, 9)), start=start,
    )


def _next_gap(state: _State, p: V2Params, rng: np.random.Generator) -> float:
    """Days to the next visit. Depends only on the LAST MEASURED HbA1c (past)."""
    gap = p.gap_median_days * math.exp(p.gap_sigma * float(rng.normal()))
    if state.last_measured_hba1c is not None and state.last_measured_hba1c >= p.gap_high_hba1c:
        gap *= p.gap_factor_if_high
    return float(np.clip(round(gap), p.gap_min_days, p.gap_max_days))


def _p_measure_hba1c(state: _State, true_hba1c: float, p: V2Params) -> float:
    if p.missingness_mode == "mcar":
        return p.p_hba1c
    if p.missingness_mode == "mar":   # depends on the previous measurement only
        prev = state.last_measured_hba1c
        return p.p_hba1c_high if (prev is not None and prev >= p.mar_hba1c_cut) else p.p_hba1c_low
    # mnar: depends on the current (unobserved) true value
    logit = math.log(p.p_hba1c / (1 - p.p_hba1c)) + p.mnar_slope * (true_hba1c - p.mnar_center)
    return _sigmoid(logit)


def _visit(pt: _Patient, state: _State, dt_days: float, p: V2Params,
           rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
    """Advance the latent state by ``dt_days`` and emit one visit.

    The same number of random draws is consumed whether or not values are
    observed, so missingness never shifts the random stream.
    Returns (model-facing row, truth row). Mutates ``state``.
    """
    if state.n_visits == 0:
        state.bio = float(rng.normal(0.0, p.bio_sd))
    else:
        decay = math.exp(-(dt_days / 365.25) / p.bio_tau_years)
        state.bio = state.bio * decay + math.sqrt(1 - decay ** 2) * p.bio_sd * float(rng.normal())
    state.t_days += dt_days
    years = state.t_days / 365.25
    true_a1c = pt.hba1c_base + pt.rate * years + state.bio
    a1c_value = round(round((true_a1c + float(rng.normal(0, p.assay_sd))) / p.hba1c_resolution)
                      * p.hba1c_resolution, 2)
    true_glu = (p.glucose_slope * true_a1c + p.glucose_intercept + p.glucose_fasting_offset
                + float(rng.normal(0, p.glucose_bio_sd)))
    glu_value = round(true_glu + float(rng.normal(0, p.glucose_assay_sd)), 1)

    sys = pt.sys0 + 0.4 * years + float(rng.normal(0, 6))
    vitals = {
        "bmi": round(float(np.clip(pt.bmi0 + pt.bmi_rate * years + rng.normal(0, 0.4), 15, 60)), 2),
        "systolic_bp": round(float(np.clip(sys, 85, 210)), 1),
        "diastolic_bp": round(float(np.clip(0.6 * sys + 5 + rng.normal(0, 5), 45, 130)), 1),
        "heart_rate": round(float(np.clip(pt.hr0 + rng.normal(0, 4), 40, 130)), 1),
    }
    labs = {
        "ldl_cholesterol": round(float(np.clip(pt.ldl0 + rng.normal(0, 8), 40, 260)), 1),
        "hdl_cholesterol": round(float(np.clip(pt.hdl0 + rng.normal(0, 3), 20, 110)), 1),
        "triglycerides": round(float(np.clip(pt.tg0 + rng.normal(0, 15), 40, 600)), 1),
    }
    u_a1c, u_glu, u_lab, u_vit = (float(rng.random()) for _ in range(4))
    a1c_obs = u_a1c < _p_measure_hba1c(state, true_a1c, p)
    glu_obs = u_glu < p.p_glucose
    lab_obs = u_lab < p.p_lipids
    vit_obs = u_vit < p.p_vitals

    hba1c = a1c_value if a1c_obs else float("nan")
    glucose = glu_value if glu_obs else float("nan")
    if a1c_obs:
        state.last_measured_hba1c = a1c_value
    if (a1c_obs and a1c_value >= p.dx_hba1c) or (glu_obs and glu_value >= p.dx_glucose):
        state.diagnosed = True
    if vit_obs and vitals["systolic_bp"] >= 140:
        state.hypertension = True
    state.n_visits += 1

    visit_date = pt.start + pd.Timedelta(int(round(state.t_days)), unit="D")
    row: dict[str, Any] = {
        "patient_id": pt.pid,
        "visit_date": visit_date,
        "visit_number": state.n_visits,
        "hba1c": hba1c,
        "fasting_glucose": glucose,
        **{k: (v if vit_obs else float("nan")) for k, v in vitals.items()},
        **{k: (v if lab_obs else float("nan")) for k, v in labs.items()},
        "age_at_visit": round(pt.age0 + years, 2),
        "sex": pt.sex,
        "smoking_status": pt.smoking,
        "physical_activity_level": pt.activity,
        "family_history_diabetes": pt.family_history,
        "metformin_use": pt.metformin,
        "hypertension_diagnosis": "yes" if state.hypertension else "no",
        "diabetes_diagnosis": int(state.diagnosed),
    }
    truth = {"patient_id": pt.pid, "visit_date": visit_date, "true_hba1c": round(true_a1c, 4),
             "true_fasting_glucose": round(true_glu, 2), "hba1c_measured": bool(a1c_obs)}
    return row, truth


def _copy_state(s: _State) -> _State:
    return _State(**vars(s))


def _future_label(rows: list[dict[str, Any]], index_days: float, start: pd.Timestamp,
                  p: V2Params) -> tuple[int, bool]:
    """(label, ascertained) using the cohort rule within the label horizon."""
    label, ascertained = 0, False
    for r in rows:
        days = (r["visit_date"] - start).days - index_days
        if days <= 0 or days > p.label_horizon_days:
            continue
        measured = not math.isnan(r["hba1c"])
        if r["diabetes_diagnosis"] == 1 or (measured and r["hba1c"] >= p.dx_hba1c):
            label = 1
        ascertained = ascertained or measured or r["diabetes_diagnosis"] == 1
    return label, ascertained


def _simulate_future(pt: _Patient, state: _State, p: V2Params, rng: np.random.Generator,
                     allow_dropout: bool = True) -> tuple[list, list, bool]:
    dropout = allow_dropout and float(rng.random()) < p.dropout_prob
    rows, truths = [], []
    if dropout:
        return rows, truths, True
    index_days = state.t_days
    for _ in range(p.max_future_visits):
        gap = _next_gap(state, p, rng)
        if state.t_days + gap - index_days > p.followup_days:
            break
        row, truth = _visit(pt, state, gap, p, rng)
        rows.append(row)
        truths.append(truth)
    return rows, truths, False


def _generate_v2(n_patients: int, seed: int, p: V2Params, compute_oracle: bool,
                 oracle_draws: int) -> ScenarioResult:
    root = np.random.SeedSequence(seed)
    rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    patient_truth: list[dict[str, Any]] = []
    for i, child in enumerate(root.spawn(n_patients)):
        obs_ss, fut_ss, orc_ss = child.spawn(3)
        obs_rng = np.random.default_rng(obs_ss)
        pid = f"SYN2_{i + 1:05d}"
        pt = _draw_patient(pid, p, obs_rng)
        state = _State()

        # --- observation window (observation stream only) ---
        n_obs = int(obs_rng.integers(p.n_obs_min, p.n_obs_max + 1))
        obs_rows = []
        for v in range(n_obs):
            gap = 0.0 if v == 0 else _next_gap(state, p, obs_rng)
            row, truth = _visit(pt, state, gap, p, obs_rng)
            row["is_future_visit"] = False
            truth["is_future_visit"] = False
            obs_rows.append(row)
            truth_rows.append(truth)
        index_days = state.t_days
        index_state = _copy_state(state)
        index_true = truth_rows[-1]["true_hba1c"]

        # --- future (future stream only; depends on, never feeds, the past) ---
        fut_rows, fut_truth, dropout = _simulate_future(pt, state, p, np.random.default_rng(fut_ss))
        for r, t in zip(fut_rows, fut_truth):
            r["is_future_visit"] = True
            t["is_future_visit"] = True
        truth_rows.extend(fut_truth)
        label, ascertained = _future_label(fut_rows, index_days, pt.start, p)
        target = label if fut_rows else None
        for r in obs_rows + fut_rows:
            r["target"] = target
        rows.extend(obs_rows + fut_rows)

        # --- G2: latent-state oracle (Monte Carlo over the future process) ---
        oracle = float("nan")
        if compute_oracle and oracle_draws > 0:
            orc_rng = np.random.default_rng(orc_ss)
            labels, asc = [], []
            for _ in range(oracle_draws):
                sim_rows, _, _ = _simulate_future(pt, _copy_state(index_state), p, orc_rng,
                                                  allow_dropout=False)
                lab, a = _future_label(sim_rows, index_days, pt.start, p)
                labels.append(lab)
                asc.append(a)
            labels_arr, asc_arr = np.array(labels), np.array(asc)
            ascertained_labels = labels_arr[asc_arr] if asc_arr.any() else labels_arr
            oracle = float(ascertained_labels.mean())

        patient_truth.append({
            "patient_id": pid, "subgroup": pt.subgroup, "risk_score": round(pt.risk, 4),
            "hba1c_base": round(pt.hba1c_base, 4), "hba1c_rate_per_year": round(pt.rate, 4),
            "metformin_use": pt.metformin, "n_obs_visits_generated": n_obs,
            "n_future_visits": len(fut_rows), "dropout": dropout,
            "index_date": pt.start + pd.Timedelta(int(round(index_days)), unit="D"),
            "true_hba1c_at_index": index_true, "diagnosed_by_index": index_state.diagnosed,
            "label_generator_rule": target, "outcome_ascertained": ascertained,
            "oracle_prob": oracle,
        })

    full = pd.DataFrame(rows)
    full["visit_date"] = pd.to_datetime(full["visit_date"])
    full = full[FULL_COLUMNS].sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    truth_p = pd.DataFrame(patient_truth)
    truth_v = (pd.DataFrame(truth_rows).sort_values(["patient_id", "visit_date"])
               .reset_index(drop=True))

    obs = full[~full["is_future_visit"]]
    fut = full[full["is_future_visit"]]
    meta = {
        "disclaimer": DISCLAIMER,
        "scenario": "v2_standard",
        "generator": "synthetic_v2._generate_v2",
        "generator_version": GENERATOR_V2_VERSION,
        "seed": seed,
        "n_patients": n_patients,
        "params": p.to_dict(),
        "params_hash": p.params_hash(),
        "dataset_sha256": dataframe_sha256(full),
        "ground_truth_available": True,
        "oracle": {"computed": bool(compute_oracle and oracle_draws > 0), "draws": oracle_draws,
                   "definition": "P(label=1 | latent state at index, ascertained), Monte Carlo "
                                 "over the future visit/measurement process"},
        "realised": {
            "total_rows": int(len(full)),
            "observation_rows": int(len(obs)),
            "future_rows": int(len(fut)),
            "dropout_patients": int(truth_p["dropout"].sum()),
            "hba1c_missing_rate_observation": round(float(obs["hba1c"].isna().mean()), 4),
            "hba1c_missing_rate_future": (round(float(fut["hba1c"].isna().mean()), 4)
                                          if len(fut) else None),
            "generator_label_prevalence": round(
                float(truth_p["label_generator_rule"].dropna().mean()), 4),
        },
        "target_definition": (
            f"target=1 iff a future visit within {p.label_horizon_days} days of the last "
            f"observation visit has diabetes_diagnosis==1 OR hba1c >= {p.dx_hba1c}; "
            "diagnosis is recorded when a measured HbA1c >= 6.5 % or fasting glucose >= 126 mg/dL. "
            "Same rule as CohortBuilder FUTURE_OUTCOME."
        ),
    }
    return ScenarioResult(full_df=full, metadata=meta, truth_patients=truth_p, truth_visits=truth_v)
