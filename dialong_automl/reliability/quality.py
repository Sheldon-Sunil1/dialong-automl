"""Input-quality flags (Phase 14D, item R4).

Rules are fixed in the Phase 14A design (section R4) and use the submitted
visits only, never labels or model output. Severity: "severe" and "moderate"
flags affect the reliability level (R6); "info" flags are shown only.

Synthetic-data project. The value bounds are schema sanity bounds mirrored
from ``frontend/src/lib/fields.ts`` (NUMERIC_RANGES), not clinical reference
ranges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

#: Mirror of frontend NUMERIC_RANGES (schema sanity bounds, not clinical ranges).
VALUE_BOUNDS: dict[str, tuple[float, float]] = {
    "hba1c": (3.0, 20.0), "fasting_glucose": (40, 500), "bmi": (10, 70),
    "systolic_bp": (70, 220), "diastolic_bp": (40, 140), "ldl_cholesterol": (20, 300),
    "hdl_cholesterol": (10, 150), "triglycerides": (20, 700), "heart_rate": (30, 160),
    "age_at_visit": (0, 120),
}
SEVERITIES = ("severe", "moderate", "info")


@dataclass(frozen=True)
class Flag:
    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _measured(visits: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in visits.columns:
        return visits.iloc[0:0]
    return visits[pd.to_numeric(visits[col], errors="coerce").notna()]


def quality_flags(visits: pd.DataFrame, max_visits: int,
                  hba1c_age_days: tuple[int, int] = (180, 365)) -> list[Flag]:
    """Flags for ONE patient's observation-window visits (any order)."""
    if visits.empty:
        return [Flag("no_visits", "severe", "No visits were submitted.")]
    v = visits.copy()
    v["visit_date"] = pd.to_datetime(v["visit_date"], errors="coerce")
    latest = v["visit_date"].max()
    flags: list[Flag] = []

    h = _measured(v, "hba1c")
    old_days, stale_days = hba1c_age_days
    if h.empty:
        flags.append(Flag("no_hba1c", "severe", "No HbA1c result in the visits used."))
    else:
        if len(h) == 1:
            flags.append(Flag("single_hba1c", "moderate",
                              "Only one HbA1c result, so no HbA1c trend is available."))
        age = (latest - h["visit_date"].max()).days if pd.notna(latest) else 0
        if age > stale_days:
            flags.append(Flag("stale_hba1c", "severe",
                              f"The latest HbA1c is {age} days older than the latest visit."))
        elif age > old_days:
            flags.append(Flag("old_hba1c", "moderate",
                              f"The latest HbA1c is {age} days older than the latest visit."))
    if _measured(v, "fasting_glucose").empty:
        flags.append(Flag("no_glucose", "moderate",
                          "No fasting glucose result in the visits used."))
    if len(v) < max_visits:
        flags.append(Flag("short_history", "info",
                          f"{len(v)} of up to {max_visits} visits were available."))
    bad = []
    for col, (lo, hi) in VALUE_BOUNDS.items():
        if col in v.columns:
            x = pd.to_numeric(v[col], errors="coerce")
            if ((x < lo) | (x > hi)).any():
                bad.append(col)
    if bad:
        flags.append(Flag("implausible_value", "severe",
                          "Values outside the accepted input range: " + ", ".join(bad) + "."))
    return flags


def quality_flags_by_patient(obs: pd.DataFrame, max_visits: int,
                             hba1c_age_days: tuple[int, int] = (180, 365)
                             ) -> dict[str, list[Flag]]:
    return {str(pid): quality_flags(g, max_visits, hba1c_age_days)
            for pid, g in obs.groupby("patient_id", sort=True)}


def worst_severity(flags: list[Flag]) -> str | None:
    for s in SEVERITIES:
        if any(f.severity == s for f in flags):
            return s
    return None


__all__ = ["Flag", "VALUE_BOUNDS", "quality_flags", "quality_flags_by_patient",
           "worst_severity"]
