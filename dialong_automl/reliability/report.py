"""Reliability level and report (Phase 14D, item R6).

Rules fixed in the Phase 14A design before any evaluation:

* Low: any severe flag (input quality R4 or OOD "far" R5); or the conformal
  set is undecided or empty (R3); or the variability-range width exceeds the
  upper width cut-off (95th percentile on the fit patients).
* Moderate: not Low, and any moderate flag (including OOD "unusual"); or the
  width exceeds the lower cut-off (80th percentile).
* High: otherwise.

Info flags are listed but never change the level. Every triggered reason is
reported in plain words. The level describes how much to trust this
prediction relative to others from the same model on synthetic data; it is
not a clinical statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dialong_automl.reliability.quality import Flag

LEVELS = ("Low", "Moderate", "High")
SET_MESSAGES = {
    "undecided": "The 90 % prediction set contains both outcomes (the model cannot separate them "
                 "for this input).",
    "empty": "The prediction set is empty (this input is unlike the calibration patients).",
}


def ood_flag(category: str, percentile: float) -> Flag | None:
    if category == "far":
        return Flag("ood_far", "severe", f"The inputs are more unusual than {percentile:.0f} % of "
                                         "training patients (beyond the 99th percentile).")
    if category == "unusual":
        return Flag("ood_unusual", "moderate", f"The inputs are more unusual than "
                                               f"{percentile:.0f} % of training patients.")
    return None


def width_cutoffs(widths: np.ndarray, percentiles: tuple[float, float] = (80.0, 95.0)
                  ) -> tuple[float, float]:
    w = np.asarray(widths, dtype=np.float64)
    if w.size == 0:
        raise ValueError("No widths to derive cut-offs from.")
    lo, hi = np.percentile(w, percentiles)
    return float(lo), float(hi)


@dataclass
class ReliabilityReport:
    level: str
    reasons: list[str]
    flags: list[Flag] = field(default_factory=list)
    prediction_set: str | None = None
    range_low: float | None = None
    range_high: float | None = None
    ood_percentile: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "reasons": self.reasons,
                "flags": [f.to_dict() for f in self.flags],
                "prediction_set": self.prediction_set, "range_low": self.range_low,
                "range_high": self.range_high, "ood_percentile": self.ood_percentile}


def reliability_level(flags: list[Flag], prediction_set: str | None = None,
                      width: float | None = None,
                      cutoffs: tuple[float, float] | None = None) -> tuple[str, list[str]]:
    """Apply the R6 rule. Missing components (None) are simply not used."""
    low, moderate = [], []
    for f in flags:
        if f.severity == "severe":
            low.append(f.message)
        elif f.severity == "moderate":
            moderate.append(f.message)        # info flags never change the level
    if prediction_set in SET_MESSAGES:
        low.append(SET_MESSAGES[prediction_set])
    if width is not None and cutoffs is not None:
        c80, c95 = cutoffs
        if width > c95:
            low.append(f"Refitted models disagree strongly (range width {width:.2f}).")
        elif width > c80:
            moderate.append(f"Refitted models disagree more than usual (range width {width:.2f}).")
    if low:
        return "Low", low + moderate
    if moderate:
        return "Moderate", moderate
    return "High", []


def build_report(flags: list[Flag], prediction_set: str | None = None,
                 range_low: float | None = None, range_high: float | None = None,
                 cutoffs: tuple[float, float] | None = None,
                 ood_category: str | None = None,
                 ood_percentile: float | None = None) -> ReliabilityReport:
    all_flags = list(flags)
    if ood_category is not None:
        f = ood_flag(ood_category, ood_percentile or 0.0)
        if f is not None:
            all_flags.append(f)
    width = None if range_low is None or range_high is None else range_high - range_low
    level, reasons = reliability_level(all_flags, prediction_set, width, cutoffs)
    return ReliabilityReport(level, reasons, all_flags, prediction_set, range_low, range_high,
                             ood_percentile)
