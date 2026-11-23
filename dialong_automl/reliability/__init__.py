"""Prediction reliability (Phase 14).

14B: post-hoc calibration (R1). 14C: model-variability range (R2) and
class-conditional conformal sets (R3). Later phases add input-quality and
out-of-distribution flags and the reliability level
(see docs/research/phase_14a_reliability_design.md).
"""

from dialong_automl.reliability.calibration import (
    CALIBRATION_METHODS,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    calibration_slope_intercept,
    calibration_summary,
    calibrator_from_dict,
    ece_equal_mass,
    make_calibrator,
)
from dialong_automl.reliability.conformal import MondrianConformal, coverage_summary
from dialong_automl.reliability.ensemble import member_subsamples, variability_range
from dialong_automl.reliability.ood import MahalanobisOOD
from dialong_automl.reliability.quality import Flag, quality_flags
from dialong_automl.reliability.report import ReliabilityReport, build_report, reliability_level

__all__ = [
    "Flag",
    "MahalanobisOOD",
    "ReliabilityReport",
    "build_report",
    "quality_flags",
    "reliability_level",
    "MondrianConformal",
    "coverage_summary",
    "member_subsamples",
    "variability_range",
    "CALIBRATION_METHODS",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "calibration_slope_intercept",
    "calibration_summary",
    "calibrator_from_dict",
    "ece_equal_mass",
    "make_calibrator",
]
