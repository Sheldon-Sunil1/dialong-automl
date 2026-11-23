"""Prediction reliability (Phase 14).

14B: post-hoc calibration (R1). Later phases add the ensemble range, conformal
sets, input-quality and out-of-distribution flags, and the reliability level
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

__all__ = [
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
