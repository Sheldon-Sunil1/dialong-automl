"""Fixed reference baselines B0-B3 (Phase 13C).

Comparators reported next to the AutoML-selected model. They are deliberately
NOT part of the model registry, so they can never be Optuna candidates or the
selected model. See ``predictors`` and ``runner``.
"""

from dialong_automl.baselines.predictors import (
    AggregateLogisticBaseline,
    Baseline,
    BaselineInputs,
    LastHbA1cBaseline,
    PrevalenceBaseline,
    TrendExtrapolationBaseline,
    b3_columns,
    default_baselines,
    hba1c_slope_per_day,
)
from dialong_automl.baselines.runner import (
    BaselineRunner,
    comparison_rows,
    select_raw_score_threshold,
)

__all__ = [
    "AggregateLogisticBaseline",
    "Baseline",
    "BaselineInputs",
    "BaselineRunner",
    "LastHbA1cBaseline",
    "PrevalenceBaseline",
    "TrendExtrapolationBaseline",
    "b3_columns",
    "comparison_rows",
    "default_baselines",
    "hba1c_slope_per_day",
    "select_raw_score_threshold",
]
