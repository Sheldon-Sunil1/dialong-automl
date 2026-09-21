"""Data sub-package for DiaLong-AutoML.

Phase 2 — synthetic generation, loading, validation, profiling.
Phase 3 — cohort construction, temporal splitting.

Canonical usage pattern
-----------------------
The intended workflow for the longitudinal research experiment is::

    1.  Generate or load the *full* timeline CSV
        (includes both observation and future visits).

    2.  Load it with ``load_longitudinal_csv``.

    3.  Build the cohort with ``CohortBuilder`` in
        ``CohortMode.FUTURE_OUTCOME`` — the canonical mode.
        This derives the target from post-cutoff future visits within
        the configured ``prediction_horizon_days`` window and returns
        a leakage-free ``observation_data`` DataFrame.

    4.  Split patients (never rows) with ``PatientSplitter``.

    5.  Pass ``CohortResult.observation_data`` to the model.
        Never pass ``future_outcome_audit`` or raw future-visit rows.

Temporal-leakage guarantees (enforced in code, tested in
``tests/test_no_temporal_leakage.py``)
---------------------------------------------------------
* Future observations are **never** returned in ``observation_data``.
* The ``target``, ``is_future_visit``, and all columns matching
  ``_FUTURE_LEAKAGE_COLUMNS`` are stripped from ``observation_data``
  before it is returned.
* A hard ``RuntimeError`` is raised if a programming error causes
  any observation row to have ``visit_date > cutoff_date``.
* ``get_safe_observation_columns`` provides an explicit safelist of
  columns appropriate for model input.

Pre-labeled mode note
---------------------
``CohortMode.PRE_LABELED`` reads the ``target`` column directly from the
input DataFrame without applying a prediction horizon.  It is intended
only for unit-testing or datasets with pre-existing clinically verified
labels, and **must not** be used as a shortcut for FUTURE_OUTCOME on
the synthetic full-timeline data.

Public API::

    from dialong_automl.data import (
        # Phase 2
        SyntheticGenerator, GeneratorConfig, GenerationResult,
        load_longitudinal_csv, LoadError,
        DataValidator, ValidationConfig, ValidationResult,
        DataProfiler, DataProfile,
        # Phase 3
        CohortBuilder, CohortConfig, CohortMode, CohortResult, CohortAudit,
        PatientSplitter, SplitConfig, SplitResult,
        get_safe_observation_columns,
    )
"""

# Phase 2 — Generation
from dialong_automl.data.synthetic_generator import (
    GenerationResult,
    GeneratorConfig,
    SyntheticGenerator,
)

# Phase 2 — Loading
from dialong_automl.data.loader import LoadError, load_longitudinal_csv

# Phase 2 — Validation
from dialong_automl.data.validator import (
    DataValidator,
    ValidationConfig,
    ValidationResult,
)

# Phase 2 — Profiling
from dialong_automl.data.profiler import DataProfile, DataProfiler

# Phase 3 — Cohort construction
from dialong_automl.data.cohort import (
    CohortAudit,
    CohortBuilder,
    CohortConfig,
    CohortMode,
    CohortResult,
    get_safe_observation_columns,
)

# Phase 3 — Splitting
from dialong_automl.data.splitter import PatientSplitter, SplitConfig, SplitResult

__all__ = [
    # Generation
    "SyntheticGenerator",
    "GeneratorConfig",
    "GenerationResult",
    # Loading
    "load_longitudinal_csv",
    "LoadError",
    # Validation
    "DataValidator",
    "ValidationConfig",
    "ValidationResult",
    # Profiling
    "DataProfiler",
    "DataProfile",
    # Cohort
    "CohortBuilder",
    "CohortConfig",
    "CohortMode",
    "CohortResult",
    "CohortAudit",
    "get_safe_observation_columns",
    # Splitting
    "PatientSplitter",
    "SplitConfig",
    "SplitResult",
]
