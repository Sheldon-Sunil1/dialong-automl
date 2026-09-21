"""Feature engineering sub-package for DiaLong-AutoML.

Phase 4 — Wide and sequence representations, training-only preprocessing.

Public API::

    from dialong_automl.features import (
        # Wide representation
        WideRepresentationBuilder,
        WideRepresentationResult,
        FeatureInfo,
        LeakageError,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        # Sequence representation
        SequenceRepresentationBuilder,
        SequenceRepresentationResult,
        # Preprocessing
        WidePreprocessor,
        SequencePreprocessor,
    )

Canonical usage pattern
-----------------------
All representations are built from ``CohortResult.observation_data`` only.
Future observations must never enter either representation.

1. Build the future-outcome cohort (Phase 3).
2. Split patients (Phase 3).
3. Build wide or sequence representations per split.
4. Fit preprocessors on training data only.
5. Transform validation and test using the fitted training preprocessor.
"""

from dialong_automl.features.wide import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeatureInfo,
    LeakageError,
    WideRepresentationBuilder,
    WideRepresentationResult,
)

from dialong_automl.features.sequence import (
    SequenceRepresentationBuilder,
    SequenceRepresentationResult,
)

from dialong_automl.features.preprocessing import (
    SequencePreprocessor,
    WidePreprocessor,
)

__all__ = [
    # Wide
    "WideRepresentationBuilder",
    "WideRepresentationResult",
    "FeatureInfo",
    "LeakageError",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    # Sequence
    "SequenceRepresentationBuilder",
    "SequenceRepresentationResult",
    # Preprocessing
    "WidePreprocessor",
    "SequencePreprocessor",
]
