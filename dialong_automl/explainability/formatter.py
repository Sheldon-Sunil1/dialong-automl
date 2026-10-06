"""Explanation result containers for DiaLong-AutoML.

All explanation objects are JSON-serialisable and carry the required
disclaimer.  They never contain test-set labels or features used during
model selection.

DISCLAIMER embedded in every persisted artifact:
    "Explanation values describe model behavior on the supplied data
     and do not establish medical causality."
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Required disclaimer — must appear in every persisted explanation artifact
EXPLANATION_DISCLAIMER = (
    "Explanation values describe model behavior on the supplied data "
    "and do not establish medical causality."
)


# ---------------------------------------------------------------------------
# Feature importance entry
# ---------------------------------------------------------------------------

@dataclass
class FeatureImportanceEntry:
    """One entry in a global feature importance ranking.

    Attributes
    ----------
    rank:           1-based rank (1 = most important).
    column_name:    Raw preprocessor output column name.
    readable_name:  Human-readable label from Phase 4 feature mapping.
    importance:     Importance value (SHAP mean |shap| or permutation drop).
    direction:      ``"positive"``, ``"negative"``, or ``"mixed"`` for SHAP;
                    ``None`` for permutation importance (unsigned).
    explanation_method: ``"shap_tree"``, ``"shap_linear"``, or
                        ``"permutation"``.
    """

    rank: int
    column_name: str
    readable_name: str
    importance: float
    direction: str | None
    explanation_method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Global explanation
# ---------------------------------------------------------------------------

@dataclass
class GlobalExplanation:
    """Global feature importance for the trained model.

    Attributes
    ----------
    model_name:     Registry name of the explained model.
    representation: ``"wide"`` or ``"sequence"``.
    explanation_method: Method used (``"shap_tree"``, ``"permutation"``, …).
    n_samples:      Number of samples used to compute importances.
    features:       Ordered list of :class:`FeatureImportanceEntry`.
    disclaimer:     Always equals :data:`EXPLANATION_DISCLAIMER`.
    metadata:       Additional context (timestamp, model version, …).
    """

    model_name: str
    representation: str
    explanation_method: str
    n_samples: int
    features: list[FeatureImportanceEntry]
    disclaimer: str = EXPLANATION_DISCLAIMER
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "representation": self.representation,
            "explanation_method": self.explanation_method,
            "n_samples": self.n_samples,
            "disclaimer": self.disclaimer,
            "metadata": self.metadata,
            "features": [f.to_dict() for f in self.features],
        }

    def save(self, path: str | Path) -> None:
        """Write to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("GlobalExplanation saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GlobalExplanation":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        features = [FeatureImportanceEntry(**f) for f in d.pop("features", [])]
        d.pop("disclaimer", None)
        return cls(**d, features=features)


# ---------------------------------------------------------------------------
# Patient-level contribution entry
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    """One feature's contribution to a single patient prediction.

    Attributes
    ----------
    rank:           Absolute rank by |shap_value| (1 = largest contributor).
    column_name:    Raw preprocessor output column name.
    readable_name:  Human-readable label.
    feature_value:  The preprocessed feature value for this patient.
    shap_value:     SHAP value (positive = pushes toward class 1).
    direction:      ``"increases_risk"`` when shap_value > 0,
                    ``"decreases_risk"`` when shap_value < 0.
    """

    rank: int
    column_name: str
    readable_name: str
    feature_value: float | None
    shap_value: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Patient explanation
# ---------------------------------------------------------------------------

@dataclass
class PatientExplanation:
    """Prediction explanation for a single patient / sample.

    Attributes
    ----------
    patient_id:     Identifier for this sample.  ``None`` when anonymous.
    prediction_probability: Model output probability for class 1.
    predicted_class:        Binary prediction at ``threshold``.
    threshold:              Classification threshold used.
    model_name:     Registry name of the explained model.
    explanation_method: SHAP variant or fallback.
    top_positive:   Top features pushing toward class 1, ordered by |shap|.
    top_negative:   Top features pushing toward class 0, ordered by |shap|.
    disclaimer:     Always equals :data:`EXPLANATION_DISCLAIMER`.
    metadata:       Additional context.

    Notes
    -----
    The model contribution shown here describes the model's internal
    reasoning on the supplied data only.  It does NOT establish clinical
    causality.
    """

    patient_id: str | None
    prediction_probability: float
    predicted_class: int
    threshold: float
    model_name: str
    explanation_method: str
    top_positive: list[FeatureContribution]
    top_negative: list[FeatureContribution]
    disclaimer: str = EXPLANATION_DISCLAIMER
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "prediction_probability": self.prediction_probability,
            "predicted_class": self.predicted_class,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "explanation_method": self.explanation_method,
            "disclaimer": self.disclaimer,
            "metadata": self.metadata,
            "top_positive_contributions": [c.to_dict() for c in self.top_positive],
            "top_negative_contributions": [c.to_dict() for c in self.top_negative],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("PatientExplanation saved → %s", path)


# ---------------------------------------------------------------------------
# Temporal occlusion entry
# ---------------------------------------------------------------------------

@dataclass
class TimestepInfluence:
    """Influence of one masked timestep on a GRU prediction.

    Attributes
    ----------
    timestep:           0-based timestep index.
    baseline_probability: Model probability without occlusion.
    occluded_probability: Model probability when this timestep is masked.
    influence:          ``baseline - occluded`` (positive = timestep
                        increases risk; negative = decreases risk).
    rank:               Rank by ``|influence|`` (1 = most influential).
    is_real_visit:      Whether this timestep contained a real visit
                        (``False`` for padding).
    """

    timestep: int
    baseline_probability: float
    occluded_probability: float
    influence: float
    rank: int
    is_real_visit: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Temporal explanation
# ---------------------------------------------------------------------------

@dataclass
class TemporalExplanation:
    """GRU temporal occlusion explanation for one patient.

    Attributes
    ----------
    patient_id:     Sample identifier.
    model_name:     ``"gru"``.
    baseline_probability: Original GRU prediction probability.
    predicted_class: Binary prediction at ``threshold``.
    threshold:      Classification threshold.
    timesteps:      One :class:`TimestepInfluence` per sequence position.
    explanation_method: Always ``"temporal_occlusion"``.
    disclaimer:     Always equals :data:`EXPLANATION_DISCLAIMER`.
    metadata:       Additional context.
    """

    patient_id: str | None
    model_name: str
    baseline_probability: float
    predicted_class: int
    threshold: float
    timesteps: list[TimestepInfluence]
    explanation_method: str = "temporal_occlusion"
    disclaimer: str = EXPLANATION_DISCLAIMER
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "model_name": self.model_name,
            "baseline_probability": self.baseline_probability,
            "predicted_class": self.predicted_class,
            "threshold": self.threshold,
            "explanation_method": self.explanation_method,
            "disclaimer": self.disclaimer,
            "metadata": self.metadata,
            "timesteps": [t.to_dict() for t in self.timesteps],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("TemporalExplanation saved → %s", path)
