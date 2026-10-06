"""Temporal occlusion explainability for the GRU model.

For each timestep in a patient's sequence, this module:
1. Creates a copy of the input with that timestep zeroed (occluded).
2. Obtains the prediction probability from the GRU.
3. Computes the change relative to the baseline prediction.
4. Ranks timesteps by absolute influence.

Masking semantics
-----------------
The occluded representation uses the same mask convention as the
sequence representation:
    mask[t] = 0  →  padding (zeroed in preprocessed array)
    mask[t] = 1  →  real visit

When occluding timestep *t*, we set ``X[:, t, :] = 0.0`` and
``mask[t] = 0`` to match the padding convention the GRU was trained
with.  This preserves expected input statistics.

DISCLAIMER
----------
Every persisted artifact contains:
    "Explanation values describe model behavior on the supplied data
     and do not establish medical causality."

Usage::

    from dialong_automl.explainability.temporal_occlusion import TemporalOcclusionExplainer

    explainer = TemporalOcclusionExplainer(gru_model)
    result = explainer.explain(X_seq, mask, patient_id="SYN_00001")
    result.save("artifacts/explanations/patient_SYN_00001_temporal.json")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from dialong_automl.explainability.formatter import (
    EXPLANATION_DISCLAIMER,
    TemporalExplanation,
    TimestepInfluence,
)
from dialong_automl.models.base import BaseModel

logger = logging.getLogger(__name__)


class TemporalOcclusionExplainer:
    """Compute GRU temporal occlusion sensitivity for one patient.

    Parameters
    ----------
    model:
        A fitted GRU model (must implement
        ``predict_proba_with_mask(X, mask) -> np.ndarray``).
    threshold:
        Classification threshold.
    """

    def __init__(self, model: BaseModel, threshold: float = 0.5) -> None:
        model._require_fitted()
        if not hasattr(model, "predict_proba_with_mask"):
            raise TypeError(
                f"TemporalOcclusionExplainer requires a GRU model with "
                f"predict_proba_with_mask(). Got {type(model).__name__}."
            )
        self.model = model
        self.threshold = threshold

    def explain(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        patient_id: str | None = None,
        sample_index: int = 0,
    ) -> TemporalExplanation:
        """Compute temporal occlusion for one patient.

        Parameters
        ----------
        X:
            3-D preprocessed sequence array ``(N, T, F)`` for a batch.
            Only ``sample_index`` row is explained.
        mask:
            2-D mask array ``(N, T)``.
        patient_id:
            Optional identifier for the output artifact.
        sample_index:
            Which row in *X* to explain.

        Returns
        -------
        TemporalExplanation
        """
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D, got shape {X.shape}")
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

        N, T, F = X.shape

        # Baseline: predict on the original sample
        X_single = X[[sample_index]]
        mask_single = mask[[sample_index]]
        baseline_proba = float(self.model.predict_proba_with_mask(X_single, mask_single)[0])
        pred_class = int(baseline_proba >= self.threshold)

        influences: list[TimestepInfluence] = []

        for t in range(T):
            is_real = bool(mask_single[0, t] == 1)

            # Occlude timestep t: zero the features and mark as padding
            X_occ = X_single.copy()
            mask_occ = mask_single.copy()
            X_occ[0, t, :] = 0.0
            mask_occ[0, t] = 0

            # If we just zeroed the only real timestep, use a length of 1
            # to avoid packing errors (GRU needs at least length 1)
            real_count = int(mask_occ[0].sum())
            if real_count == 0:
                # Cannot occlude the last remaining timestep meaningfully
                # — record a zero influence
                occluded_proba = baseline_proba
            else:
                occluded_proba = float(
                    self.model.predict_proba_with_mask(X_occ, mask_occ)[0]
                )

            influence = baseline_proba - occluded_proba

            influences.append(TimestepInfluence(
                timestep=t,
                baseline_probability=baseline_proba,
                occluded_probability=occluded_proba,
                influence=influence,
                rank=0,  # filled below
                is_real_visit=is_real,
            ))

        # Rank by |influence| — only real visits are meaningful
        real_influences = [inf for inf in influences if inf.is_real_visit]
        real_influences.sort(key=lambda x: -abs(x.influence))
        for rank, inf in enumerate(real_influences, start=1):
            inf.rank = rank
        # Padding timesteps get rank 0 (kept but clearly marked non-real)

        return TemporalExplanation(
            patient_id=patient_id,
            model_name=self.model.name,
            baseline_probability=baseline_proba,
            predicted_class=pred_class,
            threshold=self.threshold,
            timesteps=influences,
            explanation_method="temporal_occlusion",
            disclaimer=EXPLANATION_DISCLAIMER,
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sample_index": sample_index,
                "sequence_length": T,
                "n_features": F,
                "n_real_timesteps": int(mask_single[0].sum()),
            },
        )

    def explain_batch(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        patient_ids: list[str | None] | None = None,
    ) -> list[TemporalExplanation]:
        """Explain each sample in a batch.

        Parameters
        ----------
        X:  3-D array ``(N, T, F)``.
        mask: 2-D array ``(N, T)``.
        patient_ids: Optional list of N identifiers.

        Returns
        -------
        list of :class:`TemporalExplanation`.
        """
        N = X.shape[0]
        ids = patient_ids or [None] * N
        return [
            self.explain(X, mask, patient_id=ids[i], sample_index=i)
            for i in range(N)
        ]
