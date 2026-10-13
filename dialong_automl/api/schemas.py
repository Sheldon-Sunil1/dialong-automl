"""Pydantic request/response schemas for the DiaLong-AutoML API (Phase 9).

These mirror the CLI-compatible prediction contract in
``dialong_automl.predict`` so the HTTP API and the CLI accept the same input
and return the same information.

DISCLAIMER: Operates on synthetic demonstration data only. Predictions are
not clinical evidence and are not suitable for medical decisions.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VisitModel(BaseModel):
    """A single observation visit.

    Only ``visit_date`` is required. Any subset of the known clinical columns
    may be supplied; unknown extra keys are allowed here but are screened for
    forbidden future/outcome/leakage columns in the request handler (reusing
    the exact set enforced by the prediction CLI). Keeping ``extra='allow'``
    lets clients send the same visit dicts the CLI accepts.
    """

    model_config = {"extra": "allow"}

    visit_date: str = Field(
        ...,
        description="Visit date (ISO-8601 string, e.g. '2020-01-01').",
        examples=["2020-01-01"],
    )


class PredictRequest(BaseModel):
    """Patient trajectory request body (CLI-compatible)."""

    model_config = {"extra": "forbid"}

    patient_id: str = Field(
        default="UNKNOWN",
        description="Optional patient identifier (echoed back in the response).",
    )
    visits: list[VisitModel] = Field(
        ...,
        min_length=1,
        description="Chronological list of observation visits (>= 1).",
    )


class PredictResponse(BaseModel):
    """Prediction result for a single patient trajectory."""

    patient_id: str
    probability: float = Field(..., ge=0.0, le=1.0, description="P(positive class).")
    prediction: int = Field(..., description="Binary prediction at the threshold (0 or 1).")
    threshold: float = Field(..., description="Decision threshold applied.")
    model: str = Field(..., description="Selected model name.")
    representation: str = Field(..., description="Feature representation (wide/sequence).")
    n_visits: int = Field(..., description="Number of visits scored.")
    disclaimer: str = Field(..., description="Synthetic-data / non-clinical disclaimer.")


class HealthResponse(BaseModel):
    """Service health and artifact-load status."""

    status: str = Field(..., description="'ok' when the service is responsive.")
    model_loaded: bool = Field(..., description="True when model artifacts loaded at startup.")
    detail: str = Field("", description="Human-readable status detail (no filesystem paths).")


class MetadataResponse(BaseModel):
    """Public, sanitized model metadata (no filesystem paths)."""

    model_loaded: bool
    model: str | None = Field(None, description="Selected model name.")
    representation: str | None = Field(None, description="Feature representation.")
    threshold: float | None = Field(None, description="Decision threshold.")
    observation_visits: int | None = Field(
        None, description="Observation-window size used at training time."
    )
    synthetic_data: bool | None = Field(
        None, description="Whether the model was trained on synthetic data."
    )
    disclaimer: str = Field(..., description="Synthetic-data / non-clinical disclaimer.")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional non-sensitive metadata fields (paths excluded).",
    )


class ErrorResponse(BaseModel):
    """Structured error payload (never contains stack traces or paths)."""

    detail: str
