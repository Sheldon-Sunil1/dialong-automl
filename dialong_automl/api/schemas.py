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


# ---------------------------------------------------------------------------
# Patient / visit persistence (Phase 10 enhancement)
# ---------------------------------------------------------------------------


class VisitInput(BaseModel):
    """Clinical fields for creating/updating a persisted visit.

    Only ``visit_date`` is required; clinical fields are optional (NULL ==
    missing, consistent with the inference pipeline). Unknown extra keys are
    rejected so leakage/typo fields never reach persistence.
    """

    model_config = {"extra": "forbid"}

    visit_date: str = Field(..., description="ISO-8601 visit date (YYYY-MM-DD).")
    hba1c: float | None = None
    fasting_glucose: float | None = None
    bmi: float | None = None
    systolic_bp: float | None = None
    diastolic_bp: float | None = None
    ldl_cholesterol: float | None = None
    hdl_cholesterol: float | None = None
    triglycerides: float | None = None
    heart_rate: float | None = None
    age_at_visit: float | None = None
    sex: str | None = None
    smoking_status: str | None = None
    physical_activity_level: str | None = None
    family_history_diabetes: str | None = None
    metformin_use: str | None = None
    hypertension_diagnosis: str | None = None


class VisitRecord(VisitInput):
    """A persisted visit, with its database id and timestamps."""

    id: int
    created_at: str | None = None
    updated_at: str | None = None


class PatientHistoryResponse(BaseModel):
    """A patient's id plus its chronological visit history."""

    patient_id: str
    created: bool = Field(
        False, description="True if this request created the patient record."
    )
    visits: list[VisitRecord] = Field(default_factory=list)


class PatientLookupResponse(BaseModel):
    """Result of looking up (and optionally creating) a patient."""

    patient_id: str
    exists: bool
    created: bool = False
    visits: list[VisitRecord] = Field(default_factory=list)


class PredictSelectionRequest(BaseModel):
    """DB-backed prediction request: which saved visits to score."""

    model_config = {"extra": "forbid"}

    visit_ids: list[int] = Field(
        ..., min_length=1, description="IDs of the saved visits to use (>= 1)."
    )


class SelectedPredictResponse(PredictResponse):
    """Prediction response that also traces the exact selected visits."""

    selected_visit_ids: list[int] = Field(default_factory=list)
    selected_visit_dates: list[str] = Field(default_factory=list)
