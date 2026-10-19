"""Patient/visit persistence + DB-backed prediction routes (Phase 10).

All SQL is isolated in :class:`dialong_automl.db.repository.PatientRepository`;
these routes only orchestrate validation, ownership, and responses. Errors are
translated to appropriate HTTP status codes without leaking stack traces or
filesystem paths.

Routes
------
* ``POST   /patients``                      — find-or-create a patient.
* ``GET    /patients/{patient_id}``          — patient id + chronological visits.
* ``POST   /patients/{patient_id}/visits``   — add a visit.
* ``PUT    /patients/{patient_id}/visits/{visit_id}`` — update a visit (ownership).
* ``DELETE /patients/{patient_id}/visits/{visit_id}`` — delete a visit (ownership).
* ``POST   /patients/{patient_id}/predict``  — predict from SELECTED saved visits.

The legacy stateless ``POST /predict`` route is preserved unchanged elsewhere.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import APIRouter, HTTPException

from dialong_automl.api.schemas import (
    PatientHistoryResponse,
    PatientLookupResponse,
    PredictRequest,
    PredictSelectionRequest,
    SelectedPredictResponse,
    VisitInput,
    VisitRecord,
)
from dialong_automl.db.models import visit_to_dict
from dialong_automl.db.repository import (
    PatientNotFoundError,
    PatientRepository,
    RepositoryError,
    VisitNotFoundError,
    VisitOwnershipError,
    normalize_patient_id,
)

logger = logging.getLogger("dialong_automl.api.patients")

# Prediction callable signature: (patient_id, visits) -> result dict | raises.
PredictFn = Callable[[str, list[dict[str, Any]]], dict[str, Any]]


def _visit_record(visit_dict: dict) -> VisitRecord:
    return VisitRecord(**visit_dict)


def create_patient_router(
    repo_getter: Callable[[], PatientRepository | None],
    predict_fn: PredictFn,
    *,
    model_loaded: Callable[[], bool],
    observation_limit: Callable[[], int | None],
) -> APIRouter:
    """Build the patient/visit router.

    Parameters
    ----------
    repo_getter:        callable returning the live persistence repository (or
                        ``None`` when the DB is unavailable). Read per request
                        so the repo can be created during app startup.
    predict_fn:         ``(patient_id, visits) -> result`` using saved artifacts.
    model_loaded:       callable returning whether model artifacts are loaded.
    observation_limit:  callable returning the max visits the model accepts
                        (``observation_visits`` from metadata), or ``None``.
    """
    router = APIRouter(tags=["patients"])

    def _repo() -> PatientRepository:
        repo = repo_getter()
        if repo is None:
            raise HTTPException(status_code=503, detail="Patient database is unavailable.")
        return repo

    def _norm_or_422(patient_id: str) -> str:
        try:
            return normalize_patient_id(patient_id)
        except RepositoryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @router.post("/patients", response_model=PatientLookupResponse)
    def find_or_create_patient(body: dict) -> PatientLookupResponse:
        raw = body.get("patient_id") if isinstance(body, dict) else None
        if not raw or not str(raw).strip():
            raise HTTPException(status_code=422, detail="patient_id is required.")
        pid = _norm_or_422(str(raw))
        repo = _repo()
        existed = repo.patient_exists(pid)
        try:
            _, created = repo.get_or_create_patient(pid)
            visits = repo.list_visits(pid)
        except RepositoryError as exc:
            logger.warning("find_or_create_patient failed: %s", exc)
            raise HTTPException(status_code=503, detail="Patient database is unavailable.") from None
        return PatientLookupResponse(
            patient_id=pid,
            exists=existed,
            created=created,
            visits=[_visit_record(visit_to_dict(v)) for v in visits],
        )

    @router.get("/patients/{patient_id}", response_model=PatientHistoryResponse)
    def get_patient(patient_id: str) -> PatientHistoryResponse:
        pid = _norm_or_422(patient_id)
        repo = _repo()
        try:
            visits = repo.list_visits(pid)
        except PatientNotFoundError:
            raise HTTPException(status_code=404, detail=f"Patient {pid!r} not found.") from None
        except RepositoryError:
            raise HTTPException(status_code=503, detail="Patient database is unavailable.") from None
        return PatientHistoryResponse(
            patient_id=pid,
            visits=[_visit_record(visit_to_dict(v)) for v in visits],
        )

    @router.post("/patients/{patient_id}/visits", response_model=VisitRecord, status_code=201)
    def add_visit(patient_id: str, visit: VisitInput) -> VisitRecord:
        pid = _norm_or_422(patient_id)
        repo = _repo()
        # Ensure the patient exists (create on first write is acceptable for the
        # workflow, but here we require an explicit prior create via /patients).
        if not repo.patient_exists(pid):
            raise HTTPException(status_code=404, detail=f"Patient {pid!r} not found.")
        data = visit.model_dump()
        vdate = data.pop("visit_date")
        try:
            created = repo.add_visit(pid, vdate, data)
        except RepositoryError as exc:
            logger.warning("add_visit failed: %s", exc)
            raise HTTPException(status_code=503, detail="Could not save the visit.") from None
        return _visit_record(visit_to_dict(created))

    @router.put("/patients/{patient_id}/visits/{visit_id}", response_model=VisitRecord)
    def update_visit(patient_id: str, visit_id: int, visit: VisitInput) -> VisitRecord:
        pid = _norm_or_422(patient_id)
        repo = _repo()
        data = visit.model_dump()
        vdate = data.pop("visit_date")
        try:
            updated = repo.update_visit(pid, visit_id, vdate, data)
        except PatientNotFoundError:
            raise HTTPException(status_code=404, detail=f"Patient {pid!r} not found.") from None
        except VisitOwnershipError:
            # Do not reveal that the visit exists under another patient.
            raise HTTPException(status_code=404, detail="Visit not found for this patient.") from None
        except VisitNotFoundError:
            raise HTTPException(status_code=404, detail="Visit not found for this patient.") from None
        except RepositoryError:
            raise HTTPException(status_code=503, detail="Could not update the visit.") from None
        return _visit_record(visit_to_dict(updated))

    @router.delete("/patients/{patient_id}/visits/{visit_id}", status_code=204)
    def delete_visit(patient_id: str, visit_id: int) -> None:
        pid = _norm_or_422(patient_id)
        repo = _repo()
        try:
            repo.delete_visit(pid, visit_id)
        except PatientNotFoundError:
            raise HTTPException(status_code=404, detail=f"Patient {pid!r} not found.") from None
        except (VisitOwnershipError, VisitNotFoundError):
            raise HTTPException(status_code=404, detail="Visit not found for this patient.") from None
        except RepositoryError:
            raise HTTPException(status_code=503, detail="Could not delete the visit.") from None
        return None

    @router.post("/patients/{patient_id}/predict", response_model=SelectedPredictResponse)
    def predict_selected(patient_id: str, body: PredictSelectionRequest) -> SelectedPredictResponse:
        pid = _norm_or_422(patient_id)
        if not model_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model artifacts are not available; prediction is unavailable.",
            )

        # 1. Resolve the selected visits from the DB, enforcing ownership.
        repo = _repo()
        try:
            selected = repo.get_visits_by_ids(pid, body.visit_ids)
        except PatientNotFoundError:
            raise HTTPException(status_code=404, detail=f"Patient {pid!r} not found.") from None
        except (VisitOwnershipError, VisitNotFoundError):
            raise HTTPException(
                status_code=404,
                detail="One or more selected visits do not belong to this patient.",
            ) from None
        except RepositoryError:
            raise HTTPException(status_code=503, detail="Patient database is unavailable.") from None

        # 2. Enforce the observation-window limit EXPLICITLY (no silent truncation).
        limit = observation_limit()
        if limit is not None and len(selected) > limit:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"The model accepts at most {limit} visits per prediction "
                    f"(observation window). You selected {len(selected)}. "
                    "Deselect visits so the selection fits the window."
                ),
            )

        # 3. Build server-trusted visit dicts from the DB (not client values).
        visits: list[dict[str, Any]] = []
        for v in selected:  # already chronological
            d = visit_to_dict(v)
            d.pop("id", None)
            d.pop("created_at", None)
            d.pop("updated_at", None)
            visits.append({k: val for k, val in d.items() if val is not None or k == "visit_date"})

        # 4. Predict via the existing fitted-artifact pipeline.
        from dialong_automl.predict import PredictionError

        try:
            result = predict_fn(pid, visits)
        except PredictionError as exc:
            msg = str(exc)
            logger.warning("Selected-visit prediction failed: %s", msg)
            raise HTTPException(status_code=422, detail=_safe_predict_error(msg)) from None
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error during selected-visit prediction.")
            raise HTTPException(status_code=500, detail="Internal error during prediction.") from None

        return SelectedPredictResponse(
            **result,
            selected_visit_ids=[v.id for v in selected],
            selected_visit_dates=[v.visit_date for v in selected],
        )

    return router


def _safe_predict_error(message: str) -> str:
    if ("\\" in message) or ("/" in message):
        return "Invalid visit selection for prediction."
    return message


__all__ = ["create_patient_router", "PredictFn"]
