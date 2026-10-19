"""Repository layer — the ONLY place that touches patient/visit persistence.

All methods manage their own transactions and translate low-level errors into
domain exceptions. API routes call these methods; they never embed SQL.

Ownership rule: every visit operation is scoped to the owning patient. A visit
id belonging to another patient is treated as not-found for the requesting
patient (never read/updated/deleted across patients).
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from dialong_automl.db.models import (
    CLINICAL_COLUMNS,
    Patient,
    Visit,
)

logger = logging.getLogger("dialong_automl.db.repository")


class RepositoryError(RuntimeError):
    """Generic persistence failure (DB unavailable, write failed, etc.)."""


class PatientNotFoundError(RepositoryError):
    """Raised when a patient_id has no record."""


class VisitNotFoundError(RepositoryError):
    """Raised when a visit id does not exist for the given patient."""


class VisitOwnershipError(RepositoryError):
    """Raised when a visit exists but belongs to a different patient."""


def normalize_patient_id(patient_id: str) -> str:
    """Trim surrounding whitespace. Case is preserved (IDs are case-sensitive)."""
    if patient_id is None:
        raise RepositoryError("patient_id is required.")
    pid = str(patient_id).strip()
    if pid == "":
        raise RepositoryError("patient_id must be a non-empty string.")
    return pid


def _clean_clinical(values: dict) -> dict:
    """Keep only canonical clinical columns; empty strings become None (missing)."""
    out: dict = {}
    for col in CLINICAL_COLUMNS:
        if col not in values:
            continue
        v = values[col]
        if v == "" or v is None:
            out[col] = None
        else:
            out[col] = v
    return out


class PatientRepository:
    """Transactional access to patients and their visits."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    # ------------------------------------------------------------------
    # Patients
    # ------------------------------------------------------------------

    def get_patient(self, patient_id: str) -> Patient | None:
        pid = normalize_patient_id(patient_id)
        with self._sf() as s:
            return s.scalar(select(Patient).where(Patient.patient_id == pid))

    def patient_exists(self, patient_id: str) -> bool:
        return self.get_patient(patient_id) is not None

    def get_or_create_patient(self, patient_id: str) -> tuple[Patient, bool]:
        """Return (patient, created). Concurrency-safe via the unique constraint.

        A race that violates the unique constraint is caught and resolved by
        re-reading the now-existing row, so concurrent requests cannot create
        duplicate patients.
        """
        pid = normalize_patient_id(patient_id)
        with self._sf() as s:
            existing = s.scalar(select(Patient).where(Patient.patient_id == pid))
            if existing is not None:
                return existing, False
            patient = Patient(patient_id=pid)
            s.add(patient)
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                existing = s.scalar(select(Patient).where(Patient.patient_id == pid))
                if existing is None:  # pragma: no cover — defensive
                    raise RepositoryError("Failed to create or fetch patient.") from None
                return existing, False
            except SQLAlchemyError as exc:
                s.rollback()
                logger.warning("get_or_create_patient failed: %s", exc)
                raise RepositoryError("Database write failed while creating patient.") from exc
            s.refresh(patient)
            return patient, True

    # ------------------------------------------------------------------
    # Visits
    # ------------------------------------------------------------------

    def list_visits(self, patient_id: str) -> list[Visit]:
        """Return the patient's visits in chronological order. Raises if absent."""
        pid = normalize_patient_id(patient_id)
        with self._sf() as s:
            patient = s.scalar(select(Patient).where(Patient.patient_id == pid))
            if patient is None:
                raise PatientNotFoundError(f"No patient with id {pid!r}.")
            rows = s.scalars(
                select(Visit)
                .where(Visit.patient_fk == patient.id)
                .order_by(Visit.visit_date.asc(), Visit.id.asc())
            ).all()
            return list(rows)

    def add_visit(self, patient_id: str, visit_date: str, clinical: dict) -> Visit:
        pid = normalize_patient_id(patient_id)
        if not visit_date or str(visit_date).strip() == "":
            raise RepositoryError("visit_date is required.")
        with self._sf() as s:
            patient = s.scalar(select(Patient).where(Patient.patient_id == pid))
            if patient is None:
                raise PatientNotFoundError(f"No patient with id {pid!r}.")
            visit = Visit(
                patient_fk=patient.id,
                visit_date=str(visit_date).strip(),
                **_clean_clinical(clinical),
            )
            s.add(visit)
            try:
                s.commit()
            except SQLAlchemyError as exc:
                s.rollback()
                logger.warning("add_visit failed: %s", exc)
                raise RepositoryError("Database write failed while adding visit.") from exc
            s.refresh(visit)
            return visit

    def update_visit(
        self, patient_id: str, visit_id: int, visit_date: str | None, clinical: dict
    ) -> Visit:
        """Update a visit, enforcing that it belongs to the patient."""
        pid = normalize_patient_id(patient_id)
        with self._sf() as s:
            patient = s.scalar(select(Patient).where(Patient.patient_id == pid))
            if patient is None:
                raise PatientNotFoundError(f"No patient with id {pid!r}.")
            visit = s.get(Visit, visit_id)
            if visit is None:
                raise VisitNotFoundError(f"No visit with id {visit_id}.")
            if visit.patient_fk != patient.id:
                raise VisitOwnershipError(
                    f"Visit {visit_id} does not belong to patient {pid!r}."
                )
            if visit_date is not None and str(visit_date).strip() != "":
                visit.visit_date = str(visit_date).strip()
            for col, val in _clean_clinical(clinical).items():
                setattr(visit, col, val)
            try:
                s.commit()
            except SQLAlchemyError as exc:
                s.rollback()
                logger.warning("update_visit failed: %s", exc)
                raise RepositoryError("Database write failed while updating visit.") from exc
            s.refresh(visit)
            return visit

    def delete_visit(self, patient_id: str, visit_id: int) -> None:
        """Delete a visit, enforcing ownership."""
        pid = normalize_patient_id(patient_id)
        with self._sf() as s:
            patient = s.scalar(select(Patient).where(Patient.patient_id == pid))
            if patient is None:
                raise PatientNotFoundError(f"No patient with id {pid!r}.")
            visit = s.get(Visit, visit_id)
            if visit is None:
                raise VisitNotFoundError(f"No visit with id {visit_id}.")
            if visit.patient_fk != patient.id:
                raise VisitOwnershipError(
                    f"Visit {visit_id} does not belong to patient {pid!r}."
                )
            s.delete(visit)
            try:
                s.commit()
            except SQLAlchemyError as exc:
                s.rollback()
                logger.warning("delete_visit failed: %s", exc)
                raise RepositoryError("Database write failed while deleting visit.") from exc

    def get_visits_by_ids(self, patient_id: str, visit_ids: list[int]) -> list[Visit]:
        """Fetch specific visits for a patient, enforcing ownership for each id.

        Returns them sorted chronologically. Raises VisitOwnershipError if any
        id belongs to another patient, VisitNotFoundError if any id is unknown.
        """
        pid = normalize_patient_id(patient_id)
        if not visit_ids:
            raise RepositoryError("No visit ids supplied.")
        with self._sf() as s:
            patient = s.scalar(select(Patient).where(Patient.patient_id == pid))
            if patient is None:
                raise PatientNotFoundError(f"No patient with id {pid!r}.")
            rows = s.scalars(select(Visit).where(Visit.id.in_(visit_ids))).all()
            found = {v.id: v for v in rows}
            for vid in visit_ids:
                if vid not in found:
                    raise VisitNotFoundError(f"No visit with id {vid}.")
                if found[vid].patient_fk != patient.id:
                    raise VisitOwnershipError(
                        f"Visit {vid} does not belong to patient {pid!r}."
                    )
            selected = [found[vid] for vid in visit_ids]
            selected.sort(key=lambda v: (v.visit_date, v.id))
            return selected
