"""SQLAlchemy ORM models for patient/visit persistence.

The ``visits`` table mirrors the project's canonical clinical feature schema
(``dialong_automl.features.wide.NUMERIC_FEATURES`` /
``CATEGORICAL_FEATURES``) plus ``visit_date``. Clinical columns are nullable so
a genuinely missing measurement is stored as NULL — consistent with the
inference pipeline's missing-value contract. No new clinical features are
invented and the model's expected feature schema is unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Patient(Base):
    """A patient, identified by an externally-supplied unique ``patient_id``."""

    __tablename__ = "patients"
    __table_args__ = (UniqueConstraint("patient_id", name="uq_patients_patient_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
        nullable=False,
    )

    visits: Mapped[list["Visit"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="Visit.visit_date",
    )


class Visit(Base):
    """A single observation visit owned by exactly one patient.

    ``patient_fk`` is the internal FK (ON DELETE CASCADE). Clinical columns
    match the canonical feature schema and are nullable (NULL == missing).
    """

    __tablename__ = "visits"
    __table_args__ = (
        Index("ix_visits_patient_fk_date", "patient_fk", "visit_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_fk: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )

    visit_date: Mapped[str] = mapped_column(String(32), nullable=False)

    # --- Numeric clinical features (NUMERIC_FEATURES) ---
    hba1c: Mapped[float | None] = mapped_column(Float, nullable=True)
    fasting_glucose: Mapped[float | None] = mapped_column(Float, nullable=True)
    bmi: Mapped[float | None] = mapped_column(Float, nullable=True)
    systolic_bp: Mapped[float | None] = mapped_column(Float, nullable=True)
    diastolic_bp: Mapped[float | None] = mapped_column(Float, nullable=True)
    ldl_cholesterol: Mapped[float | None] = mapped_column(Float, nullable=True)
    hdl_cholesterol: Mapped[float | None] = mapped_column(Float, nullable=True)
    triglycerides: Mapped[float | None] = mapped_column(Float, nullable=True)
    heart_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    age_at_visit: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- Categorical clinical features (CATEGORICAL_FEATURES) ---
    sex: Mapped[str | None] = mapped_column(String(32), nullable=True)
    smoking_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    physical_activity_level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    family_history_diabetes: Mapped[str | None] = mapped_column(String(32), nullable=True)
    metformin_use: Mapped[str | None] = mapped_column(String(32), nullable=True)
    hypertension_diagnosis: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(back_populates="visits")


# Canonical ordered clinical columns (must match features/wide.py).
NUMERIC_COLUMNS: tuple[str, ...] = (
    "hba1c",
    "fasting_glucose",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
    "heart_rate",
    "age_at_visit",
)
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "sex",
    "smoking_status",
    "physical_activity_level",
    "family_history_diabetes",
    "metformin_use",
    "hypertension_diagnosis",
)
CLINICAL_COLUMNS: tuple[str, ...] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def visit_to_dict(visit: Visit) -> dict:
    """Serialize a Visit ORM row to a plain dict (API-friendly)."""
    out: dict = {
        "id": visit.id,
        "visit_date": visit.visit_date,
    }
    for col in CLINICAL_COLUMNS:
        out[col] = getattr(visit, col)
    out["created_at"] = visit.created_at.isoformat() if visit.created_at else None
    out["updated_at"] = visit.updated_at.isoformat() if visit.updated_at else None
    return out
