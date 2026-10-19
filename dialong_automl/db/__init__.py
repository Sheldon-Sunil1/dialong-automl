"""Persistence layer for DiaLong-AutoML (Phase 10 enhancement).

Provides a configurable SQLite database (via SQLAlchemy) storing independent
patient records and their longitudinal visits, plus a repository layer that
isolates all SQL behind explicit, transactional methods.

Design notes
------------
* The database path is configurable (``DIALONG_DB_PATH`` env var, else the
  configured ``AppConfig`` default ``data/dialong_patients.db``).
* Schema initialization is idempotent and NEVER drops/recreates tables, so
  existing patient data survives restarts.
* Foreign keys are enforced (SQLite ``PRAGMA foreign_keys=ON``).
* Only patient/visit data is stored here — no secrets or model artifacts.
* The SQLAlchemy models keep a clean boundary so a future migration to
  PostgreSQL is straightforward (no SQLite-only SQL in the repository).
"""

from dialong_automl.db.engine import (
    DatabaseConfig,
    create_session_factory,
    get_default_db_path,
    init_db,
    resolve_db_path,
)
from dialong_automl.db.models import Base, Patient, Visit
from dialong_automl.db.repository import (
    PatientRepository,
    RepositoryError,
    VisitNotFoundError,
    VisitOwnershipError,
)

__all__ = [
    "DatabaseConfig",
    "create_session_factory",
    "get_default_db_path",
    "resolve_db_path",
    "init_db",
    "Base",
    "Patient",
    "Visit",
    "PatientRepository",
    "RepositoryError",
    "VisitNotFoundError",
    "VisitOwnershipError",
]
