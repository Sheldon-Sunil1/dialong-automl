"""Engine, session factory, and idempotent schema initialization.

Database path resolution (highest precedence first):
    1. explicit path passed to the factory
    2. ``DIALONG_DB_PATH`` environment variable
    3. ``AppConfig.paths.data_dir / 'dialong_patients.db'`` (default
       ``data/dialong_patients.db``)

The schema is created with ``create_all`` (create-if-absent). It is NEVER
dropped or recreated automatically, so existing patient data persists across
restarts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from dialong_automl.db.models import Base

logger = logging.getLogger("dialong_automl.db")

_DEFAULT_DB_FILENAME = "dialong_patients.db"


def get_default_db_path() -> Path:
    """Default DB path from config (``database.db_path``, else data_dir)."""
    try:
        from dialong_automl.config import load_config

        cfg = load_config(None)
        configured = getattr(getattr(cfg, "database", None), "db_path", None)
        if configured:
            return Path(configured)
        return Path(cfg.paths.data_dir) / _DEFAULT_DB_FILENAME
    except Exception:  # noqa: BLE001 — config best-effort
        return Path("data") / _DEFAULT_DB_FILENAME


def resolve_db_path(db_path: str | Path | None) -> Path:
    """Resolve the DB path from arg, env var, or config default."""
    if db_path is not None:
        return Path(db_path)
    env = os.environ.get("DIALONG_DB_PATH")
    if env:
        return Path(env)
    return get_default_db_path()


@dataclass
class DatabaseConfig:
    """Resolved database configuration."""

    db_path: Path
    echo: bool = False

    @property
    def url(self) -> str:
        # Absolute path → sqlite:////abs ; relative → sqlite:///rel
        p = self.db_path
        return f"sqlite:///{p.as_posix()}"


def _enable_sqlite_fk(dbapi_connection, _connection_record) -> None:
    """Enforce foreign keys on every SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def create_db_engine(db_path: str | Path | None = None, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine for the resolved SQLite path.

    Creates the parent directory if needed. Does not create/modify the schema.
    """
    resolved = resolve_db_path(db_path)
    if resolved.parent and str(resolved.parent) not in ("", "."):
        resolved.parent.mkdir(parents=True, exist_ok=True)

    cfg = DatabaseConfig(db_path=resolved, echo=echo)
    engine = create_engine(
        cfg.url,
        echo=echo,
        future=True,
        connect_args={"check_same_thread": False},
    )
    event.listen(engine, "connect", _enable_sqlite_fk)
    logger.info("Database engine created for %s", resolved)
    return engine


def init_db(engine: Engine) -> None:
    """Create tables if absent. Idempotent; never drops existing tables/data."""
    Base.metadata.create_all(engine)
    logger.info("Database schema ensured (idempotent create_all).")


def create_session_factory(
    db_path: str | Path | None = None,
    *,
    echo: bool = False,
    initialize: bool = True,
) -> tuple[Engine, sessionmaker[Session]]:
    """Return an ``(engine, SessionFactory)`` pair.

    When *initialize* is True the schema is ensured (create-if-absent).
    """
    engine = create_db_engine(db_path, echo=echo)
    if initialize:
        init_db(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return engine, factory
