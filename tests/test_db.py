"""Phase 10 — patient/visit persistence (repository layer) tests.

All tests use an isolated temporary SQLite database (never the developer's).
"""

from __future__ import annotations

import pytest

from dialong_automl.db import PatientRepository, create_session_factory
from dialong_automl.db.engine import resolve_db_path
from dialong_automl.db.repository import (
    PatientNotFoundError,
    RepositoryError,
    VisitNotFoundError,
    VisitOwnershipError,
    normalize_patient_id,
)


@pytest.fixture()
def repo(tmp_path):
    db = tmp_path / "patients.db"
    _engine, sf = create_session_factory(db)
    return PatientRepository(sf), db


def test_normalize_patient_id_trims_but_preserves_case():
    assert normalize_patient_id("  SYN_0001 ") == "SYN_0001"
    assert normalize_patient_id("Abc") == "Abc"  # case preserved
    with pytest.raises(RepositoryError):
        normalize_patient_id("   ")


def test_init_is_idempotent_and_non_destructive(tmp_path):
    db = tmp_path / "patients.db"
    _e1, sf1 = create_session_factory(db)
    r1 = PatientRepository(sf1)
    r1.get_or_create_patient("SYN_0001")
    r1.add_visit("SYN_0001", "2020-01-01", {"hba1c": 5.6})
    # Re-initialize (as a restart would) — must not drop data.
    _e2, sf2 = create_session_factory(db)
    r2 = PatientRepository(sf2)
    assert len(r2.list_visits("SYN_0001")) == 1


def test_get_or_create_and_uniqueness(repo):
    r, _ = repo
    p1, created1 = r.get_or_create_patient("SYN_0001")
    assert created1 is True
    p2, created2 = r.get_or_create_patient("SYN_0001")
    assert created2 is False
    assert p1.id == p2.id  # same row, no duplicate


def test_add_and_list_visits_chronological(repo):
    r, _ = repo
    r.get_or_create_patient("SYN_0001")
    r.add_visit("SYN_0001", "2020-06-01", {"hba1c": 5.9})
    r.add_visit("SYN_0001", "2020-01-01", {"hba1c": 5.6})
    dates = [v.visit_date for v in r.list_visits("SYN_0001")]
    assert dates == ["2020-01-01", "2020-06-01"]  # chronological


def test_missing_values_stored_as_null(repo):
    r, _ = repo
    r.get_or_create_patient("SYN_0001")
    v = r.add_visit("SYN_0001", "2020-01-01", {"hba1c": 5.6, "bmi": "", "sex": None})
    assert v.hba1c == 5.6
    assert v.bmi is None
    assert v.sex is None


def test_update_visit_enforces_ownership(repo):
    r, _ = repo
    r.get_or_create_patient("SYN_0001")
    r.get_or_create_patient("SYN_0002")
    v1 = r.add_visit("SYN_0001", "2020-01-01", {"hba1c": 5.6})
    # Updating via the wrong patient must be rejected.
    with pytest.raises(VisitOwnershipError):
        r.update_visit("SYN_0002", v1.id, "2020-01-01", {"hba1c": 9.0})
    # Correct owner can update.
    updated = r.update_visit("SYN_0001", v1.id, None, {"hba1c": 6.1})
    assert updated.hba1c == 6.1


def test_delete_visit_enforces_ownership(repo):
    r, _ = repo
    r.get_or_create_patient("SYN_0001")
    r.get_or_create_patient("SYN_0002")
    v1 = r.add_visit("SYN_0001", "2020-01-01", {"hba1c": 5.6})
    with pytest.raises(VisitOwnershipError):
        r.delete_visit("SYN_0002", v1.id)
    r.delete_visit("SYN_0001", v1.id)
    assert r.list_visits("SYN_0001") == []


def test_get_visits_by_ids_ownership_and_order(repo):
    r, _ = repo
    r.get_or_create_patient("SYN_0001")
    r.get_or_create_patient("SYN_0002")
    a = r.add_visit("SYN_0001", "2020-06-01", {"hba1c": 5.9})
    b = r.add_visit("SYN_0001", "2020-01-01", {"hba1c": 5.6})
    other = r.add_visit("SYN_0002", "2021-01-01", {"hba1c": 7.0})

    # Returns chronological regardless of input order.
    sel = r.get_visits_by_ids("SYN_0001", [a.id, b.id])
    assert [v.visit_date for v in sel] == ["2020-01-01", "2020-06-01"]

    # Another patient's visit id is rejected.
    with pytest.raises(VisitOwnershipError):
        r.get_visits_by_ids("SYN_0001", [other.id])
    # Unknown id is rejected.
    with pytest.raises(VisitNotFoundError):
        r.get_visits_by_ids("SYN_0001", [999999])


def test_list_visits_unknown_patient_raises(repo):
    r, _ = repo
    with pytest.raises(PatientNotFoundError):
        r.list_visits("NOPE")


def test_independent_patient_histories(repo):
    r, _ = repo
    r.get_or_create_patient("SYN_0001")
    r.get_or_create_patient("SYN_0002")
    r.add_visit("SYN_0001", "2020-01-01", {"hba1c": 5.6})
    r.add_visit("SYN_0002", "2021-01-01", {"hba1c": 7.0})
    r.add_visit("SYN_0002", "2021-06-01", {"hba1c": 7.2})
    assert len(r.list_visits("SYN_0001")) == 1
    assert len(r.list_visits("SYN_0002")) == 2


def test_resolve_db_path_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DIALONG_DB_PATH", str(tmp_path / "env.db"))
    assert resolve_db_path(None) == tmp_path / "env.db"
    # Explicit arg wins over env.
    assert resolve_db_path(tmp_path / "explicit.db") == tmp_path / "explicit.db"
