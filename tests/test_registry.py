"""Tests 6 & 7 — Model registry behaviour and TCN planned_not_implemented.

Verifies:
- REGISTRY is a ModelRegistry instance
- All expected model names are registered
- Status values are correct
- get_factory raises for planned-not-implemented entries
- TCN cannot be accidentally trained
"""

from __future__ import annotations

import pytest

from dialong_automl.models.registry import (
    REGISTRY,
    ModelRegistry,
    ModelStatus,
    RegistryEntry,
)


# ---------------------------------------------------------------------------
# Registry type and size
# ---------------------------------------------------------------------------


def test_registry_is_model_registry() -> None:
    """REGISTRY is a ModelRegistry instance."""
    assert isinstance(REGISTRY, ModelRegistry)


def test_registry_has_six_entries() -> None:
    """REGISTRY has exactly 6 entries (lr, rf, xgb, lgbm, gru, tcn)."""
    assert len(REGISTRY) == 6


# ---------------------------------------------------------------------------
# All expected models are registered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["logistic_regression", "random_forest", "xgboost", "lightgbm", "gru", "tcn"],
)
def test_model_is_registered(name: str) -> None:
    """Each expected model name is in the registry."""
    entry = REGISTRY.get(name)
    assert entry.name == name


# ---------------------------------------------------------------------------
# Status values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["logistic_regression", "random_forest", "xgboost", "lightgbm", "gru"],
)
def test_tabular_and_sequential_models_are_planned(name: str) -> None:
    """Non-TCN models have status PLANNED (not yet available)."""
    entry = REGISTRY.get(name)
    assert entry.status == ModelStatus.PLANNED


def test_tcn_is_planned_not_implemented() -> None:
    """TCN has status PLANNED_NOT_IMPLEMENTED."""
    entry = REGISTRY.get("tcn")
    assert entry.status == ModelStatus.PLANNED_NOT_IMPLEMENTED


def test_no_models_available_in_phase_1() -> None:
    """No models are available (factories registered) in Phase 1."""
    assert REGISTRY.list_available() == []


def test_all_models_in_planned_list() -> None:
    """All 6 models appear in list_planned()."""
    planned = REGISTRY.list_planned()
    assert set(planned) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "gru",
        "tcn",
    }


# ---------------------------------------------------------------------------
# TCN — must not be trainable
# ---------------------------------------------------------------------------


def test_tcn_get_factory_raises_not_implemented() -> None:
    """get_factory('tcn') raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        REGISTRY.get_factory("tcn")


def test_tcn_is_not_implemented_method() -> None:
    """TCN entry.is_implemented() returns False."""
    entry = REGISTRY.get("tcn")
    assert entry.is_implemented() is False


def test_tcn_is_not_available() -> None:
    """TCN entry.is_available() returns False."""
    entry = REGISTRY.get("tcn")
    assert entry.is_available() is False


def test_tcn_not_in_list_available() -> None:
    """TCN does not appear in REGISTRY.list_available()."""
    assert "tcn" not in REGISTRY.list_available()


def test_tcn_factory_is_none() -> None:
    """TCN entry has no factory registered."""
    entry = REGISTRY.get("tcn")
    assert entry.factory is None


# ---------------------------------------------------------------------------
# Planned (non-TCN) models also lack factories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["logistic_regression", "random_forest", "xgboost", "lightgbm", "gru"],
)
def test_planned_model_factory_raises_not_implemented(name: str) -> None:
    """get_factory on a PLANNED model with no factory raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        REGISTRY.get_factory(name)


# ---------------------------------------------------------------------------
# Unknown model
# ---------------------------------------------------------------------------


def test_unknown_model_raises_key_error() -> None:
    """get() on an unregistered name raises KeyError."""
    with pytest.raises(KeyError):
        REGISTRY.get("nonexistent_model")


# ---------------------------------------------------------------------------
# Registry mutation (isolated — uses a fresh registry)
# ---------------------------------------------------------------------------


def test_register_new_entry() -> None:
    """A fresh registry accepts new entries correctly."""
    reg = ModelRegistry()
    entry = RegistryEntry(
        name="test_model",
        status=ModelStatus.AVAILABLE,
        description="Test model",
        factory=lambda: None,
    )
    reg.register(entry)
    assert "test_model" in reg.list_all()
    assert "test_model" in reg.list_available()


def test_register_overwrites_existing(caplog: pytest.LogCaptureFixture) -> None:
    """Registering a duplicate name overwrites with a warning."""
    import logging

    reg = ModelRegistry()
    e1 = RegistryEntry(name="m", status=ModelStatus.PLANNED, description="first")
    e2 = RegistryEntry(name="m", status=ModelStatus.AVAILABLE, description="second")

    reg.register(e1)
    with caplog.at_level(logging.WARNING):
        reg.register(e2)

    assert reg.get("m").status == ModelStatus.AVAILABLE
    assert any("overwriting" in r.message.lower() for r in caplog.records)


def test_entry_tags_stored() -> None:
    """RegistryEntry stores tags correctly."""
    entry = RegistryEntry(
        name="tagged",
        status=ModelStatus.PLANNED,
        description="Tagged model",
        tags=["tabular", "tree"],
    )
    assert "tabular" in entry.tags
    assert "tree" in entry.tags
