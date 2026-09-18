"""Model registry skeleton for DiaLong-AutoML.

The registry is the single source of truth for which models exist in the
project and what their current implementation status is.  Future phases
register concrete model factories here rather than importing them directly.

Statuses
--------
``available``
    The model is implemented and can be trained.
``planned``
    The model is on the roadmap but not yet implemented.
``planned_not_implemented``
    Explicit marker for models that MUST NOT be accidentally trained.
    Attempting to retrieve the factory for such a model raises
    :class:`NotImplementedError`.

Usage::

    from dialong_automl.models.registry import REGISTRY

    # Check what is available
    available = REGISTRY.list_available()

    # Retrieve a factory (raises for planned_not_implemented entries)
    factory = REGISTRY.get_factory("xgboost")   # Phase 2+: returns callable
    model   = factory(config)

    # Safely inspect an entry without triggering errors
    entry = REGISTRY.get("tcn")
    print(entry.status)   # "planned_not_implemented"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Lifecycle status of a registry entry."""

    AVAILABLE = "available"
    PLANNED = "planned"
    PLANNED_NOT_IMPLEMENTED = "planned_not_implemented"


@dataclass
class RegistryEntry:
    """Metadata and optional factory for a single model.

    Parameters
    ----------
    name:
        Unique model identifier used as the registry key.
    status:
        Current implementation status.
    description:
        Human-readable description.
    factory:
        Callable that constructs a model instance.  ``None`` until the
        model is implemented.
    tags:
        Arbitrary tags for filtering (e.g. ``"tabular"``, ``"sequential"``).
    """

    name: str
    status: ModelStatus
    description: str
    factory: Callable[..., Any] | None = field(default=None, repr=False)
    tags: list[str] = field(default_factory=list)

    def is_available(self) -> bool:
        """Return ``True`` if the model can be instantiated."""
        return self.status == ModelStatus.AVAILABLE

    def is_implemented(self) -> bool:
        """Return ``True`` unless the model is explicitly not-implemented."""
        return self.status != ModelStatus.PLANNED_NOT_IMPLEMENTED


class ModelRegistry:
    """Central registry for DiaLong-AutoML model entries.

    Models are registered at module load time via :meth:`register`.
    The registry is designed to be extended by future phases without
    modifying existing entries.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, entry: RegistryEntry) -> None:
        """Add or replace a :class:`RegistryEntry`.

        Parameters
        ----------
        entry:
            The entry to register.  An existing entry with the same name
            will be overwritten with a warning.
        """
        if entry.name in self._entries:
            logger.warning(
                "ModelRegistry: overwriting existing entry %r", entry.name
            )
        self._entries[entry.name] = entry
        logger.debug("ModelRegistry: registered %r (status=%s)", entry.name, entry.status.value)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> RegistryEntry:
        """Return the :class:`RegistryEntry` for *name*.

        Parameters
        ----------
        name:
            Model identifier.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        try:
            return self._entries[name]
        except KeyError:
            known = list(self._entries)
            raise KeyError(
                f"Model {name!r} is not in the registry.  Known models: {known}"
            ) from None

    def get_factory(self, name: str) -> Callable[..., Any]:
        """Return the factory callable for *name*.

        Parameters
        ----------
        name:
            Model identifier.

        Raises
        ------
        KeyError
            If *name* is not registered.
        NotImplementedError
            If the model's status is ``planned_not_implemented`` or its
            factory is ``None``.
        """
        entry = self.get(name)

        if entry.status == ModelStatus.PLANNED_NOT_IMPLEMENTED:
            raise NotImplementedError(
                f"Model {name!r} is marked as 'planned_not_implemented'. "
                "It cannot be trained in the current phase.  "
                "Do not add its implementation until the relevant phase is started."
            )

        if entry.factory is None:
            raise NotImplementedError(
                f"Model {name!r} has status {entry.status.value!r} "
                "but its factory has not been registered yet.  "
                "This will be resolved in a future implementation phase."
            )

        return entry.factory

    # ------------------------------------------------------------------
    # Listing helpers
    # ------------------------------------------------------------------

    def list_all(self) -> list[str]:
        """Return all registered model names."""
        return list(self._entries)

    def list_available(self) -> list[str]:
        """Return model names whose status is ``available``."""
        return [n for n, e in self._entries.items() if e.is_available()]

    def list_planned(self) -> list[str]:
        """Return model names that are planned but not yet implemented."""
        return [
            n
            for n, e in self._entries.items()
            if e.status in {ModelStatus.PLANNED, ModelStatus.PLANNED_NOT_IMPLEMENTED}
        ]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        summary = {n: e.status.value for n, e in self._entries.items()}
        return f"ModelRegistry({summary})"

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Module-level singleton — populated below
# ---------------------------------------------------------------------------

REGISTRY = ModelRegistry()

# ---------------------------------------------------------------------------
# Phase 1 entries — planned placeholders only.
# Factories are None until the implementing phase registers them.
# ---------------------------------------------------------------------------

REGISTRY.register(
    RegistryEntry(
        name="logistic_regression",
        status=ModelStatus.PLANNED,
        description=(
            "Logistic Regression baseline (scikit-learn).  "
            "Planned for Phase 2."
        ),
        tags=["tabular", "linear", "baseline"],
    )
)

REGISTRY.register(
    RegistryEntry(
        name="random_forest",
        status=ModelStatus.PLANNED,
        description=(
            "Random Forest ensemble (scikit-learn).  "
            "Planned for Phase 2."
        ),
        tags=["tabular", "ensemble", "tree"],
    )
)

REGISTRY.register(
    RegistryEntry(
        name="xgboost",
        status=ModelStatus.PLANNED,
        description=(
            "Gradient-boosted trees via XGBoost.  "
            "Planned for Phase 2."
        ),
        tags=["tabular", "ensemble", "tree", "gradient-boosting"],
    )
)

REGISTRY.register(
    RegistryEntry(
        name="lightgbm",
        status=ModelStatus.PLANNED,
        description=(
            "Gradient-boosted trees via LightGBM.  "
            "Planned for Phase 2."
        ),
        tags=["tabular", "ensemble", "tree", "gradient-boosting"],
    )
)

REGISTRY.register(
    RegistryEntry(
        name="gru",
        status=ModelStatus.PLANNED,
        description=(
            "Gated Recurrent Unit (PyTorch) for longitudinal sequences.  "
            "Planned for Phase 4."
        ),
        tags=["sequential", "rnn", "deep-learning"],
    )
)

# TCN: explicitly planned_not_implemented — must never be accidentally trained.
REGISTRY.register(
    RegistryEntry(
        name="tcn",
        status=ModelStatus.PLANNED_NOT_IMPLEMENTED,
        description=(
            "Temporal Convolutional Network (PyTorch).  "
            "Architecture is on the roadmap but has NOT been designed or "
            "implemented yet.  Do not add a factory until the TCN phase begins."
        ),
        tags=["sequential", "cnn", "deep-learning"],
    )
)
