"""Models sub-package for DiaLong-AutoML.

Phase 1 exposes only the :class:`ModelRegistry` skeleton and the
:data:`REGISTRY` singleton.  Concrete model implementations are added in
later phases.

Usage::

    from dialong_automl.models import REGISTRY

    entry = REGISTRY.get("xgboost")
    print(entry.status)   # "planned" or "available"
"""

from dialong_automl.models.registry import ModelRegistry, RegistryEntry, ModelStatus, REGISTRY

__all__ = ["ModelRegistry", "RegistryEntry", "ModelStatus", "REGISTRY"]
