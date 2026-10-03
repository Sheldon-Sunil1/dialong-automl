"""Models sub-package for DiaLong-AutoML.

Phase 5 registers functional factories for all implemented models.
TCN remains PLANNED_NOT_IMPLEMENTED.

Heavy backends (xgboost, lightgbm, torch) are imported lazily so a blocked
compiled dependency cannot prevent using the other models.

Usage::

    from dialong_automl.models import REGISTRY, build_model

    model = build_model("xgboost", {"n_estimators": 200})
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)
"""

from __future__ import annotations

from typing import Any

from dialong_automl.models.base import (
    REPRESENTATION_SEQUENCE,
    REPRESENTATION_WIDE,
    BaseModel,
)
from dialong_automl.models.registry import (
    REGISTRY,
    ModelRegistry,
    ModelStatus,
    RegistryEntry,
)

_LAZY_MODELS = {
    "LogisticRegressionModel": "dialong_automl.models.logistic_regression",
    "RandomForestModel": "dialong_automl.models.random_forest",
    "XGBoostModel": "dialong_automl.models.xgboost_model",
    "LightGBMModel": "dialong_automl.models.lightgbm_model",
    "GRUModel": "dialong_automl.models.gru",
}


def build_model(name: str, hyperparams: dict[str, Any] | None = None) -> BaseModel:
    """Instantiate a model by registry name.

    Parameters
    ----------
    name:
        Registry key, e.g. ``"xgboost"``.
    hyperparams:
        Optional hyperparameter dict passed to the constructor.

    Returns
    -------
    BaseModel
        An unfitted model instance.

    Raises
    ------
    NotImplementedError
        If *name* is ``tcn`` or otherwise not implemented.
    ImportError
        If the model's required backend cannot be imported.
    """
    factory = REGISTRY.get_factory(name)
    return factory(hyperparams=hyperparams)


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODELS:
        import importlib

        module = importlib.import_module(_LAZY_MODELS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "REGISTRY",
    "ModelRegistry",
    "RegistryEntry",
    "ModelStatus",
    "BaseModel",
    "REPRESENTATION_WIDE",
    "REPRESENTATION_SEQUENCE",
    "LogisticRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "GRUModel",
    "build_model",
]
