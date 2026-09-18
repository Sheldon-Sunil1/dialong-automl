"""Configuration sub-package for DiaLong-AutoML.

Exports the primary config loader and the top-level :class:`AppConfig` model
so callers can do::

    from dialong_automl.config import load_config, AppConfig
"""

from dialong_automl.config.loader import load_config
from dialong_automl.config.schema import AppConfig

__all__ = ["load_config", "AppConfig"]
