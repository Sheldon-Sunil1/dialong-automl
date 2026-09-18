"""YAML configuration loader for DiaLong-AutoML.

Usage::

    from dialong_automl.config import load_config

    cfg = load_config()                          # uses configs/demo_fast.yaml by default
    cfg = load_config("configs/my_config.yaml")  # explicit path
    cfg = load_config(None)                      # pure defaults (no file required)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from dialong_automl.config.schema import AppConfig

logger = logging.getLogger(__name__)

# Default config file path relative to the repository root.
_DEFAULT_CONFIG: Path = Path("configs") / "demo_fast.yaml"


def _find_repo_root() -> Path:
    """Walk up from this file to find the repo root (contains pyproject.toml)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: working directory
    return Path.cwd()


def load_config(config_path: str | Path | None = _DEFAULT_CONFIG) -> AppConfig:
    """Load and validate the application configuration.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.  Can be absolute or relative
        (resolved against the repository root).  Pass ``None`` to skip file
        loading and return an :class:`AppConfig` built from defaults only.

    Returns
    -------
    AppConfig
        A validated Pydantic model containing the merged configuration.

    Raises
    ------
    FileNotFoundError
        If *config_path* is provided but does not exist.
    yaml.YAMLError
        If the YAML file is malformed.
    pydantic.ValidationError
        If the configuration values fail schema validation.
    """
    raw: dict[str, Any] = {}

    if config_path is not None:
        path = Path(config_path)
        if not path.is_absolute():
            path = _find_repo_root() / path

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}\n"
                "Create one or pass config_path=None to use defaults."
            )

        logger.debug("Loading configuration from %s", path)
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    # Environment-variable override: DIALONG_CONFIG_PATH
    env_path = os.environ.get("DIALONG_CONFIG_PATH")
    if env_path and config_path is None:
        env_file = Path(env_path)
        if env_file.exists():
            logger.debug("Loading configuration from env DIALONG_CONFIG_PATH=%s", env_file)
            with env_file.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}

    config = AppConfig.model_validate(raw)
    logger.info(
        "Configuration loaded: project=%s version=%s seed=%d",
        config.project_name,
        config.version,
        config.reproducibility.seed,
    )
    return config
