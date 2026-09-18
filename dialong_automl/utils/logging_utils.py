"""Logging infrastructure for DiaLong-AutoML.

Provides :func:`configure_logging` to initialise the root logger from an
:class:`~dialong_automl.config.schema.LoggingConfig` instance, and
:func:`get_logger` as a thin convenience wrapper around
``logging.getLogger``.

Usage::

    from dialong_automl.utils.logging_utils import configure_logging, get_logger
    from dialong_automl.config import load_config

    cfg = load_config()
    configure_logging(cfg.logging, logs_dir=cfg.paths.logs_dir)

    logger = get_logger(__name__)
    logger.info("Ready.")
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dialong_automl.config.schema import LoggingConfig

# Package-level logger (used internally by this module only)
_internal = logging.getLogger(__name__)

_CONFIGURED: bool = False


def configure_logging(
    cfg: LoggingConfig,
    logs_dir: Path | None = None,
    *,
    force: bool = False,
) -> None:
    """Configure the root logger from a :class:`LoggingConfig`.

    Parameters
    ----------
    cfg:
        Logging configuration section from the loaded :class:`AppConfig`.
    logs_dir:
        Directory where rotating log files are written when
        ``cfg.log_to_file`` is ``True``.  Defaults to ``Path("logs")``.
    force:
        Re-apply configuration even if already applied this process.

    Notes
    -----
    This function is idempotent by default: calling it a second time without
    *force=True* has no effect, preventing duplicate handlers when modules
    import each other during testing.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(cfg.level)

    formatter = logging.Formatter(cfg.format)

    # Remove any pre-existing handlers to avoid duplication on re-configure.
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Console handler — always present.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Optional rotating file handler.
    if cfg.log_to_file:
        _logs_dir = logs_dir or Path("logs")
        _logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = _logs_dir / cfg.log_filename
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        _internal.debug("File logging enabled: %s", log_file)

    _CONFIGURED = True
    _internal.debug("Logging configured at level %s", cfg.level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Thin wrapper around :func:`logging.getLogger` so that callers only need
    a single import from the project's utility layer.

    Parameters
    ----------
    name:
        Logger name — conventionally ``__name__`` of the calling module.
    """
    return logging.getLogger(name)
