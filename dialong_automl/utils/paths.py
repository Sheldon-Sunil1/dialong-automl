"""Artifact and output directory utilities for DiaLong-AutoML.

Provides :func:`ensure_dirs` and the :class:`ArtifactPaths` helper that
resolves all project directories from a :class:`~dialong_automl.config.schema.PathsConfig`.

Usage::

    from dialong_automl.config import load_config
    from dialong_automl.utils.paths import ArtifactPaths, ensure_dirs

    cfg = load_config()
    ap = ArtifactPaths.from_config(cfg.paths)
    ap.ensure_all()           # creates every directory if absent
    print(ap.artifacts_dir)   # Path("artifacts")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dialong_automl.config.schema import PathsConfig

logger = logging.getLogger(__name__)


def ensure_dirs(*paths: Path | str) -> list[Path]:
    """Create each directory (and its parents) if it does not already exist.

    Parameters
    ----------
    *paths:
        One or more :class:`~pathlib.Path` objects or path strings.

    Returns
    -------
    list[Path]
        The resolved :class:`~pathlib.Path` objects that were ensured.
    """
    resolved: list[Path] = []
    for raw in paths:
        p = Path(raw)
        p.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", p.resolve())
        resolved.append(p)
    return resolved


class ArtifactPaths:
    """Centralised path resolver for DiaLong-AutoML directories.

    All paths are exposed as :class:`~pathlib.Path` attributes so that
    modules can do::

        ap.artifacts_dir / "xgboost" / "model.pkl"

    rather than building paths ad-hoc.

    Parameters
    ----------
    data_dir:
        Root directory for raw and processed datasets.
    artifacts_dir:
        Root directory for trained model artefacts and checkpoints.
    outputs_dir:
        Root directory for evaluation results, plots, and reports.
    logs_dir:
        Root directory for log files.
    """

    def __init__(
        self,
        data_dir: Path | str = Path("data"),
        artifacts_dir: Path | str = Path("artifacts"),
        outputs_dir: Path | str = Path("outputs"),
        logs_dir: Path | str = Path("logs"),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.outputs_dir = Path(outputs_dir)
        self.logs_dir = Path(logs_dir)

    # ------------------------------------------------------------------
    # Sub-directory helpers (lazy — not created until ensure_all() or
    # individual ensure_* calls)
    # ------------------------------------------------------------------

    @property
    def raw_data_dir(self) -> Path:
        """``data/raw/`` — landing zone for raw input files."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """``data/processed/`` — cleaned and feature-engineered datasets."""
        return self.data_dir / "processed"

    @property
    def model_artifacts_dir(self) -> Path:
        """``artifacts/models/`` — serialised model files."""
        return self.artifacts_dir / "models"

    @property
    def plots_dir(self) -> Path:
        """``outputs/plots/`` — generated figures."""
        return self.outputs_dir / "plots"

    @property
    def reports_dir(self) -> Path:
        """``outputs/reports/`` — evaluation reports and metrics."""
        return self.outputs_dir / "reports"

    # ------------------------------------------------------------------
    # Directory creation helpers
    # ------------------------------------------------------------------

    def ensure_all(self) -> None:
        """Create every standard project directory if absent."""
        ensure_dirs(
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.artifacts_dir,
            self.model_artifacts_dir,
            self.outputs_dir,
            self.plots_dir,
            self.reports_dir,
            self.logs_dir,
        )
        logger.info("All project directories ensured.")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, paths_cfg: PathsConfig) -> ArtifactPaths:
        """Construct an :class:`ArtifactPaths` from a :class:`PathsConfig`.

        Parameters
        ----------
        paths_cfg:
            The ``paths`` section of the loaded :class:`AppConfig`.
        """
        return cls(
            data_dir=paths_cfg.data_dir,
            artifacts_dir=paths_cfg.artifacts_dir,
            outputs_dir=paths_cfg.outputs_dir,
            logs_dir=paths_cfg.logs_dir,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ArtifactPaths("
            f"data={self.data_dir}, "
            f"artifacts={self.artifacts_dir}, "
            f"outputs={self.outputs_dir}, "
            f"logs={self.logs_dir})"
        )
