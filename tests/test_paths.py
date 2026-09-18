"""Test 5 — Artifact/output directory path handling.

Verifies :class:`~dialong_automl.utils.paths.ArtifactPaths` and
:func:`~dialong_automl.utils.paths.ensure_dirs`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dialong_automl.utils.paths import ArtifactPaths, ensure_dirs


# ---------------------------------------------------------------------------
# ArtifactPaths defaults
# ---------------------------------------------------------------------------


def test_default_paths_are_path_objects() -> None:
    """All ArtifactPaths attributes are pathlib.Path instances."""
    ap = ArtifactPaths()
    assert isinstance(ap.data_dir, Path)
    assert isinstance(ap.artifacts_dir, Path)
    assert isinstance(ap.outputs_dir, Path)
    assert isinstance(ap.logs_dir, Path)


def test_default_dir_names() -> None:
    """Default directory names match project conventions."""
    ap = ArtifactPaths()
    assert ap.data_dir == Path("data")
    assert ap.artifacts_dir == Path("artifacts")
    assert ap.outputs_dir == Path("outputs")
    assert ap.logs_dir == Path("logs")


def test_sub_directory_properties() -> None:
    """Sub-directory properties compose correctly."""
    ap = ArtifactPaths()
    assert ap.raw_data_dir == Path("data") / "raw"
    assert ap.processed_data_dir == Path("data") / "processed"
    assert ap.model_artifacts_dir == Path("artifacts") / "models"
    assert ap.plots_dir == Path("outputs") / "plots"
    assert ap.reports_dir == Path("outputs") / "reports"


def test_custom_paths() -> None:
    """ArtifactPaths accepts custom directory arguments."""
    ap = ArtifactPaths(
        data_dir="/tmp/mydata",
        artifacts_dir="/tmp/myartifacts",
        outputs_dir="/tmp/myoutputs",
        logs_dir="/tmp/mylogs",
    )
    assert ap.data_dir == Path("/tmp/mydata")
    assert ap.artifacts_dir == Path("/tmp/myartifacts")


def test_string_paths_converted_to_path() -> None:
    """String arguments are converted to Path objects."""
    ap = ArtifactPaths(data_dir="some/data")
    assert isinstance(ap.data_dir, Path)


# ---------------------------------------------------------------------------
# from_config factory
# ---------------------------------------------------------------------------


def test_from_config_factory() -> None:
    """ArtifactPaths.from_config constructs from a PathsConfig."""
    from dialong_automl.config import load_config

    cfg = load_config(None)
    ap = ArtifactPaths.from_config(cfg.paths)

    assert ap.data_dir == cfg.paths.data_dir
    assert ap.artifacts_dir == cfg.paths.artifacts_dir
    assert ap.outputs_dir == cfg.paths.outputs_dir
    assert ap.logs_dir == cfg.paths.logs_dir


# ---------------------------------------------------------------------------
# ensure_dirs
# ---------------------------------------------------------------------------


def test_ensure_dirs_creates_directory(tmp_path: Path) -> None:
    """ensure_dirs creates a directory that does not exist."""
    new_dir = tmp_path / "new_directory"
    assert not new_dir.exists()
    ensure_dirs(new_dir)
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_ensure_dirs_is_idempotent(tmp_path: Path) -> None:
    """ensure_dirs on an existing directory does not raise."""
    existing = tmp_path / "existing"
    existing.mkdir()
    ensure_dirs(existing)  # should not raise
    assert existing.is_dir()


def test_ensure_dirs_creates_nested(tmp_path: Path) -> None:
    """ensure_dirs creates deeply nested directories."""
    deep = tmp_path / "a" / "b" / "c" / "d"
    ensure_dirs(deep)
    assert deep.is_dir()


def test_ensure_dirs_returns_paths(tmp_path: Path) -> None:
    """ensure_dirs returns a list of Path objects."""
    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    result = ensure_dirs(d1, d2)
    assert len(result) == 2
    assert all(isinstance(p, Path) for p in result)


def test_ensure_dirs_accepts_strings(tmp_path: Path) -> None:
    """ensure_dirs accepts string paths as well as Path objects."""
    new_dir = tmp_path / "str_dir"
    ensure_dirs(str(new_dir))
    assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# ensure_all
# ---------------------------------------------------------------------------


def test_ensure_all_creates_expected_dirs(tmp_path: Path) -> None:
    """ArtifactPaths.ensure_all() creates all standard directories."""
    ap = ArtifactPaths(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        outputs_dir=tmp_path / "outputs",
        logs_dir=tmp_path / "logs",
    )
    ap.ensure_all()

    expected = [
        ap.data_dir,
        ap.raw_data_dir,
        ap.processed_data_dir,
        ap.artifacts_dir,
        ap.model_artifacts_dir,
        ap.outputs_dir,
        ap.plots_dir,
        ap.reports_dir,
        ap.logs_dir,
    ]
    for d in expected:
        assert d.is_dir(), f"Expected directory not created: {d}"
