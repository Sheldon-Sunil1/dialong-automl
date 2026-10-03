"""Joblib/pickle persistence helpers for sklearn-style model wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def dump_fitted(obj: Any, path: str | Path) -> None:
    """Serialize *obj* to *path* using joblib when available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
    except ImportError:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        return
    joblib.dump(obj, path)


def load_fitted(path: str | Path) -> Any:
    """Load an object previously written by :func:`dump_fitted`."""
    path = Path(path)
    try:
        import joblib
    except ImportError:
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)
    return joblib.load(path)
