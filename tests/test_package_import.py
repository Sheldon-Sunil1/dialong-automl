"""Test 1 — Package import.

Verifies that the top-level ``dialong_automl`` package and each sub-package
import without errors and expose the expected public attributes.
"""

import importlib


def test_top_level_import() -> None:
    """dialong_automl imports cleanly."""
    import dialong_automl  # noqa: F401 — import-only test


def test_version_attribute() -> None:
    """dialong_automl.__version__ is a non-empty string."""
    import dialong_automl

    assert isinstance(dialong_automl.__version__, str)
    assert len(dialong_automl.__version__) > 0


def test_subpackage_config() -> None:
    """dialong_automl.config imports cleanly."""
    importlib.import_module("dialong_automl.config")


def test_subpackage_utils() -> None:
    """dialong_automl.utils imports cleanly."""
    importlib.import_module("dialong_automl.utils")


def test_subpackage_models() -> None:
    """dialong_automl.models imports cleanly."""
    importlib.import_module("dialong_automl.models")


def test_subpackage_data() -> None:
    """dialong_automl.data imports cleanly."""
    importlib.import_module("dialong_automl.data")


def test_subpackage_features() -> None:
    """dialong_automl.features imports cleanly."""
    importlib.import_module("dialong_automl.features")


def test_subpackage_explainability() -> None:
    """dialong_automl.explainability imports cleanly."""
    importlib.import_module("dialong_automl.explainability")


def test_subpackage_evaluation() -> None:
    """dialong_automl.evaluation imports cleanly."""
    importlib.import_module("dialong_automl.evaluation")


def test_subpackage_api() -> None:
    """dialong_automl.api imports cleanly."""
    importlib.import_module("dialong_automl.api")
