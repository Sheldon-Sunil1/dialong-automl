"""
DiaLong-AutoML
==============
Explainable Longitudinal AutoML for Diabetes Progression Prediction.

Phase 1 — Foundation.
"""

from importlib.metadata import PackageNotFoundError, version

__author__ = "DiaLong-AutoML Contributors"
__license__ = "MIT"

try:
    __version__ = version("dialong-automl")
except PackageNotFoundError:
    # Package is not installed (e.g. running from source tree)
    __version__ = "0.1.0-dev"

__all__ = ["__version__", "__author__", "__license__"]
