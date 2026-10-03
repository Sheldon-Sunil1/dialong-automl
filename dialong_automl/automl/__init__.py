"""AutoML sub-package for DiaLong-AutoML — Phase 5.

Provides the Optuna-based mixed-model search optimizer.

Critical scientific rule
------------------------
Optuna optimizes ONLY on the validation set.
The test set must NEVER be accessed during search, model selection,
threshold selection, or preprocessing fitting.

Usage::

    from dialong_automl.automl import AutoMLOptimizer
"""

from dialong_automl.automl.optimizer import AutoMLOptimizer

__all__ = ["AutoMLOptimizer"]
