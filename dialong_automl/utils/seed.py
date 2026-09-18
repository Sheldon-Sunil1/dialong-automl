"""Reproducibility utilities for DiaLong-AutoML.

Provides a single entry-point :func:`set_global_seed` that seeds Python's
built-in ``random`` module, ``numpy``, and (when available) ``torch``.

Usage::

    from dialong_automl.utils.seed import set_global_seed

    set_global_seed(42)
    # or, reading from config:
    set_global_seed(cfg.reproducibility.seed)
"""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)

_DEFAULT_SEED: int = 42


def set_global_seed(
    seed: int = _DEFAULT_SEED,
    *,
    deterministic_torch: bool = True,
    benchmark_cudnn: bool = False,
) -> int:
    """Set the random seed for Python, NumPy, and PyTorch (if available).

    Parameters
    ----------
    seed:
        Integer seed value.  Defaults to ``42``.
    deterministic_torch:
        When ``True`` (and PyTorch is importable), calls
        ``torch.use_deterministic_algorithms(True)`` and sets
        ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` required by CUDA ≥ 10.2.
        Disable if you hit unsupported-op errors and do not need strict
        reproducibility.
    benchmark_cudnn:
        Set ``torch.backends.cudnn.benchmark``.  Keeping this ``False``
        ensures deterministic cuDNN behaviour at the cost of some throughput.

    Returns
    -------
    int
        The seed that was applied (useful for logging / assertions).

    Notes
    -----
    PyTorch and NumPy imports are lazy so this function remains importable
    even if those packages are not installed.
    """
    # Python built-in random
    random.seed(seed)

    # Environment variable used by hash-randomisation and some C extensions
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)
        logger.debug("NumPy seed set to %d", seed)
    except ImportError:
        logger.debug("NumPy not available — skipping NumPy seed.")

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = benchmark_cudnn
        torch.backends.cudnn.deterministic = not benchmark_cudnn

        if deterministic_torch:
            # CUBLAS env var is required for deterministic CUDA operations.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            logger.debug("torch.use_deterministic_algorithms(True) enabled.")

        logger.debug("PyTorch seed set to %d", seed)
    except ImportError:
        logger.debug("PyTorch not available — skipping torch seed.")

    logger.info("Global random seed set to %d", seed)
    return seed
