"""Explanation-stability utilities for DiaLong-AutoML (Phase 11).

Quantifies how similar two explanations are. These are generic, method-agnostic
helpers: they operate on either a ``{key: score}`` mapping or an ordered list of
keys. They work equally for SHAP feature importances
(``GlobalExplanation.features`` → ``{column_name: importance}``) and GRU temporal
occlusion (``TemporalExplanation.timesteps`` → ``{timestep: influence}``). No
assumption is made that both methods share a representation.

Metrics
-------
* ``topk_jaccard``       — overlap of the top-k keys (by score magnitude).
                           Range ``[0, 1]``; 1 = identical top-k sets.
* ``spearman_rank_corr`` — Spearman rank correlation over the shared keys.
                           Range ``[-1, 1]``; ``None`` when undefined (constant
                           ranking, < 2 shared keys).

Pure NumPy, deterministic. Spearman uses average ranks for ties.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _as_score_map(explanation: Mapping[Any, float] | Sequence[Any]) -> dict[Any, float]:
    """Normalize an explanation to a ``{key: score}`` dict.

    * Mapping            → used as-is (values are scores).
    * Sequence of keys   → treated as a ranking; score = descending position
                           (first item highest), so magnitude ordering matches.
    """
    if isinstance(explanation, Mapping):
        return {k: float(v) for k, v in explanation.items()}
    seq = list(explanation)
    n = len(seq)
    # Highest score for the first (most important) key.
    return {k: float(n - i) for i, k in enumerate(seq)}


def _top_k_keys(score_map: dict[Any, float], k: int) -> set[Any]:
    """Keys of the k largest scores by MAGNITUDE (|score|), deterministic ties.

    Ties are broken by the key's string form for determinism, so the result is
    independent of dict insertion order.
    """
    items = sorted(score_map.items(), key=lambda kv: (-abs(kv[1]), str(kv[0])))
    return {k_ for k_, _ in items[:k]}


def topk_jaccard(
    a: Mapping[Any, float] | Sequence[Any],
    b: Mapping[Any, float] | Sequence[Any],
    k: int,
) -> float:
    """Jaccard similarity of the top-k keys of two explanations.

    Range ``[0, 1]``. If both top-k sets are empty (e.g. ``k == 0``), returns
    ``1.0`` (vacuously identical). Keys absent from one explanation simply do
    not appear in its top-k set.
    """
    if k < 0:
        raise ValueError("k must be >= 0.")
    sa = _top_k_keys(_as_score_map(a), k)
    sb = _top_k_keys(_as_score_map(b), k)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return float(len(sa & sb) / len(union))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average (fractional) ranks, handling ties — like scipy's 'average'."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        # Positions i..j are tied → assign their average rank (1-based).
        avg = (i + j) / 2.0 + 1.0
        for p in range(i, j + 1):
            ranks[order[p]] = avg
        i = j + 1
    return ranks


def spearman_rank_corr(
    a: Mapping[Any, float] | Sequence[Any],
    b: Mapping[Any, float] | Sequence[Any],
) -> float | None:
    """Spearman rank correlation over the keys present in BOTH explanations.

    Returns a value in ``[-1, 1]``, or ``None`` when undefined:
    * fewer than 2 shared keys, or
    * either ranking is constant over the shared keys (zero variance).

    Keys missing from one side are excluded from the shared set (reported via
    the companion :func:`shared_key_count`). Ties use average ranks.
    """
    ma = _as_score_map(a)
    mb = _as_score_map(b)
    shared = [k for k in ma if k in mb]
    if len(shared) < 2:
        return None
    va = np.array([ma[k] for k in shared], dtype=np.float64)
    vb = np.array([mb[k] for k in shared], dtype=np.float64)
    ra = _average_ranks(va)
    rb = _average_ranks(vb)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return None  # constant ranking → correlation undefined
    # Pearson correlation of the ranks (valid Spearman with tie handling).
    ra_c = ra - ra.mean()
    rb_c = rb - rb.mean()
    denom = np.sqrt((ra_c**2).sum() * (rb_c**2).sum())
    if denom == 0.0:
        return None
    corr = float((ra_c * rb_c).sum() / denom)
    return max(-1.0, min(1.0, corr))


def shared_key_count(
    a: Mapping[Any, float] | Sequence[Any],
    b: Mapping[Any, float] | Sequence[Any],
) -> int:
    """Number of keys present in both explanations."""
    ma = _as_score_map(a)
    mb = _as_score_map(b)
    return sum(1 for k in ma if k in mb)


def stability_report(
    a: Mapping[Any, float] | Sequence[Any],
    b: Mapping[Any, float] | Sequence[Any],
    k: int = 5,
) -> dict[str, Any]:
    """Bundle of stability metrics between two explanations."""
    return {
        "k": k,
        "shared_keys": shared_key_count(a, b),
        "topk_jaccard": topk_jaccard(a, b, k),
        "spearman": spearman_rank_corr(a, b),
        "note": "spearman is None when undefined (constant ranking or <2 shared keys).",
    }
