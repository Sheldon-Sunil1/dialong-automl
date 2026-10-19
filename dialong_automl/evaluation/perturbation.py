"""Deterministic perturbation interface for robustness evaluation (Phase 11).

Operates on the project's canonical **long-format observation DataFrame**
(one row per visit, with ``patient_id``, ``visit_date``, and clinical feature
columns) — the exact contract consumed by ``WideRepresentationBuilder`` and
``SequenceRepresentationBuilder``. Perturbing at this layer means the downstream
representation/preprocessing is unchanged and the fixed feature layout is
preserved (missing measurements become NaN, which the fitted preprocessor
already imputes).

Guarantees (all enforced and unit-tested)
----------------------------------------
* The source DataFrame is never mutated (a copy is returned).
* ``patient_id`` values and split membership are preserved — rows are only
  blanked (measurements) or dropped (whole visits); no patient is added,
  removed, or moved.
* Labels are NEVER touched (labels live in a separate Series; this module only
  takes/returns observation features).
* No future information is introduced; perturbations only remove/alter existing
  observation data.
* The clinical feature column set (layout) is preserved — columns are never
  added or removed; measurements are set to NaN in place.
* Deterministic given a seed.

This module provides the *operators* only; it does not run a research
experiment or call the model.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from dialong_automl.features.wide import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

_CLINICAL_COLUMNS: tuple[str, ...] = tuple(NUMERIC_FEATURES) + tuple(CATEGORICAL_FEATURES)


def _stable_token(*parts: Any) -> int:
    """Derive a stable, process-independent 64-bit integer from the given parts.

    Uses BLAKE2b over the UTF-8 joined string form of ``parts``. Unlike Python's
    built-in ``hash()``, this is deterministic across processes and runs (the
    built-in is randomized per-process for ``str``). The returned token is used
    to seed per-entity NumPy generators so that an entity's random stream is a
    pure function of its own identity plus the user seed — never of unrelated
    rows/patients or DataFrame ordering.
    """
    joined = "\x1f".join(str(p) for p in parts)
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _patient_rng(seed: int, kind: str, patient_id: Any) -> np.random.Generator:
    """A NumPy Generator seeded deterministically from (seed, kind, patient_id).

    The ``kind`` (perturbation-condition name) is folded in so the same patient
    under two different operators does not share a stream. Independence from
    unrelated patients and from row/column ordering is the key invariant.
    """
    token = _stable_token(seed, kind, patient_id)
    # SeedSequence accepts a list of non-negative ints; combine seed + token.
    ss = np.random.SeedSequence([int(seed) & 0xFFFFFFFFFFFFFFFF, token])
    return np.random.default_rng(ss)


@dataclass
class PerturbationResult:
    """A perturbed copy of the observation data plus explicit condition metadata."""

    observation_data: pd.DataFrame
    condition: dict[str, Any] = field(default_factory=dict)


def _check_contract(df: pd.DataFrame) -> None:
    for col in ("patient_id", "visit_date"):
        if col not in df.columns:
            raise ValueError(f"observation_data must contain a {col!r} column.")


def _present_clinical(df: pd.DataFrame) -> list[str]:
    return [c for c in _CLINICAL_COLUMNS if c in df.columns]


def remove_measurements(
    observation_data: pd.DataFrame,
    fraction: float,
    *,
    seed: int = 42,
    columns: list[str] | None = None,
) -> PerturbationResult:
    """Randomly blank a *fraction* of individual clinical measurements (set NaN).

    Whole visits (rows) are kept; only selected cell values are set to NaN,
    simulating missing measurements. Column layout is preserved.

    Determinism / invariance
    -------------------------
    The blank/keep decision for a single cell is drawn from a random stream
    seeded by ``(seed, "remove_measurements", patient_id)`` and then indexed by
    the ``(visit_date, feature_name)`` of that cell. Consequently:

    * The same patient with the same visits and features receives the **same**
      mask regardless of DataFrame row order, column order, or the presence of
      other (unrelated) patients.
    * Each (patient, visit, feature) cell has an independent, reproducible
      Bernoulli(``fraction``) blank decision.

    Note: the set of ``columns`` actually considered still determines which
    features *could* be blanked, but the decision per retained (patient, visit,
    feature) is invariant to the ordering of ``columns``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1].")
    _check_contract(observation_data)
    out = observation_data.copy(deep=True)
    cols = columns if columns is not None else _present_clinical(out)
    cols = [c for c in cols if c in out.columns]

    n_cells = 0
    n_blanked = 0
    # Each (patient, visit, feature) cell draws an independent uniform seeded by
    # the cell's stable identity. This makes the blank/keep decision invariant to
    # row order, column order, and the presence of unrelated patients.
    for pid, grp in out.groupby("patient_id", sort=True):
        for idx in grp.index:
            visit_key = out.at[idx, "visit_date"]
            for col in cols:
                n_cells += 1
                cell_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [_stable_token(seed, "remove_measurements", pid, visit_key, col)]
                    )
                )
                if float(cell_rng.random()) < fraction:
                    out.at[idx, col] = np.nan
                    n_blanked += 1

    return PerturbationResult(
        observation_data=out,
        condition={
            "kind": "remove_measurements",
            "strength": float(fraction),
            "params": {
                "seed": seed,
                "columns": list(cols),
                "cells_total": int(n_cells),
                "cells_blanked": int(n_blanked),
            },
        },
    )


def remove_visits(
    observation_data: pd.DataFrame,
    fraction: float,
    *,
    seed: int = 42,
    min_visits_kept: int = 1,
) -> PerturbationResult:
    """Randomly drop whole visits (rows), per patient, keeping >= ``min_visits_kept``.

    Patients are never removed entirely: each patient retains at least
    ``min_visits_kept`` of their earliest-to-latest visits. Patient IDs and the
    set of patients are preserved.

    Determinism / invariance
    -------------------------
    Each patient's visit-drop pattern is drawn from a random stream seeded by
    ``(seed, "remove_visits", patient_id)``. The stream is a pure function of
    the patient's own identity plus ``seed`` and does **not** depend on how many
    visits other patients have, nor on whether unrelated patients are added or
    removed from the cohort. (The previous implementation shared one generator
    across the per-patient loop, so a patient's pattern depended on preceding
    patients' visit counts — that cross-cohort coupling is now removed.)
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1].")
    if min_visits_kept < 1:
        raise ValueError("min_visits_kept must be >= 1.")
    _check_contract(observation_data)
    out = observation_data.copy(deep=True)

    keep_index: list[Any] = []
    dropped = 0
    # Stable per-patient processing for determinism.
    for pid, grp in out.groupby("patient_id", sort=True):
        grp_sorted = grp.sort_values("visit_date")
        n = len(grp_sorted)
        n_drop = int(np.floor(n * fraction))
        n_drop = min(n_drop, max(0, n - min_visits_kept))
        if n_drop <= 0:
            keep_index.extend(grp_sorted.index.tolist())
            continue
        # Per-patient stream → independent of other patients / cohort size.
        rng = _patient_rng(seed, "remove_visits", pid)
        drop_pos = rng.choice(n, size=n_drop, replace=False)
        drop_set = set(drop_pos.tolist())
        for pos, idx in enumerate(grp_sorted.index.tolist()):
            if pos in drop_set:
                dropped += 1
            else:
                keep_index.append(idx)

    perturbed = out.loc[keep_index].sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    return PerturbationResult(
        observation_data=perturbed,
        condition={
            "kind": "remove_visits",
            "strength": float(fraction),
            "params": {
                "seed": seed,
                "min_visits_kept": min_visits_kept,
                "visits_dropped": int(dropped),
                "patients_before": int(out["patient_id"].nunique()),
                "patients_after": int(perturbed["patient_id"].nunique()),
            },
        },
    )


def truncate_to_recent_visits(
    observation_data: pd.DataFrame,
    keep_last_n: int,
) -> PerturbationResult:
    """Keep only each patient's most recent ``keep_last_n`` visits (recency truncation).

    This is a **recency truncation**, not a change to follow-up *timing*: it
    retains the latest ``keep_last_n`` visits per patient and does not modify
    any ``visit_date`` or simulate irregular inter-visit intervals. For a
    timing perturbation see :func:`jitter_visit_timing`.

    Fully deterministic — it involves no randomness (there is no ``seed``
    parameter). Patients with <= ``keep_last_n`` visits are unchanged; all
    patients are retained.
    """
    if keep_last_n < 1:
        raise ValueError("keep_last_n must be >= 1.")
    _check_contract(observation_data)
    out = observation_data.copy(deep=True)

    keep_index: list[Any] = []
    for _pid, grp in out.groupby("patient_id", sort=True):
        grp_sorted = grp.sort_values("visit_date")
        keep_index.extend(grp_sorted.index.tolist()[-keep_last_n:])

    perturbed = out.loc[keep_index].sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    return PerturbationResult(
        observation_data=perturbed,
        condition={
            "kind": "truncate_to_recent_visits",
            "strength": float(keep_last_n),
            "params": {
                "keep_last_n": keep_last_n,
                "patients_before": int(out["patient_id"].nunique()),
                "patients_after": int(perturbed["patient_id"].nunique()),
            },
        },
    )


def jitter_visit_timing(
    observation_data: pd.DataFrame,
    max_shift_days: int,
    *,
    seed: int = 42,
) -> PerturbationResult:
    """Perturb inter-visit *timing* by jittering each visit date within bounds.

    This is the timing counterpart to :func:`truncate_to_recent_visits`: it
    shifts ``visit_date`` values to simulate irregular follow-up intervals while
    keeping the set of visits and patients fixed.

    Contract (enforced)
    -------------------
    * Each visit date is shifted by an integer number of days drawn from
      ``[-max_shift_days, +max_shift_days]`` using a per-patient deterministic
      stream seeded by ``(seed, "jitter_visit_timing", patient_id)``.
    * **Visit ordering is preserved.** After jittering, each patient's dates are
      re-sorted and clamped so they remain strictly non-decreasing in their
      original order; a later visit is never moved before an earlier one. No
      visit is dropped, added, or reassigned to another patient.
    * Shifts stay within the per-patient original date span: the first visit is
      never moved earlier than the patient's original first date and the last is
      never moved later than the patient's original last date. This keeps all
      jittered dates inside the originally observed window, so no visit can cross
      a downstream prediction cutoff that was derived from that window.
    * Labels are never touched (they live in a separate Series).
    * Source DataFrame is never mutated.

    Limitation
    ----------
    Without an explicit, caller-supplied prediction-cutoff date this operator
    conservatively confines jitter to each patient's own observed date span. If
    your experiment defines a global cutoff, pass pre-truncated observation data
    (observations already restricted to the observation window) so the invariant
    above guarantees no information crosses the cutoff. This module makes no
    clinical assumptions about realistic visit spacing.
    """
    if max_shift_days < 0:
        raise ValueError("max_shift_days must be >= 0.")
    _check_contract(observation_data)
    out = observation_data.copy(deep=True)

    for pid, grp in out.groupby("patient_id", sort=True):
        grp_sorted = grp.sort_values("visit_date")
        idxs = grp_sorted.index.tolist()
        orig_dates = pd.to_datetime(grp_sorted["visit_date"]).tolist()
        if max_shift_days == 0 or len(idxs) == 0:
            continue
        lo = orig_dates[0]
        hi = orig_dates[-1]
        rng = _patient_rng(seed, "jitter_visit_timing", pid)
        shifts = rng.integers(-max_shift_days, max_shift_days + 1, size=len(idxs))
        jittered = [
            pd.Timestamp(d) + timedelta(days=int(s))
            for d, s in zip(orig_dates, shifts)
        ]
        # Clamp to the patient's original span so nothing leaves the observed
        # window (and thus cannot cross a window-derived cutoff).
        jittered = [min(max(d, lo), hi) for d in jittered]
        # Enforce non-decreasing order to preserve visit sequence.
        for i in range(1, len(jittered)):
            if jittered[i] < jittered[i - 1]:
                jittered[i] = jittered[i - 1]
        for idx, new_date in zip(idxs, jittered):
            out.at[idx, "visit_date"] = new_date

    perturbed = out.sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    return PerturbationResult(
        observation_data=perturbed,
        condition={
            "kind": "jitter_visit_timing",
            "strength": float(max_shift_days),
            "params": {
                "seed": seed,
                "max_shift_days": int(max_shift_days),
                "patients_before": int(out["patient_id"].nunique()),
                "patients_after": int(perturbed["patient_id"].nunique()),
            },
        },
    )


def thin_followup(
    observation_data: pd.DataFrame,
    keep_last_n: int,
    *,
    seed: int = 42,
) -> PerturbationResult:
    """Deprecated alias for :func:`truncate_to_recent_visits`.

    Retained for backwards compatibility with existing exports and callers. The
    name "thin_followup" was misleading: the operation truncates to the most
    recent visits and does **not** alter follow-up timing. The ``seed``
    parameter is accepted but ignored (the operation is deterministic); prefer
    :func:`truncate_to_recent_visits`, which has no ``seed`` and a
    ``kind == "truncate_to_recent_visits"`` condition tag.
    """
    del seed  # operation is deterministic; parameter kept only for compatibility
    return truncate_to_recent_visits(observation_data, keep_last_n)
