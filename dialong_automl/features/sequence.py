"""Padded sequence representation for DiaLong-AutoML (Phase 4).

Converts ``CohortResult.observation_data`` (long format) into a padded
3-D NumPy array suitable for GRU / LSTM models.

Output shape::

    X      : (n_patients, max_seq_len, n_features)   float32
    mask   : (n_patients, max_seq_len)                bool / uint8
    lengths: (n_patients,)                            int32

Mask convention
---------------
mask[i, t] = 1  →  real observation (not padding)
mask[i, t] = 0  →  padding (no visit exists for this patient at position t)

Padding is appended at the END of each sequence so that real visits are
left-aligned (visit_1 at index 0, visit_2 at index 1, …).

Feature encoding
----------------
Numeric features
    Taken as-is (NaN preserved for later imputation by
    :class:`~dialong_automl.features.preprocessing.SequencePreprocessor`).

Categorical features — **one-hot per visit**
    Each categorical variable is expanded into a block of binary indicator
    columns, one per observed category plus two sentinels:

    * position 0 in the block: ``<MISSING>`` (NaN / empty value)
    * position 1 in the block: ``<UNK>``     (category not in training vocab)
    * position 2+:              sorted training categories

    For example, ``sex`` with training categories {F, M} expands to three
    slots: ``sex__<MISSING>``, ``sex__<UNK>``, ``sex__F``, ``sex__M``.
    At each visit exactly one slot is 1, the rest are 0.

    **Why one-hot and not integer codes?**
    A GRU treats its input as continuous real-valued signals.  Integer
    label codes (0, 1, 2, …) impose an artificial ordinal magnitude — the
    model would perceive "current smoker" (code 2) as twice "former smoker"
    (code 1) plus some continuous distance from "never" (code 4).  These
    relationships have no clinical meaning and would introduce noise into
    the learned representations.  One-hot encoding avoids this entirely:
    every category is equidistant from every other in the feature space.

Feature ordering (fixed and deterministic)
------------------------------------------
::

    axis-2 index:  0 … (n_num-1)               numeric features in order
                   n_num … (n_num+n_cat_cols-1)  one-hot blocks for each
                                                  categorical feature in order

    where n_cat_cols = sum(len(vocab[cat]) for cat in categorical_features)

The full ordered list is stored in
:attr:`SequenceRepresentationResult.feature_names`.

Vocabulary (fitted on training data only)
-----------------------------------------
When no ``categorical_vocabularies`` argument is provided, the builder
constructs the vocabulary from the supplied DataFrame.  For
**validation and test sets** the caller MUST pass the training vocabulary::

    train_result = SequenceRepresentationBuilder(max_seq_len=4).build(train_df)
    vocab = train_result.categorical_vocabularies

    val_result = SequenceRepresentationBuilder(
        max_seq_len=4,
        categorical_vocabularies=dict(vocab),
    ).build(val_df)

This ensures:
* No validation or test category leaks into the vocabulary.
* Unknown val/test categories are handled safely (mapped to ``<UNK>`` slot).

Leakage protection
------------------
* Raises :class:`~dialong_automl.features.wide.LeakageError` for any
  leakage column in the input.
* The ``target`` is stored separately in ``result.y``; it is never in ``X``.

Usage::

    from dialong_automl.features.sequence import SequenceRepresentationBuilder

    builder = SequenceRepresentationBuilder(max_seq_len=4)
    result  = builder.build(cohort.observation_data)

    result.X            # ndarray (n_patients, max_seq_len, n_features)
    result.mask         # ndarray (n_patients, max_seq_len)
    result.lengths      # ndarray (n_patients,)
    result.patient_ids  # list[str]
    result.feature_names  # list[str]  — order matches axis-2 of X
    result.categorical_vocabularies  # dict[str, list[str]] — reuse for val/test
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dialong_automl.features.wide import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    LeakageError,
    _REPRESENTATION_LEAKAGE_COLUMNS,
    _require_columns,
)

logger = logging.getLogger(__name__)

# Sentinel labels inside each one-hot vocabulary block
_UNK = "<UNK>"
_MISSING_CAT = "<MISSING>"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SequenceRepresentationResult:
    """Output of :meth:`SequenceRepresentationBuilder.build`.

    Attributes
    ----------
    X:
        3-D float32 array of shape ``(n_patients, max_seq_len, n_features)``.
        Categorical features are one-hot encoded; numeric features are raw.
        Padding positions contain 0.0 (distinguish via ``mask``).
    mask:
        Boolean array of shape ``(n_patients, max_seq_len)``.
        1 = real visit, 0 = padding.
    lengths:
        1-D int32 array of shape ``(n_patients,)``.
        Number of real (non-padded) visits per patient.
    patient_ids:
        Ordered list of patient IDs matching axis-0 of ``X``.
    y:
        Optional Series indexed by patient_id.  ``None`` when no target
        was provided.  Never included in ``X``.
    feature_names:
        Ordered list of feature names matching axis-2 of ``X``.
        Numeric features keep their original names.
        One-hot slots are named ``{feature}__{category}``, e.g.
        ``sex__F``, ``sex__M``, ``sex__<MISSING>``, ``sex__<UNK>``.
    categorical_vocabularies:
        Dict mapping each categorical feature name to its ordered vocabulary
        list (same order as the one-hot columns in ``X``).
        Index 0 = ``<MISSING>``, index 1 = ``<UNK>``, index 2+ = sorted
        training categories.
        **Reuse this for validation and test sets** to prevent vocabulary
        leakage from held-out data.
    metadata:
        Builder configuration and shape statistics.
    """

    X: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    patient_ids: list[str]
    y: pd.Series | None
    feature_names: list[str]
    categorical_vocabularies: dict[str, list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class SequenceRepresentationBuilder:
    """Build a padded sequence representation from long-format observation data.

    Categorical features are **one-hot encoded** per visit — not integer
    label-encoded — to avoid imposing false ordinal relationships on the GRU.

    Parameters
    ----------
    max_seq_len:
        Maximum sequence length (number of time steps).  Patients with
        more visits are truncated to their *most recent* ``max_seq_len``
        visits.  Patients with fewer visits are zero-padded at the end.
    numeric_features:
        Numeric columns to include.  Order is preserved in ``feature_names``.
    categorical_features:
        Categorical columns to include.  Each is one-hot encoded using the
        vocabulary built from training data.
    categorical_vocabularies:
        Optional pre-built vocabulary dict (from a previous ``build`` call
        on training data).  When provided the builder skips vocabulary
        construction and uses these mappings.  Unknown categories are mapped
        to the ``<UNK>`` one-hot slot.
        **Must be supplied for validation and test sets.**
    """

    def __init__(
        self,
        max_seq_len: int = 4,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        categorical_vocabularies: dict[str, list[str]] | None = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.numeric_features = numeric_features or NUMERIC_FEATURES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self._vocab: dict[str, list[str]] = categorical_vocabularies or {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(
        self,
        observation_df: pd.DataFrame,
        patient_targets: pd.Series | None = None,
    ) -> SequenceRepresentationResult:
        """Build the padded one-hot sequence representation.

        Parameters
        ----------
        observation_df:
            Long-format observation DataFrame.  One row per visit.
            Must contain ``patient_id`` and ``visit_date``.
            Must NOT contain leakage columns.
        patient_targets:
            Optional Series indexed by patient_id.  Stored in ``result.y``
            but never in ``X``.

        Returns
        -------
        SequenceRepresentationResult
        """
        self._check_leakage(observation_df)
        _require_columns(observation_df, ["patient_id", "visit_date"])

        df = observation_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["visit_date"]):
            df["visit_date"] = pd.to_datetime(df["visit_date"])

        # Build vocabulary from this dataset if not already provided
        if not self._vocab:
            self._vocab = self._build_vocabularies(df)

        # -------------------------------------------------------------------
        # Determine feature names and axis-2 layout
        # -------------------------------------------------------------------
        numeric_present = [f for f in self.numeric_features if f in df.columns]
        cat_present = [f for f in self.categorical_features if f in df.columns]

        # Each categorical expands to len(vocab[cat]) one-hot slots
        cat_slot_names: list[str] = []
        for cat in cat_present:
            vocab = self._vocab.get(cat, [_MISSING_CAT, _UNK])
            for label in vocab:
                cat_slot_names.append(f"{cat}__{label}")

        feature_names: list[str] = numeric_present + cat_slot_names
        n_features = len(feature_names)

        # Pre-compute: for each categorical, the start index of its block
        # and a dict from string value → block-relative offset
        cat_block_info: list[tuple[int, dict[str, int]]] = []
        block_start = len(numeric_present)
        for cat in cat_present:
            vocab = self._vocab.get(cat, [_MISSING_CAT, _UNK])
            val_to_offset = {v: i for i, v in enumerate(vocab)}
            cat_block_info.append((block_start, val_to_offset))
            block_start += len(vocab)

        # -------------------------------------------------------------------
        # Allocate arrays
        # -------------------------------------------------------------------
        patient_ids_ordered: list[str] = sorted(df["patient_id"].unique())
        n_patients = len(patient_ids_ordered)

        X = np.zeros((n_patients, self.max_seq_len, n_features), dtype=np.float32)
        mask = np.zeros((n_patients, self.max_seq_len), dtype=np.uint8)
        lengths = np.zeros(n_patients, dtype=np.int32)

        # -------------------------------------------------------------------
        # Fill arrays
        # -------------------------------------------------------------------
        for i, pid in enumerate(patient_ids_ordered):
            grp = (
                df[df["patient_id"] == pid]
                .sort_values("visit_date")
                .reset_index(drop=True)
            )

            # Truncate to most recent max_seq_len visits
            if len(grp) > self.max_seq_len:
                grp = grp.tail(self.max_seq_len).reset_index(drop=True)

            n_real = len(grp)
            lengths[i] = n_real

            for t in range(n_real):
                visit_row = grp.iloc[t]
                mask[i, t] = 1

                # Numeric features — raw float (NaN preserved)
                for j, feat in enumerate(numeric_present):
                    val = visit_row.get(feat, float("nan"))
                    X[i, t, j] = float(val) if pd.notna(val) else float("nan")

                # Categorical features — one-hot block per feature
                for (blk_start, val_to_offset), cat in zip(cat_block_info, cat_present):
                    raw = visit_row.get(cat, None)
                    if pd.isna(raw) or str(raw).strip().lower() in ("nan", "none", "<na>", ""):
                        raw_str = _MISSING_CAT
                    else:
                        raw_str = str(raw).strip()

                    offset = val_to_offset.get(raw_str, val_to_offset.get(_UNK, 1))
                    X[i, t, blk_start + offset] = 1.0

        # -------------------------------------------------------------------
        # Target (kept separate, never in X)
        # -------------------------------------------------------------------
        y: pd.Series | None = None
        if patient_targets is not None:
            y = patient_targets.loc[
                [p for p in patient_ids_ordered if p in patient_targets.index]
            ].copy()
            y.index.name = "patient_id"

        # -------------------------------------------------------------------
        # Metadata
        # -------------------------------------------------------------------
        n_cat_slots = len(cat_slot_names)
        metadata: dict[str, Any] = {
            "max_seq_len": self.max_seq_len,
            "n_patients": n_patients,
            "n_features": n_features,
            "n_numeric_features": len(numeric_present),
            "n_categorical_features": len(cat_present),
            "n_categorical_slots": n_cat_slots,
            "categorical_encoding": "one_hot_per_visit",
            "X_shape": list(X.shape),
            "mask_shape": list(mask.shape),
            "seq_length_min": int(lengths.min()) if n_patients > 0 else 0,
            "seq_length_max": int(lengths.max()) if n_patients > 0 else 0,
            "seq_length_median": float(np.median(lengths)) if n_patients > 0 else 0.0,
            "seq_length_mean": float(lengths.mean()) if n_patients > 0 else 0.0,
        }

        logger.info(
            "Sequence representation built: shape=%s, mask=%s, "
            "encoding=one_hot, n_cat_slots=%d, seq_len min/median/max=%d/%.1f/%d",
            X.shape, mask.shape, n_cat_slots,
            metadata["seq_length_min"],
            metadata["seq_length_median"],
            metadata["seq_length_max"],
        )

        return SequenceRepresentationResult(
            X=X,
            mask=mask,
            lengths=lengths,
            patient_ids=patient_ids_ordered,
            y=y,
            feature_names=feature_names,
            categorical_vocabularies=dict(self._vocab),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def _build_vocabularies(
        self, df: pd.DataFrame
    ) -> dict[str, list[str]]:
        """Build one-hot vocabularies from the provided DataFrame.

        The vocabulary list order determines the one-hot column ordering:

        * Index 0: ``<MISSING>`` — for NaN or empty values.
        * Index 1: ``<UNK>``     — for categories unseen at transform time.
        * Index 2+: sorted observed categories (alphabetical for reproducibility).

        This vocabulary must be built ONLY from training data and reused for
        validation and test sets.
        """
        vocab: dict[str, list[str]] = {}
        for feat in self.categorical_features:
            if feat not in df.columns:
                vocab[feat] = [_MISSING_CAT, _UNK]
                continue
            raw = df[feat].dropna()
            raw = raw[
                ~raw.astype(str).str.strip().str.lower().isin(["nan", "none", "<na>", ""])
            ]
            categories = sorted(raw.astype(str).str.strip().unique().tolist())
            vocab[feat] = [_MISSING_CAT, _UNK] + categories
        return vocab

    # ------------------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------------------

    def _check_leakage(self, df: pd.DataFrame) -> None:
        """Raise LeakageError if any leakage column is present."""
        found = [c for c in df.columns if c in _REPRESENTATION_LEAKAGE_COLUMNS]
        if found:
            raise LeakageError(
                f"SequenceRepresentationBuilder received leakage columns {found}. "
                "Pass CohortResult.observation_data (leakage columns are stripped "
                "there) rather than the raw DataFrame."
            )
