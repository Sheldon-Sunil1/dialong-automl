"""Patient-level train/validation/test splitter for DiaLong-AutoML.

Split guarantees
----------------
1. No patient appears in more than one split.
2. Splits are by *patient*, never by individual visit row.
3. Stratification by patient-level target when ``stratify=True``.
4. Reproducible: same seed always produces same split.
5. Different seed produces different split (where possible).
6. Returns patient ID sets / DataFrames, never raw clinical data.
7. Raises clear errors when a valid stratified split cannot be formed.

Usage::

    from dialong_automl.data.splitter import PatientSplitter, SplitConfig

    cfg = SplitConfig(train_size=0.70, validation_size=0.15, test_size=0.15, seed=42)
    splitter = PatientSplitter(cfg)
    result = splitter.split(observation_df, patient_targets)

    result.train_ids      # list[str]
    result.validation_ids # list[str]
    result.test_ids       # list[str]
    result.summary()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

_SPLIT_TOLERANCE = 1e-6


@dataclass
class SplitConfig:
    """Configuration for :class:`PatientSplitter`.

    Parameters
    ----------
    train_size:
        Fraction of patients for the training set.
    validation_size:
        Fraction of patients for the validation set.
    test_size:
        Fraction of patients for the test set.
        The three must sum to 1.0 (within floating-point tolerance).
    seed:
        Random seed for reproducibility.
    stratify:
        When ``True``, ensure both classes appear in every split.
    chronological:
        When ``True``, sort patients by their first visit date and split
        in chronological order rather than randomly.  Useful for
        research experiments on temporal generalization.
        Requires ``observation_df`` with ``visit_date`` column.
    """

    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    stratify: bool = True
    chronological: bool = False

    def __post_init__(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > _SPLIT_TOLERANCE:
            raise ValueError(
                f"train_size + validation_size + test_size must equal 1.0, "
                f"got {total:.6f}."
            )
        for name, val in [
            ("train_size", self.train_size),
            ("validation_size", self.validation_size),
            ("test_size", self.test_size),
        ]:
            if not (0.0 < val < 1.0):
                raise ValueError(f"{name} must be strictly between 0 and 1, got {val}.")


@dataclass
class SplitResult:
    """Output of :meth:`PatientSplitter.split`.

    Attributes
    ----------
    train_ids:
        Patient IDs assigned to the training set.
    validation_ids:
        Patient IDs assigned to the validation set.
    test_ids:
        Patient IDs assigned to the test set.
    train_df:
        Observation rows for training patients.
    validation_df:
        Observation rows for validation patients.
    test_df:
        Observation rows for test patients.
    train_targets:
        Target series for training patients.
    validation_targets:
        Target series for validation patients.
    test_targets:
        Target series for test patients.
    split_stats:
        Dict with counts and class distributions for each split.
    """

    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]

    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame

    train_targets: pd.Series
    validation_targets: pd.Series
    test_targets: pd.Series

    split_stats: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable split summary."""
        lines = ["=== SplitResult ==="]
        for split_name in ("train", "validation", "test"):
            ids = getattr(self, f"{split_name}_ids")
            targets = getattr(self, f"{split_name}_targets")
            n_pos = int((targets == 1).sum())
            n_neg = int((targets == 0).sum())
            total = len(ids)
            pos_rate = n_pos / total if total > 0 else 0.0
            lines.append(
                f"  {split_name:12s}: {total:5d} patients  "
                f"(pos={n_pos}, neg={n_neg}, rate={pos_rate:.3f})"
            )
        return "\n".join(lines)

    def assert_no_overlap(self) -> None:
        """Raise AssertionError if any patient appears in more than one split."""
        train_set = set(self.train_ids)
        val_set = set(self.validation_ids)
        test_set = set(self.test_ids)

        tv = train_set & val_set
        tt = train_set & test_set
        vt = val_set & test_set
        if tv or tt or vt:
            raise AssertionError(
                f"Patient overlap detected! "
                f"train&val={len(tv)}, train&test={len(tt)}, val&test={len(vt)}"
            )


class PatientSplitter:
    """Split a cohort into patient-level train/validation/test sets.

    Parameters
    ----------
    config:
        Split configuration.  Uses defaults when omitted.
    """

    def __init__(self, config: SplitConfig | None = None) -> None:
        self.cfg = config or SplitConfig()

    def split(
        self,
        observation_df: pd.DataFrame,
        patient_targets: pd.Series,
    ) -> SplitResult:
        """Perform the patient-level split.

        Parameters
        ----------
        observation_df:
            Observation data (output of ``CohortResult.observation_data``).
            Must contain ``patient_id``.
        patient_targets:
            Series indexed by ``patient_id``, values are 0 or 1.
            Output of ``CohortResult.patient_targets``.

        Returns
        -------
        SplitResult

        Raises
        ------
        ValueError
            If there are not enough patients to satisfy the split sizes, or
            if stratification cannot be achieved.
        """
        if "patient_id" not in observation_df.columns:
            raise ValueError("observation_df must contain a 'patient_id' column.")

        # Canonical patient list: intersection of df patients and target patients
        df_patients = set(observation_df["patient_id"].unique())
        target_patients = set(patient_targets.index)
        all_patients = sorted(df_patients & target_patients)

        if len(all_patients) == 0:
            raise ValueError(
                "No patients are present in both observation_df and patient_targets. "
                "Check that CohortResult.observation_data and patient_targets were built "
                "from the same cohort."
            )

        if len(all_patients) < 3:
            raise ValueError(
                f"Need at least 3 patients to split into 3 groups, got {len(all_patients)}."
            )

        targets_aligned = patient_targets.loc[all_patients]

        if self.cfg.chronological:
            train_ids, val_ids, test_ids = self._chronological_split(
                observation_df, all_patients
            )
        else:
            train_ids, val_ids, test_ids = self._stratified_split(
                all_patients, targets_aligned
            )

        # Verify class presence in each split when stratify=True
        if self.cfg.stratify and not self.cfg.chronological:
            for name, ids in [("train", train_ids), ("validation", val_ids), ("test", test_ids)]:
                split_targets = targets_aligned.loc[ids]
                unique_classes = set(split_targets.dropna().astype(int).unique())
                if len(unique_classes) < 2:
                    raise ValueError(
                        f"Stratified split failed: '{name}' split contains only class(es) "
                        f"{unique_classes} with {len(ids)} patients. "
                        "Try increasing the number of patients or adjusting split ratios."
                    )

        # Slice DataFrames
        train_df = observation_df[observation_df["patient_id"].isin(train_ids)].copy()
        val_df = observation_df[observation_df["patient_id"].isin(val_ids)].copy()
        test_df = observation_df[observation_df["patient_id"].isin(test_ids)].copy()

        train_targets = targets_aligned.loc[train_ids]
        val_targets = targets_aligned.loc[val_ids]
        test_targets = targets_aligned.loc[test_ids]

        n = len(all_patients)
        split_stats = self._compute_stats(
            train_ids, val_ids, test_ids,
            train_targets, val_targets, test_targets,
            n,
        )

        result = SplitResult(
            train_ids=train_ids,
            validation_ids=val_ids,
            test_ids=test_ids,
            train_df=train_df,
            validation_df=val_df,
            test_df=test_df,
            train_targets=train_targets,
            validation_targets=val_targets,
            test_targets=test_targets,
            split_stats=split_stats,
        )

        logger.info(
            "Split complete: train=%d, val=%d, test=%d (total=%d)",
            len(train_ids), len(val_ids), len(test_ids), n,
        )
        return result

    # ------------------------------------------------------------------
    # Stratified split via sklearn StratifiedShuffleSplit
    # ------------------------------------------------------------------

    def _stratified_split(
        self,
        patients: list[str],
        targets: pd.Series,
    ) -> tuple[list[str], list[str], list[str]]:
        """Stratified patient split using StratifiedShuffleSplit.

        Two-step:
          1. Split off test from the full set.
          2. Split validation from the remaining train+val set.
        """
        patients_arr = np.array(patients)
        labels = targets.loc[patients].values

        # Step 1: carve out the test set
        sss_test = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.cfg.test_size,
            random_state=self.cfg.seed,
        )
        try:
            train_val_idx, test_idx = next(sss_test.split(patients_arr, labels))
        except ValueError as exc:
            raise ValueError(
                f"Cannot create a stratified test split from {len(patients)} patients: {exc}"
            ) from exc

        train_val_patients = patients_arr[train_val_idx]
        test_patients = patients_arr[test_idx].tolist()
        train_val_labels = labels[train_val_idx]

        # Step 2: carve out validation from the train+val portion
        val_frac_of_trainval = self.cfg.validation_size / (
            self.cfg.train_size + self.cfg.validation_size
        )
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_frac_of_trainval,
            random_state=self.cfg.seed,
        )
        try:
            train_idx, val_idx = next(sss_val.split(train_val_patients, train_val_labels))
        except ValueError as exc:
            raise ValueError(
                f"Cannot create a stratified validation split from "
                f"{len(train_val_patients)} train+val patients: {exc}"
            ) from exc

        train_patients = train_val_patients[train_idx].tolist()
        val_patients = train_val_patients[val_idx].tolist()

        return train_patients, val_patients, test_patients

    # ------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------

    def _chronological_split(
        self, df: pd.DataFrame, patients: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Split patients in chronological order of first visit date."""
        if "visit_date" not in df.columns:
            raise ValueError(
                "chronological=True requires 'visit_date' in observation_df."
            )
        first_visit = (
            df.groupby("patient_id")["visit_date"].min()
            .reindex(patients)
            .sort_values()
        )
        ordered = list(first_visit.index)
        n = len(ordered)
        n_train = max(1, round(n * self.cfg.train_size))
        n_val = max(1, round(n * self.cfg.validation_size))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_val = max(1, n_val - 1)
            n_test = n - n_train - n_val
        train = ordered[:n_train]
        val = ordered[n_train: n_train + n_val]
        test = ordered[n_train + n_val:]
        return train, val, test

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(
        train_ids: list[str],
        val_ids: list[str],
        test_ids: list[str],
        train_targets: pd.Series,
        val_targets: pd.Series,
        test_targets: pd.Series,
        total_patients: int,
    ) -> dict[str, Any]:
        """Compute per-split statistics."""

        def _split_stat(ids: list[str], tgts: pd.Series) -> dict[str, Any]:
            n = len(ids)
            n_pos = int((tgts == 1).sum())
            n_neg = int((tgts == 0).sum())
            return {
                "n_patients": n,
                "fraction": round(n / total_patients, 4) if total_patients > 0 else 0.0,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "positive_rate": round(n_pos / n, 4) if n > 0 else 0.0,
            }

        return {
            "total_patients": total_patients,
            "train": _split_stat(train_ids, train_targets),
            "validation": _split_stat(val_ids, val_targets),
            "test": _split_stat(test_ids, test_targets),
        }
