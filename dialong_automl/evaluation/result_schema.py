"""Versioned experiment-result schema for DiaLong-AutoML (Phase 11).

A small, typed, JSON-serialisable container for a single evaluation run so that
baseline and perturbed runs can be compared with a consistent structure and
stable metric names. This does not change how experiments are run; it is a
record format the experiment runner (and future research phases) can populate.

Privacy: this record stores NO patient-identifying information — only dataset
and split *identifiers* (opaque labels/hashes), configuration hashes, seeds,
condition metadata, and aggregate metrics.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


def config_hash(config: Any) -> str:
    """Deterministic short hash of a JSON-serialisable config (sha256[:16]).

    Accepts a dict or any object with ``model_dump``/``to_dict``. Keys are
    sorted so the hash is stable regardless of ordering.
    """
    if hasattr(config, "model_dump"):
        payload = config.model_dump(mode="json")
    elif hasattr(config, "to_dict"):
        payload = config.to_dict()
    elif isinstance(config, dict):
        payload = config
    else:
        payload = {"repr": repr(config)}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def new_run_id() -> str:
    """A stable, unique run identifier (UTC timestamp + short uuid)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{ts}_{uuid.uuid4().hex[:8]}"


@dataclass
class ExperimentCondition:
    """Describes the condition under which a run was evaluated.

    ``kind`` is a free-form but stable label: ``"baseline"`` for an unperturbed
    run, or a perturbation type (e.g. ``"drop_measurements"``). ``strength`` is
    the perturbation magnitude (e.g. a fraction) or ``None`` for baseline.
    """

    kind: str = "baseline"
    strength: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    """A single evaluation run's result record.

    Unavailable optional values are represented explicitly as ``None`` (never
    omitted silently). Metric names live in a flat ``metrics`` mapping so that
    baseline and perturbed runs share identical keys.
    """

    run_id: str
    schema_version: str = SCHEMA_VERSION
    config_hash: str | None = None
    seeds: dict[str, int | None] = field(default_factory=dict)
    dataset_id: str | None = None
    split_id: str | None = None
    condition: ExperimentCondition = field(default_factory=ExperimentCondition)
    metrics: dict[str, float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- validation -------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError if the record is structurally invalid."""
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("run_id must be a non-empty string.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}."
            )
        if not isinstance(self.condition, ExperimentCondition):
            raise ValueError("condition must be an ExperimentCondition.")
        for key, val in self.metrics.items():
            if not isinstance(key, str):
                raise ValueError(f"metric key {key!r} must be a string.")
            if val is not None and not isinstance(val, (int, float)):
                raise ValueError(
                    f"metric {key!r} must be a number or None, got {type(val).__name__}."
                )

    # --- (de)serialization ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "seeds": dict(self.seeds),
            "dataset_id": self.dataset_id,
            "split_id": self.split_id,
            "condition": self.condition.to_dict(),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
            "created_utc": self.created_utc,
        }

    def to_json(self) -> str:
        # sort_keys → deterministic serialization for stable diffs/hashes.
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunResult":
        cond_d = d.get("condition") or {}
        condition = ExperimentCondition(
            kind=cond_d.get("kind", "baseline"),
            strength=cond_d.get("strength"),
            params=cond_d.get("params", {}) or {},
        )
        result = cls(
            run_id=d["run_id"],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            config_hash=d.get("config_hash"),
            seeds=d.get("seeds", {}) or {},
            dataset_id=d.get("dataset_id"),
            split_id=d.get("split_id"),
            condition=condition,
            metrics=d.get("metrics", {}) or {},
            metadata=d.get("metadata", {}) or {},
            created_utc=d.get("created_utc") or datetime.now(timezone.utc).isoformat(),
        )
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
