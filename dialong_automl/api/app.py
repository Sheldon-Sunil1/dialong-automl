"""FastAPI application for DiaLong-AutoML (Phase 9 — API & Serving).

Exposes the EXISTING prediction pipeline over HTTP. It does not train models
and does not reimplement prediction: every scoring request is delegated to
:func:`dialong_automl.predict.predict_from_artifacts`, and request validation
reuses the same leakage/forbidden-column rules as the prediction CLI.

Endpoints
---------
* ``GET  /health``   — service health + whether artifacts are loaded.
* ``GET  /metadata`` — sanitized model metadata, representation, threshold,
                       and the synthetic-data disclaimer.
* ``POST /predict``  — score a CLI-compatible patient trajectory.

Artifact directory resolution (no hard-coded path):
    1. explicit ``artifact_dir`` passed to :func:`create_app`
    2. environment variable ``DIALONG_ARTIFACT_DIR``
    3. ``AppConfig.paths.artifacts_dir`` (default ``artifacts``)

Safety
------
* Artifacts are loaded ONCE at startup (lifespan). Missing artifacts do not
  crash the service: ``/health`` reports ``model_loaded=false`` and
  ``/predict`` returns HTTP 503.
* No filesystem paths, secrets, or stack traces are returned to clients.
  Detailed errors are logged server-side only.
* DISCLAIMER: synthetic demonstration model — not clinical evidence, not
  clinically validated, not for medical decisions.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dialong_automl.api.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
)
from dialong_automl.features.wide import _REPRESENTATION_LEAKAGE_COLUMNS
from dialong_automl.predict import (
    PredictionError,
    _load_metadata,
    predict_from_artifacts,
)

logger = logging.getLogger("dialong_automl.api")

_DISCLAIMER = (
    "Synthetic demonstration model. Prediction is not clinical evidence, is "
    "not clinically validated, and must not be used for medical decisions."
)

# Metadata keys that may contain filesystem paths or internal detail and must
# NOT be echoed to clients.
_SENSITIVE_METADATA_KEYS = frozenset({
    "artifact_paths",
    "dataset_path",
    "optuna_study_db",
})


def _resolve_artifact_dir(artifact_dir: str | Path | None) -> Path:
    """Resolve the artifact directory from arg, env, or config (in that order)."""
    if artifact_dir is not None:
        return Path(artifact_dir)
    env = os.environ.get("DIALONG_ARTIFACT_DIR")
    if env:
        return Path(env)
    # Fall back to the configured default without requiring a YAML file.
    try:
        from dialong_automl.config import load_config

        cfg = load_config(None)
        return Path(cfg.paths.artifacts_dir)
    except Exception:  # noqa: BLE001 — config is best-effort here
        return Path("artifacts")


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Drop keys that could leak filesystem paths / internal detail."""
    return {k: v for k, v in meta.items() if k not in _SENSITIVE_METADATA_KEYS}


def create_app(
    artifact_dir: str | Path | None = None,
    db_path: str | Path | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Parameters
    ----------
    artifact_dir:
        Directory containing training artifacts. When ``None`` it is resolved
        from ``DIALONG_ARTIFACT_DIR`` or the configured default.
    db_path:
        SQLite patient-database path. When ``None`` it is resolved from
        ``DIALONG_DB_PATH`` or the configured default. Tests inject a temp path.
    """
    resolved_dir = _resolve_artifact_dir(artifact_dir)

    # Pull OpenAPI presentation settings from APIConfig (best-effort).
    try:
        from dialong_automl.config import load_config

        api_cfg = load_config(None).api
        title, version = api_cfg.title, api_cfg.version
        docs_url, redoc_url = api_cfg.docs_url, api_cfg.redoc_url
        cors_origins = list(api_cfg.cors_allow_origins)
    except Exception:  # noqa: BLE001
        title, version, docs_url, redoc_url = (
            "DiaLong-AutoML API", "0.1.0", "/docs", "/redoc",
        )
        cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Optional env override: comma-separated list of allowed origins. Never
    # defaults to a wildcard; must be set explicitly per environment.
    env_origins = os.environ.get("DIALONG_CORS_ORIGINS")
    if env_origins:
        cors_origins = [o.strip() for o in env_origins.split(",") if o.strip()]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load metadata ONCE at startup. Never trains. Missing/corrupt
        # artifacts are tolerated: the service starts in a degraded state.
        app.state.artifact_dir = resolved_dir
        app.state.model_loaded = False
        app.state.metadata = None
        app.state.load_error = None

        # Initialize the patient database (idempotent; never drops data).
        try:
            from dialong_automl.db import PatientRepository, create_session_factory

            engine, session_factory = create_session_factory(db_path)
            app.state.db_engine = engine
            app.state.repo = PatientRepository(session_factory)
            logger.info("API startup: patient database ready.")
        except Exception as exc:  # noqa: BLE001 — degrade, do not crash
            app.state.db_engine = None
            app.state.repo = None
            logger.warning("API startup: patient database unavailable (%s).", exc)

        try:
            meta = _load_metadata(resolved_dir)
            model_name = meta.get("selected_model")
            representation = meta.get("representation")
            if not model_name or not representation:
                raise PredictionError(
                    "model_metadata.json is missing 'selected_model'/'representation'."
                )
            app.state.metadata = meta
            app.state.model_loaded = True
            logger.info(
                "API startup: artifacts loaded (model=%s, representation=%s).",
                model_name, representation,
            )
        except PredictionError as exc:
            # Log full detail server-side; keep a generic flag for clients.
            app.state.load_error = str(exc)
            logger.warning("API startup: model artifacts not available (%s).", exc)
        yield

    app = FastAPI(
        title=title,
        version=version,
        docs_url=docs_url,
        redoc_url=redoc_url,
        description=(
            "Serve diabetes-progression predictions from a trained DiaLong-AutoML "
            "model. Synthetic demonstration data only — not clinically validated."
        ),
        lifespan=lifespan,
    )

    # CORS: allow the configured frontend origins only (no wildcard default) so
    # the browser dashboard can call the API cross-origin in development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # --- Patient/visit persistence + DB-backed prediction routes ---
    def _repo_getter():
        return getattr(app.state, "repo", None)

    def _model_loaded() -> bool:
        return bool(getattr(app.state, "model_loaded", False))

    def _observation_limit() -> int | None:
        meta = getattr(app.state, "metadata", None)
        if not meta:
            return None
        val = meta.get("observation_visits")
        return int(val) if val is not None else None

    def _predict_with_checks(patient_id: str, visits: list[dict[str, Any]]) -> dict[str, Any]:
        """Shared prediction path: leakage check + fitted-artifact inference."""
        for i, visit in enumerate(visits):
            forbidden = sorted(set(visit) & _REPRESENTATION_LEAKAGE_COLUMNS)
            if forbidden:
                raise PredictionError(
                    f"Visit #{i} contains forbidden future/outcome field(s) {forbidden}."
                )
        return predict_from_artifacts(app.state.artifact_dir, patient_id, visits)

    from dialong_automl.api.patients import create_patient_router

    app.include_router(
        create_patient_router(
            _repo_getter,
            _predict_with_checks,
            model_loaded=_model_loaded,
            observation_limit=_observation_limit,
        )
    )

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        loaded = bool(getattr(app.state, "model_loaded", False))
        return HealthResponse(
            status="ok",
            model_loaded=loaded,
            detail="Model artifacts loaded." if loaded
            else "Model artifacts not loaded; prediction is unavailable.",
        )

    @app.get("/metadata", response_model=MetadataResponse, tags=["model"])
    def metadata() -> MetadataResponse:
        loaded = bool(getattr(app.state, "model_loaded", False))
        meta: dict[str, Any] | None = getattr(app.state, "metadata", None)
        if not loaded or meta is None:
            return MetadataResponse(model_loaded=False, disclaimer=_DISCLAIMER)
        safe = _sanitize_metadata(meta)
        known = {
            "model": safe.get("selected_model"),
            "representation": safe.get("representation"),
            "threshold": safe.get("threshold"),
            "observation_visits": safe.get("observation_visits"),
            "synthetic_data": safe.get("synthetic_data"),
        }
        # Everything else non-sensitive goes under `extra`.
        extra = {
            k: v for k, v in safe.items()
            if k not in {
                "selected_model", "representation", "threshold",
                "observation_visits", "synthetic_data", "disclaimer",
            }
        }
        return MetadataResponse(
            model_loaded=True,
            disclaimer=safe.get("disclaimer") or _DISCLAIMER,
            extra=extra,
            **known,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["model"])
    def predict(req: PredictRequest) -> PredictResponse:
        if not getattr(app.state, "model_loaded", False):
            # 503: service is up but cannot serve predictions.
            raise HTTPException(
                status_code=503,
                detail="Model artifacts are not available; prediction is unavailable.",
            )

        # Convert visits to plain dicts (preserving any extra clinical fields).
        visits = [v.model_dump() for v in req.visits]

        # Reuse the CLI's forbidden-column rule for consistent leakage rejection.
        for i, visit in enumerate(visits):
            forbidden = sorted(set(visit) & _REPRESENTATION_LEAKAGE_COLUMNS)
            if forbidden:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Visit #{i} contains forbidden future/outcome field(s) "
                        f"{forbidden}. Submit observation data only — never future "
                        "outcomes or the target."
                    ),
                )

        try:
            result = predict_from_artifacts(
                app.state.artifact_dir, req.patient_id, visits
            )
        except PredictionError as exc:
            # Distinguish bad input (client error) from artifact problems
            # (service error) WITHOUT leaking filesystem paths.
            msg = str(exc)
            logger.warning("Prediction failed: %s", msg)
            if _is_artifact_error(msg):
                raise HTTPException(
                    status_code=503,
                    detail="Model artifacts are missing or incompatible.",
                ) from None
            raise HTTPException(
                status_code=422,
                detail=_client_safe_input_error(msg),
            ) from None
        except Exception:  # noqa: BLE001 — never leak a stack trace
            logger.exception("Unexpected error during prediction.")
            raise HTTPException(
                status_code=500, detail="Internal error during prediction."
            ) from None

        return PredictResponse(**result)

    return app


# ---------------------------------------------------------------------------
# Error-message classification (keep client responses free of paths/traces)
# ---------------------------------------------------------------------------

_ARTIFACT_ERROR_MARKERS = (
    "not found",
    "metadata",
    "preprocessor",
    "vocabulary",
    "not fitted",
    "artifact",
    "unknown representation",
    "corrupt",
)


def _is_artifact_error(message: str) -> bool:
    low = message.lower()
    return any(marker in low for marker in _ARTIFACT_ERROR_MARKERS)


def _client_safe_input_error(message: str) -> str:
    """Return an input-error message guaranteed not to contain a filesystem path.

    Input-validation errors from the predict path (bad JSON shape, missing
    visit_date, unparseable dates, forbidden columns) do not reference paths,
    so they are safe to surface. As a defensive measure, if a path-like token
    appears, fall back to a generic message.
    """
    if ("\\" in message) or ("/" in message) or (":" in message and "visit" not in message.lower()):
        return "Invalid request body."
    return message


# Module-level app for `uvicorn dialong_automl.api.app:app` (Makefile serve target).
app = create_app()
