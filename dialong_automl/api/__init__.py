"""API sub-package for DiaLong-AutoML — Phase 9 (API & Serving).

Exposes the existing prediction pipeline over FastAPI. The application never
trains models and reuses :func:`dialong_automl.predict.predict_from_artifacts`
for all scoring.

Usage::

    # Programmatic (e.g. tests):
    from dialong_automl.api import create_app
    app = create_app(artifact_dir="artifacts")

    # Server:
    uvicorn dialong_automl.api.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health    — service health + whether artifacts are loaded
    GET  /metadata  — sanitized model metadata + disclaimer
    POST /predict   — score a CLI-compatible patient trajectory

DISCLAIMER: Synthetic demonstration model — not clinical evidence, not
clinically validated, not for medical decisions.
"""

from dialong_automl.api.app import app, create_app

__all__ = ["app", "create_app"]
