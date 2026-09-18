# =============================================================================
# DiaLong-AutoML — Dockerfile skeleton
# Phase 1: foundation only.  Training and API stages are added in later phases.
# =============================================================================
# Build:   docker build -t dialong-automl:dev .
# Run:     docker run --rm dialong-automl:dev python -c "import dialong_automl; print('OK')"
# =============================================================================

# ---------------------------------------------------------------------------
# Base image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

LABEL maintainer="DiaLong-AutoML Contributors"
LABEL description="Explainable Longitudinal AutoML for Diabetes Progression Prediction"
LABEL phase="1-foundation"

# System-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 dialong \
    && useradd --uid 1000 --gid dialong --shell /bin/bash --create-home dialong

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependency installation stage
# ---------------------------------------------------------------------------
FROM base AS deps

# Copy only dependency files first to leverage Docker layer cache
COPY pyproject.toml requirements.txt ./

# Install runtime dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application stage
# ---------------------------------------------------------------------------
FROM deps AS app

# Copy source code
COPY --chown=dialong:dialong dialong_automl/ ./dialong_automl/
COPY --chown=dialong:dialong configs/ ./configs/

# Install the package in editable mode (without dev extras)
RUN pip install --no-cache-dir -e . --no-deps

# Create runtime directories
RUN mkdir -p artifacts outputs logs data/raw data/processed \
    && chown -R dialong:dialong /app

USER dialong

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import dialong_automl; print('healthy')" || exit 1

# ---------------------------------------------------------------------------
# Default command
# ---------------------------------------------------------------------------
# Phase 1: verify the import.  Later phases override this with the API server.
CMD ["python", "-c", "import dialong_automl; print('DiaLong-AutoML', dialong_automl.__version__, 'ready.')"]

# ---------------------------------------------------------------------------
# TODO (future phases)
# ---------------------------------------------------------------------------
# Phase 5+ API entrypoint:
# CMD ["uvicorn", "dialong_automl.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
