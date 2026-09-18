# =============================================================================
# DiaLong-AutoML — Makefile skeleton
# Phase 1: foundation targets only.
# Run `make help` to list available targets.
# =============================================================================

.DEFAULT_GOAL := help
PYTHON        := python
PIP           := pip
PYTEST        := pytest
PACKAGE       := dialong_automl
VENV          := .venv

# Detect OS for path separator
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV)/Scripts
else
    VENV_BIN := $(VENV)/bin
endif

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:  ## Show this help message
	@echo "DiaLong-AutoML — available Make targets"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | sort \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
.PHONY: venv
venv:  ## Create a virtual environment in .venv/
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created. Activate with:"
	@echo "  source $(VENV_BIN)/activate  (Linux/macOS)"
	@echo "  $(VENV_BIN)/activate          (Windows PowerShell)"

.PHONY: install
install:  ## Install runtime dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install-dev
install-dev:  ## Install runtime + dev dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
.PHONY: lint
lint:  ## Run ruff linter
	ruff check $(PACKAGE) tests/

.PHONY: format
format:  ## Auto-format with ruff
	ruff format $(PACKAGE) tests/

.PHONY: typecheck
typecheck:  ## Run mypy type checker
	mypy $(PACKAGE)

.PHONY: quality
quality: lint typecheck  ## Run all quality checks

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
.PHONY: test
test:  ## Run the full test suite
	$(PYTEST) -q tests/

.PHONY: test-verbose
test-verbose:  ## Run tests with full output
	$(PYTEST) -v tests/

.PHONY: test-cov
test-cov:  ## Run tests with coverage report
	$(PYTEST) --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html tests/

# ---------------------------------------------------------------------------
# Verification (Phase 1)
# ---------------------------------------------------------------------------
.PHONY: check-import
check-import:  ## Verify the package imports cleanly
	$(PYTHON) -c "import $(PACKAGE); print('$(PACKAGE)', $(PACKAGE).__version__, 'import OK')"

.PHONY: verify
verify: check-import test  ## Full Phase 1 verification: import + tests

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
.PHONY: docker-build
docker-build:  ## Build the Docker image
	docker build -t dialong-automl:dev .

.PHONY: docker-run
docker-run:  ## Run the default Docker command (import check)
	docker-compose up --abort-on-container-exit dialong

.PHONY: docker-test
docker-test:  ## Run pytest inside the container
	docker-compose run --rm dialong pytest -q tests/

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
.PHONY: dirs
dirs:  ## Create standard project directories
	$(PYTHON) -c "from dialong_automl.utils import ArtifactPaths; ArtifactPaths().ensure_all()"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
.PHONY: clean
clean:  ## Remove build artefacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage

.PHONY: clean-all
clean-all: clean  ## Remove build artefacts + virtual environment
	rm -rf $(VENV)

# ---------------------------------------------------------------------------
# Future phase targets (stubs — will be fleshed out in later phases)
# ---------------------------------------------------------------------------
# Phase 2+
# .PHONY: train
# train:  ## Train all models via AutoML pipeline [Phase 2+]
# 	$(PYTHON) -m $(PACKAGE).pipeline.train --config configs/demo_fast.yaml

# Phase 3+
# .PHONY: tune
# tune:  ## Run Optuna hyperparameter search [Phase 3+]
# 	$(PYTHON) -m $(PACKAGE).pipeline.tune --config configs/demo_fast.yaml

# Phase 5+
# .PHONY: serve
# serve:  ## Start the FastAPI prediction server [Phase 5+]
# 	uvicorn $(PACKAGE).api.app:app --host 0.0.0.0 --port 8000 --reload
