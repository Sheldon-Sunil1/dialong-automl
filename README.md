# DiaLong-AutoML

**Explainable Longitudinal AutoML for Diabetes Progression Prediction**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/phase-1%20foundation-orange.svg)](./)

---

## Overview

DiaLong-AutoML is a research-grade, modular AutoML framework designed to predict and explain diabetes progression from longitudinal clinical data.  It combines classical and deep-learning models, automated hyperparameter optimisation (Optuna), and model-agnostic explainability (SHAP + temporal occlusion) behind a clean FastAPI interface.

**Current status: Phase 1 — Foundation.**  The project is being built phase by phase.  Only the infrastructure described in this document is implemented.  No training, data processing, or API endpoints exist yet.

---

## Planned Architecture

```
dialong_automl/
├── config/           # YAML-based configuration system (Pydantic v2)
├── utils/            # Logging, seed, path utilities
├── models/           # Model registry + implementations (Phase 2+)
├── data/             # Data loading and validation (Phase 2+)
├── features/         # Feature engineering (Phase 2+)
├── explainability/   # SHAP + temporal occlusion (Phase 4+)
├── evaluation/       # Metrics and reporting (Phase 2+)
└── api/              # FastAPI server (Phase 5+)
```

---

## Planned Models

| Model | Type | Status |
|-------|------|--------|
| Logistic Regression | Tabular / baseline | Planned (Phase 2) |
| Random Forest | Tabular / ensemble | Planned (Phase 2) |
| XGBoost | Tabular / gradient boosting | Planned (Phase 2) |
| LightGBM | Tabular / gradient boosting | Planned (Phase 2) |
| GRU | Sequential / deep learning | Planned (Phase 4) |
| TCN | Sequential / deep learning | **Planned — not yet designed** |

---

## Phase Roadmap

| Phase | Scope |
|-------|-------|
| **1 — Foundation** ✅ | Package scaffold, config system, logging, seed utils, model registry skeleton |
| 2 — Data & Tabular Models | Data loading, cohort construction, LR / RF / XGBoost / LightGBM training |
| 3 — AutoML (Optuna) | Hyperparameter search, cross-validation, model selection |
| 4 — Deep Learning | GRU implementation, temporal features |
| 5 — API & Serving | FastAPI prediction endpoint, health checks |
| 6 — Explainability | SHAP, temporal occlusion, explanation reports |
| 7 — Evaluation & Reporting | Full metrics suite, plots, final benchmarks |

---

## Quick Start (Phase 1)

### Prerequisites

- Python 3.11 or later
- `pip`

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd dialong-automl

# Create and activate a virtual environment
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Install the package with dev extras
pip install -e ".[dev]"
```

### Verify the installation

```bash
python --version
python -c "import dialong_automl; print('DiaLong import OK')"
pytest -q
```

### Load configuration

```python
from dialong_automl.config import load_config

cfg = load_config("configs/demo_fast.yaml")
print(cfg.project_name)          # DiaLong-AutoML
print(cfg.reproducibility.seed)  # 42
```

### Set the global random seed

```python
from dialong_automl.utils import set_global_seed

set_global_seed(42)
```

### Browse the model registry

```python
from dialong_automl.models import REGISTRY

print(REGISTRY.list_all())        # all registered model names
print(REGISTRY.list_available())  # [] — none available until Phase 2

entry = REGISTRY.get("tcn")
print(entry.status)               # planned_not_implemented
```

---

## Configuration

All configuration is YAML-based and validated by Pydantic v2.  The default configuration lives in [`configs/demo_fast.yaml`](configs/demo_fast.yaml).

Environment variables can override paths and behaviour — copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

---

## Development

```bash
make install-dev   # install all dependencies
make test          # run the test suite
make lint          # ruff linter
make typecheck     # mypy
make verify        # import check + full test suite
make help          # list all Make targets
```

---

## Docker

```bash
# Build
docker build -t dialong-automl:dev .

# Run import check
docker-compose up dialong
```

---

## Project Conventions

- **Type hints** on all public functions and methods.
- **Docstrings** on all public modules, classes, and functions.
- **`pathlib.Path`** for all filesystem operations — no hard-coded OS paths.
- **Pydantic v2** for configuration validation.
- **Structured logging** via Python's `logging` module.
- **Seed 42** as the default global random seed.
- All Phase 1 utilities raise `NotImplementedError` rather than silently returning stub data.

---

## License

[MIT](LICENSE) © DiaLong-AutoML Contributors
