# DiaLong-AutoML

**Explainable Longitudinal AutoML for Diabetes Progression Prediction**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/phase-2%2B3%20data%20%26%20cohort-blue.svg)](./)

> **DISCLAIMER** All synthetic data produced by this project is demonstration
> data only. It is NOT based on real patients, NOT clinical evidence, and NOT
> suitable for medical validation, diagnosis, or treatment decisions.

---

## Overview

DiaLong-AutoML is a research-grade, modular AutoML framework for predicting
and explaining diabetes progression from longitudinal clinical data. It
combines classical and deep-learning models, automated hyperparameter
optimisation (Optuna), and model-agnostic explainability (SHAP + temporal
occlusion) behind a clean FastAPI interface.

**Current status: Phase 2 + Phase 3 complete.** Synthetic data generation,
loading, validation, profiling, cohort construction, temporal leakage
protection, and patient-level splitting are implemented. Model training has
not started.

---

## Package structure

```
dialong_automl/
├── config/           # YAML configuration system (Pydantic v2)
├── utils/            # Logging, seed, path utilities
├── models/           # Model registry (Phase 4+)
├── data/             # ← Phase 2+3: generation, loading, cohort, splitting
│   ├── synthetic_generator.py
│   ├── loader.py
│   ├── validator.py
│   ├── profiler.py
│   ├── cohort.py
│   └── splitter.py
├── features/         # Feature engineering (Phase 4+)
├── explainability/   # SHAP + temporal occlusion (Phase 6+)
├── evaluation/       # Metrics and reporting (Phase 7+)
└── api/              # FastAPI server (Phase 5+)
```

---

## Cohort construction and temporal-leakage guarantees

### Canonical mode: FUTURE_OUTCOME

`CohortMode.FUTURE_OUTCOME` is the canonical mode for the longitudinal
research experiment. Every experiment that trains a model must use this mode.

For each patient:

```
OBSERVATION DATA:   visit_1  visit_2  ...  visit_N   (visit_date ≤ cutoff)
                                               ↑
                                           cutoff_date
FUTURE DATA:        visit_N+1  ...                    (visit_date > cutoff)
  └─ used ONLY to derive the target. Never returned as features.
```

**Target rule** (applied within `prediction_horizon_days` of `cutoff_date`):

```
target = 1  iff any future visit v where:
                v.visit_date > cutoff_date
            AND v.visit_date ≤ cutoff_date + prediction_horizon_days
            AND (v.diabetes_diagnosis == 1
                 OR v.hba1c ≥ diabetes_onset_hba1c_threshold)

target = 0  otherwise (no qualifying event found within the horizon)
excluded    when no future visit falls within the horizon window
```

The `prediction_horizon_days` (default 365) is applied in calendar days
**relative to each patient's individual cutoff date**, not to a global date.

### PRE_LABELED mode

`CohortMode.PRE_LABELED` reads the `target` column directly from the input
DataFrame. It does **not** apply a prediction horizon.

- Use only for unit-testing or datasets with pre-existing verified labels.
- Do **not** use it as a shortcut for `FUTURE_OUTCOME` on synthetic data.
  The pre-labeled `target` in the obs-only CSV is derived from all future
  visits without a horizon cutoff and will produce a different prevalence.

### Leakage protections (enforced in code, tested in tests/)

| Protection | Where enforced | Tests |
|---|---|---|
| Future rows excluded from `observation_data` | `_build_future_outcome` | `test_no_temporal_leakage.py` |
| `target`, `is_future_visit`, leakage columns stripped | `_strip_leakage_columns` | `test_no_temporal_leakage.py` |
| Hard `RuntimeError` if obs row has `visit_date > cutoff` | `_build_future_outcome` | `test_no_temporal_leakage.py` |
| Safe column whitelist | `get_safe_observation_columns` | `test_no_temporal_leakage.py` |
| Generator and cohort builder use identical target rule | `test_target_consistency.py` | `test_target_consistency.py` |
| `diabetes_diagnosis=0` in all observation rows | `synthetic_generator.py` | `test_target_consistency.py` |

---

## Planned models

| Model | Type | Status |
|---|---|---|
| Logistic Regression | Tabular / baseline | Planned (Phase 4) |
| Random Forest | Tabular / ensemble | Planned (Phase 4) |
| XGBoost | Tabular / gradient boosting | Planned (Phase 4) |
| LightGBM | Tabular / gradient boosting | Planned (Phase 4) |
| GRU | Sequential / deep learning | Planned (Phase 4) |
| TCN | Sequential / deep learning | **Planned — not yet designed** |

---

## Phase roadmap

| Phase | Scope | Status |
|---|---|---|
| **1 — Foundation** | Package scaffold, config, logging, seed utils, model registry | ✅ Done |
| **2 — Data** | Synthetic generator, CSV loader, validator, profiler | ✅ Done |
| **3 — Cohort** | Cohort builder, temporal leakage protection, patient splitter | ✅ Done |
| 4 — Models | LR / RF / XGBoost / LightGBM / GRU implementations | Planned |
| 5 — API & Serving | FastAPI prediction endpoint, health checks | Planned |
| 6 — Explainability | SHAP, temporal occlusion, explanation reports | Planned |
| 7 — AutoML (Optuna) | Hyperparameter search, cross-validation, model selection | Planned |
| 8 — Evaluation | Full metrics suite, plots, final benchmarks | Planned |

---

## Quick start

### Prerequisites

- Python 3.11 or later
- `pip`

### Installation

```bash
git clone <repo-url>
cd dialong-automl

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

### Verify the installation

```bash
python --version
python -c "import dialong_automl; print('DiaLong import OK')"
pytest -q
```

### Generate synthetic data (500 patients)

```bash
python scripts/generate_demo_data.py \
    --output data/synthetic/diabetes_progression_demo.csv \
    --patients 500 \
    --seed 42
```

### Build a cohort and split

```python
from dialong_automl.data import (
    load_longitudinal_csv,
    CohortBuilder, CohortConfig, CohortMode,
    PatientSplitter, SplitConfig,
)

# Load the full timeline (observation + future visits)
full_df = load_longitudinal_csv("data/synthetic/diabetes_progression_full.csv")

# Build cohort — canonical future-outcome mode
cfg = CohortConfig(
    mode=CohortMode.FUTURE_OUTCOME,
    min_observation_visits=3,
    max_observation_visits=4,
    prediction_horizon_days=365,
    diabetes_onset_hba1c_threshold=6.5,
    exclude_known_diabetes_at_baseline=True,
)
cohort = CohortBuilder(cfg).build(full_df)

print(cohort.audit.positive_rate)       # actual positive rate
print(cohort.observation_data.columns)  # model-safe features only

# Split by patient (never by visit)
split = PatientSplitter(SplitConfig(seed=42)).split(
    cohort.observation_data, cohort.patient_targets
)
split.assert_no_overlap()
print(split.summary())
```

---

## Configuration

All configuration is YAML-based and validated by Pydantic v2.
See [`configs/demo_fast.yaml`](configs/demo_fast.yaml) for all options,
including `data`, `cohort`, and `split` sections added in Phase 2+3.

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
make help          # all Make targets
```

---

## Project conventions

- **Type hints** on all public functions and methods.
- **Docstrings** on all public modules, classes, and functions.
- **`pathlib.Path`** for all filesystem operations.
- **Pydantic v2** for configuration validation.
- **Structured logging** via Python's `logging` module.
- **Seed 42** as the default global random seed.
- **`FUTURE_OUTCOME` cohort mode** is the canonical mode for all
  longitudinal experiments; `PRE_LABELED` is for testing only.
- Future observations are **never** returned as model features.

---

## License

[MIT](LICENSE) © DiaLong-AutoML Contributors
