# DiaLong-AutoML — Phase 10: Persistence, Patient Management & Visit-Selection Prediction

> **Research-only disclaimer.** DiaLong-AutoML operates on **synthetic
> demonstration data**. Predictions are experimental model outputs, are **not
> clinically validated**, and must **not** be used as a medical diagnosis or
> for treatment decisions.

This document covers the Phase 10 enhancement: a SQLite-backed patient/visit
store, patient-management API, independent patient histories, manual visit
selection for prediction, and the React workflow that drives it.

## Database

- **Engine:** SQLite via SQLAlchemy (2.x). A future migration to PostgreSQL is
  straightforward because all access is isolated in a repository layer and no
  SQLite-only SQL is used in that layer.
- **Default path:** `data/dialong_patients.db`.
- **Configuration (highest precedence first):**
  1. explicit path passed to `create_app(db_path=...)` / `create_session_factory(...)`
  2. `DIALONG_DB_PATH` environment variable
  3. `AppConfig.database.db_path` (YAML config; default `data/dialong_patients.db`)
- **Safety:** the schema is created idempotently (`create_all`, create-if-absent).
  It is **never** dropped or recreated automatically, so patient data persists
  across restarts. Foreign keys are enforced (`PRAGMA foreign_keys=ON`); writes
  are transactional. Only patient/visit data is stored — no secrets or model
  artifacts.
- **Git:** the DB file and its WAL/journal siblings are git-ignored
  (`*.db`, `*.db-wal`, `*.db-shm`, `*.db-journal`, `data/dialong_patients.db`).

### Schema

`patients`: `id` (PK), `patient_id` (unique, externally supplied),
`created_at`, `updated_at`.

`visits`: `id` (PK), `patient_fk` (FK → `patients.id`, `ON DELETE CASCADE`),
`visit_date`, the canonical clinical features (10 numeric + 6 categorical, same
names as `dialong_automl.features.wide`), `created_at`, `updated_at`. An index
on `(patient_fk, visit_date)` supports chronological retrieval. Clinical columns
are **nullable** — a genuinely missing measurement is stored as `NULL`
(consistent with the inference pipeline's missing-value contract). No clinical
features were invented; the trained model's feature schema is unchanged.

## API endpoints

Existing contracts are preserved: `GET /health`, `GET /metadata`,
`POST /predict` (legacy, stateless) behave exactly as before.

New patient/visit routes:

| Method & path | Purpose | Notes |
|---|---|---|
| `POST /patients` | find-or-create a patient | body `{ "patient_id": "SYN_0001" }`; id is trimmed, case preserved. Returns `{ patient_id, exists, created, visits[] }`. |
| `GET /patients/{patient_id}` | patient id + chronological visits | 404 if unknown. |
| `POST /patients/{patient_id}/visits` | add a visit | body = `VisitInput` (`visit_date` required; clinical fields optional; unknown keys rejected). 201. |
| `PUT /patients/{patient_id}/visits/{visit_id}` | update a visit | ownership-enforced (404 if the visit isn't this patient's). |
| `DELETE /patients/{patient_id}/visits/{visit_id}` | delete a visit | ownership-enforced. 204. |
| `POST /patients/{patient_id}/predict` | predict from **selected saved visits** | body `{ "visit_ids": [1,2] }`. |

### DB-backed prediction (`/patients/{id}/predict`)

- Resolves the selected visit ids **from the database** for the active patient.
- **Rejects** any visit id that belongs to another patient (404) — the frontend
  can never predict with another patient's data.
- Uses the **server-trusted** persisted clinical values (not arbitrary client
  values) for the selected visits.
- Sorts selected visits **chronologically** before inference.
- Enforces the model's **observation window** explicitly: selecting more than
  `observation_visits` (from model metadata) returns a `422` with a clear
  message — it does **not** silently truncate.
- Reuses the existing fitted-artifact pipeline (`predict_from_artifacts`),
  preserving the canonical fixed feature layout, threshold, and model selection.
- Response (`SelectedPredictResponse`) includes everything `/predict` returns
  plus `selected_visit_ids` and `selected_visit_dates` so a result is traceable
  to its exact inputs.

## Frontend workflow

1. **Patient ID first.** The initial screen shows only a Patient ID field and a
   "Start trajectory" button. Typing alone does not unlock the form.
2. **Confirm → lookup.** Confirming a non-empty ID calls `POST /patients`
   (find-or-create) and loads the patient's saved visits. New patients show an
   empty history.
3. **Saved visits** (source of truth) are shown read-only with delete
   (confirm-required). **Add visits** uses an unsaved draft editor; **Save**
   persists drafts via `POST .../visits`, then reloads from the backend.
4. **Visit selection.** Checkboxes per saved visit, Select all / Deselect all, a
   selected-count, and a selected-dates preview. Only selected visits are sent.
5. **Predict** calls `POST /patients/{id}/predict` with the selected ids.
   Results show patient id, class, probability (%), threshold, model,
   representation, visits used, and the selected visit dates.
6. **Change patient** returns to step 1, discarding transient state.

### Data isolation & stale responses

On patient switch, the frontend clears the previous prediction, probability,
errors, validation messages, import preview, selection, and unsaved drafts
before showing the new patient. The `usePatient` hook tags each lookup with a
monotonic request id and an intended-id guard, so a delayed response for a
previous patient can never overwrite the active patient's state. Predictions are
also cleared whenever the selected visit set changes, so an old prediction is
never shown as if it came from the current selection.

### JSON import

Import (file or pasted text) is validated against the project schema; metadata
keys `_disclaimer` and `visit_number` are stripped; clinical fields are
preserved. Imported visits become **unsaved drafts** for the active patient and
are persisted only on explicit Save — import never silently saves or overwrites.
A `patient_id` mismatch between the file and the active patient raises an
explicit conflict that the user must resolve (add to the active patient, or
cancel). Malformed JSON shows a readable error.

## Running

Backend (terminal 1):

```
cd C:\Users\sheld\Desktop\dialong-automl-clean
$env:DIALONG_ARTIFACT_DIR="artifacts/research_v3"   # a dir with trained artifacts
$env:DIALONG_DB_PATH="data/dialong_patients.db"     # optional; this is the default
.\.venv\Scripts\python -m uvicorn dialong_automl.api.app:app --host 127.0.0.1 --port 8000
```

Frontend (terminal 2):

```
cd C:\Users\sheld\Desktop\dialong-automl-clean\frontend
copy .env.example .env.local     # VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev                       # http://localhost:5173
```

The app starts cleanly with an empty database and existing model artifacts; the
DB file and directory are created on first use.

## Tests

```
# Backend
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m pytest tests/test_db.py tests/test_patients_api.py -q

# Frontend
cd frontend
npm run test
npm run build
```

Tests use isolated temporary SQLite databases and never touch the developer's
real database. The mandatory `SYN_0001 → SYN_0002 → SYN_0001` isolation sequence
is covered at both the backend (`tests/test_patients_api.py`) and frontend
(`frontend/tests/patientSwitch.test.tsx`) levels.

## Limitations

- Synthetic, research-only model; **not clinically validated**.
- No authentication or multi-user access control (local/research tool).
- SQLite single-file store; PostgreSQL is a future migration, not implemented.
- The sequence model has an observation window (`observation_visits`, e.g. 4);
  predictions use at most that many visits and reject larger selections.
- Partial trajectories are accepted (missing fields imputed by the fitted
  preprocessor) but complete data is preferred; the UI surfaces completeness.
