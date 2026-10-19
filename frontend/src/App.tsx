import { useEffect, useMemo, useRef, useState } from "react";

import { apiClient } from "./api/client";
import { ApiError, SelectedPredictResponse, Visit, VisitRecord } from "./api/types";
import { Disclaimer } from "./components/Disclaimer";
import { ErrorBanner } from "./components/ErrorBanner";
import { HealthBadge } from "./components/HealthBadge";
import { JsonImport } from "./components/JsonImport";
import { MetadataPanel } from "./components/MetadataPanel";
import { PatientIdStep } from "./components/PatientIdStep";
import { PredictionResult } from "./components/PredictionResult";
import { SavedVisits } from "./components/SavedVisits";
import { Spinner } from "./components/Spinner";
import { UnavailablePanel } from "./components/UnavailablePanel";
import { VisitForm } from "./components/VisitForm";
import { VisitSelector } from "./components/VisitSelector";
import { useBackendStatus } from "./hooks/useBackendStatus";
import { usePatient } from "./hooks/usePatient";
import { cleanVisitForRequest, validateVisits } from "./lib/validation";

export default function App() {
  const status = useBackendStatus();
  const patient = usePatient();

  const [patientId, setPatientId] = useState("");
  const [idConfirmed, setIdConfirmed] = useState(false);

  // Unsaved draft visits being added (not yet persisted).
  const [drafts, setDrafts] = useState<Visit[]>([]);

  // Prediction / selection / import state — all cleared on patient switch.
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [predicting, setPredicting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [apiError, setApiError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [importConflict, setImportConflict] = useState<string | null>(null);
  const [result, setResult] = useState<SelectedPredictResponse | null>(null);
  const pendingImport = useRef<{ importedId: string; importedVisits: Visit[] } | null>(null);

  const modelLoaded = status.health?.model_loaded === true;
  const observationLimit =
    typeof status.metadata?.observation_visits === "number"
      ? status.metadata.observation_visits
      : null;

  const savedVisits: VisitRecord[] = patient.state.visits;

  /** Clear every piece of patient-specific transient state. */
  const clearTransient = () => {
    setDrafts([]);
    setSelectedIds([]);
    setValidationErrors([]);
    setApiError(null);
    setSaveError(null);
    setImportConflict(null);
    setResult(null);
    pendingImport.current = null;
  };

  const confirmPatientId = async (id: string) => {
    const trimmed = id.trim();
    if (trimmed === "") return;
    clearTransient();
    setPatientId(trimmed);
    setIdConfirmed(true);
    await patient.openPatient(trimmed);
  };

  const changePatient = () => {
    // Explicit "change patient": discard transient state and return to step 1.
    clearTransient();
    patient.reset();
    setIdConfirmed(false);
  };

  // When the active patient changes (id), clear prediction/selection so an old
  // prediction can never be shown for a new patient.
  const activeId = patient.state.activeId;
  useEffect(() => {
    setSelectedIds([]);
    setResult(null);
    setApiError(null);
  }, [activeId]);

  // Default: select all saved visits when a patient loads (user can change it).
  useEffect(() => {
    setSelectedIds(savedVisits.map((v) => v.id));
    setResult(null);
  }, [savedVisits]);

  // ------------------------------------------------------------------
  // Visit persistence
  // ------------------------------------------------------------------

  const saveDrafts = async () => {
    if (!activeId || drafts.length === 0) return;
    const { valid, errors } = validateVisits(drafts);
    setValidationErrors(errors);
    if (!valid) return;

    setSaving(true);
    setSaveError(null);
    try {
      for (const d of drafts) {
        await apiClient.addVisit(activeId, cleanVisitForRequest(d) as Visit);
      }
      setDrafts([]);
      await patient.reload();
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Failed to save visits.");
    } finally {
      setSaving(false);
    }
  };

  const deleteSaved = async (visitId: number) => {
    if (!activeId) return;
    setSaveError(null);
    try {
      await apiClient.deleteVisit(activeId, visitId);
      await patient.reload();
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Failed to delete visit.");
    }
  };

  // ------------------------------------------------------------------
  // JSON import → preview → confirm → persist
  // ------------------------------------------------------------------

  const onImport = (importedId: string, importedVisits: Visit[]) => {
    const hasImportedId = Boolean(importedId) && importedId !== "UNKNOWN";
    if (hasImportedId && importedId.trim() !== (activeId ?? patientId).trim()) {
      setImportConflict(
        `The imported file's patient_id "${importedId}" differs from the active ` +
          `patient "${activeId}". Resolve the conflict: add these visits to the ` +
          `active patient, or cancel and switch patients.`,
      );
      pendingImport.current = { importedId, importedVisits };
      return;
    }
    // Imported visits become unsaved drafts (explicit Save persists them).
    setDrafts((prev) => [...prev, ...importedVisits]);
  };

  const resolveImportToActive = () => {
    const pending = pendingImport.current;
    if (pending) setDrafts((prev) => [...prev, ...pending.importedVisits]);
    pendingImport.current = null;
    setImportConflict(null);
  };

  const cancelImport = () => {
    pendingImport.current = null;
    setImportConflict(null);
  };

  // ------------------------------------------------------------------
  // Prediction (DB-backed, selected visits only)
  // ------------------------------------------------------------------

  const canPredict =
    idConfirmed &&
    modelLoaded &&
    !predicting &&
    activeId != null &&
    selectedIds.length > 0 &&
    (observationLimit == null || selectedIds.length <= observationLimit);

  const onPredict = async () => {
    if (!activeId) return;
    setApiError(null);
    setResult(null);

    if (selectedIds.length === 0) {
      setApiError("Select at least one saved visit to predict.");
      return;
    }
    if (observationLimit != null && selectedIds.length > observationLimit) {
      setApiError(
        `Select at most ${observationLimit} visits (the model's observation window).`,
      );
      return;
    }

    setPredicting(true);
    try {
      const res = await apiClient.predictSelected(activeId, selectedIds);
      setResult(res);
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 503) {
          setApiError("Model is unavailable on the backend (artifacts missing or incompatible).");
        } else {
          setApiError(err.message);
        }
      } else {
        setApiError("Unexpected error during prediction.");
      }
    } finally {
      setPredicting(false);
    }
  };

  const toggleSelect = (id: number) =>
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  const selectAll = () => setSelectedIds(savedVisits.map((v) => v.id));
  const deselectAll = () => setSelectedIds([]);

  // Clear the prediction whenever the selection changes (so a stale result is
  // never shown as if it came from the current selection).
  const selectionKey = useMemo(() => [...selectedIds].sort().join(","), [selectedIds]);
  const lastSelectionKey = useRef(selectionKey);
  useEffect(() => {
    if (lastSelectionKey.current !== selectionKey) {
      lastSelectionKey.current = selectionKey;
      setResult(null);
    }
  }, [selectionKey]);

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__titles">
          <h1>DiaLong-AutoML</h1>
          <p className="app__subtitle">
            Explainable Longitudinal AutoML — Diabetes Progression Research Dashboard
          </p>
        </div>
        <nav className="app__nav" aria-label="Sections">
          <a href="#status">Status</a>
          <a href="#predict">Patient</a>
          <a href="#experiments">Experiments</a>
        </nav>
      </header>

      <Disclaimer text={status.metadata?.disclaimer} />

      <main className="app__main">
        <section id="status" className="app__section">
          <HealthBadge
            loading={status.loading}
            health={status.health}
            error={status.error}
            baseUrl={apiClient.baseUrl}
            onRefresh={status.refresh}
          />
          {status.error && <ErrorBanner title="Backend unavailable" messages={status.error} />}
          <MetadataPanel metadata={status.metadata} />
        </section>

        <section id="predict" className="app__section">
          <h2 className="app__section-title">Patient trajectory &amp; prediction</h2>

          <PatientIdStep
            patientId={patientId}
            confirmed={idConfirmed}
            disabled={predicting || saving}
            onConfirm={confirmPatientId}
            onChangeRequest={changePatient}
          />

          {idConfirmed && patient.state.loading && <Spinner label="Loading patient…" />}

          {idConfirmed && patient.state.error && (
            <ErrorBanner title="Could not load patient" messages={patient.state.error} />
          )}

          {idConfirmed && !patient.state.loading && activeId && (
            <div className="trajectory" data-testid="trajectory-section">
              {patient.state.created && (
                <p className="note-text" data-testid="new-patient-note">
                  New patient <strong>{activeId}</strong> created. Add visits below.
                </p>
              )}

              {/* Saved, persisted visits (source of truth) */}
              <SavedVisits
                patientId={activeId}
                visits={savedVisits}
                disabled={saving || predicting}
                onDelete={deleteSaved}
              />

              {/* Import + draft editor to add new visits */}
              <JsonImport onImport={onImport} disabled={saving || predicting} />

              {importConflict && (
                <div className="error-banner" role="alert" data-testid="import-conflict">
                  <strong>Patient ID conflict</strong> — {importConflict}
                  <div className="actions">
                    <button type="button" className="btn" onClick={resolveImportToActive}>
                      Add to {activeId}
                    </button>
                    <button type="button" className="btn btn--ghost" onClick={cancelImport}>
                      Cancel import
                    </button>
                  </div>
                </div>
              )}

              <div className="panel">
                <h3>Add visits</h3>
                <p className="muted small">
                  New or imported visits are unsaved drafts until you save them to the patient
                  record.
                </p>
                <VisitForm visits={drafts} onChange={setDrafts} disabled={saving || predicting} />
                {validationErrors.length > 0 && (
                  <ErrorBanner title="Fix these before saving" messages={validationErrors} />
                )}
                {saveError && <ErrorBanner title="Save failed" messages={saveError} />}
                <div className="actions">
                  <button
                    type="button"
                    className="btn btn--primary"
                    onClick={saveDrafts}
                    disabled={saving || predicting || drafts.length === 0}
                    data-testid="save-visits"
                  >
                    {saving ? <Spinner label="Saving…" /> : `Save ${drafts.length} visit(s)`}
                  </button>
                  {drafts.length > 0 && (
                    <span className="badge badge--muted" data-testid="unsaved-badge">
                      unsaved
                    </span>
                  )}
                </div>
              </div>

              {/* Visit selection for prediction */}
              <VisitSelector
                patientId={activeId}
                visits={savedVisits}
                selectedIds={selectedIds}
                observationLimit={observationLimit}
                disabled={predicting}
                onToggle={toggleSelect}
                onSelectAll={selectAll}
                onDeselectAll={deselectAll}
              />

              {apiError && <ErrorBanner title="Prediction failed" messages={apiError} />}

              <div className="actions">
                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={onPredict}
                  disabled={!canPredict}
                  data-testid="predict-button"
                >
                  {predicting ? (
                    <Spinner label="Predicting…" />
                  ) : (
                    `Predict progression risk (${selectedIds.length} visit(s))`
                  )}
                </button>
                {!modelLoaded && !status.loading && (
                  <span className="muted small">
                    Prediction is disabled until the backend reports a loaded model.
                  </span>
                )}
              </div>

              {result && <PredictionResult result={result} />}
            </div>
          )}
        </section>

        <section id="experiments" className="app__section">
          <details className="unavailable-group">
            <summary className="app__section-title">
              Experiments &amp; explainability
              <span className="badge badge--muted">offline research code</span>
            </summary>
            <p className="muted small">
              These capabilities exist in the project's offline research code but are not exposed
              by the current API. Nothing here is fabricated.
            </p>
            <UnavailablePanel
              title="Experiment leaderboard"
              reason="No /leaderboard endpoint. Leaderboards are produced offline by the training pipeline (outputs/.../leaderboard.csv)."
              availableVia="python scripts/run_experiments.py"
            />
            <UnavailablePanel
              title="Training & job monitoring"
              reason="No /train or job-status endpoints. Training runs offline via the CLI."
              availableVia="python -m dialong_automl.train / python scripts/run_experiments.py"
            />
            <UnavailablePanel
              title="Explainability (SHAP / temporal occlusion)"
              reason="No /explain endpoint, so no explanations are shown rather than fabricated ones."
              availableVia="dialong_automl.explainability (offline API)"
            />
          </details>
        </section>
      </main>

      <footer className="app__footer">
        <Disclaimer />
      </footer>
    </div>
  );
}
