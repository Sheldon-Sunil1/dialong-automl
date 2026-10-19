import { VisitRecord } from "../api/types";

interface Props {
  patientId: string;
  visits: VisitRecord[];
  selectedIds: number[];
  observationLimit: number | null;
  disabled?: boolean;
  onToggle: (id: number) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
}

/** Explicit selection of SAVED visits to use for a prediction. */
export function VisitSelector({
  patientId,
  visits,
  selectedIds,
  observationLimit,
  disabled,
  onToggle,
  onSelectAll,
  onDeselectAll,
}: Props) {
  const selectedSet = new Set(selectedIds);
  const selectedVisits = visits.filter((v) => selectedSet.has(v.id));
  const selectedDates = selectedVisits.map((v) => v.visit_date);
  const overLimit = observationLimit != null && selectedIds.length > observationLimit;

  if (visits.length === 0) {
    return (
      <section className="panel" aria-labelledby="select-h">
        <h3 id="select-h">Select visits for prediction</h3>
        <p className="muted" data-testid="selector-empty">
          No saved visits yet for <strong>{patientId}</strong>. Add and save visits first.
        </p>
      </section>
    );
  }

  return (
    <section className="panel" aria-labelledby="select-h" data-testid="visit-selector">
      <h3 id="select-h">Select visits for prediction</h3>
      <p className="muted small">
        Patient <strong data-testid="selector-patient">{patientId}</strong>. Choose which saved
        visits to use. Only selected visits are sent to the model.
      </p>

      <div className="selector-controls">
        <button type="button" className="btn btn--ghost" onClick={onSelectAll} disabled={disabled}>
          Select all
        </button>
        <button type="button" className="btn btn--ghost" onClick={onDeselectAll} disabled={disabled}>
          Deselect all
        </button>
        <span className="selector-count" data-testid="selector-count">
          {selectedIds.length} of {visits.length} selected
        </span>
      </div>

      <ul className="selector-list">
        {visits.map((v) => (
          <li key={v.id}>
            <label>
              <input
                type="checkbox"
                checked={selectedSet.has(v.id)}
                onChange={() => onToggle(v.id)}
                disabled={disabled}
                data-testid={`select-visit-${v.id}`}
              />
              <span className="selector-date">{v.visit_date}</span>
              {v.hba1c != null && <span className="muted small"> · HbA1c {v.hba1c}</span>}
            </label>
          </li>
        ))}
      </ul>

      <div className="selector-preview" data-testid="selected-dates">
        <strong>Selected dates:</strong>{" "}
        {selectedDates.length ? selectedDates.join(", ") : <span className="muted">none</span>}
      </div>

      {observationLimit != null && (
        <p className={`small ${overLimit ? "error-text" : "muted"}`} data-testid="limit-note">
          The model accepts at most {observationLimit} visits per prediction (observation window).
          {overLimit && " Deselect visits to fit the window."}
        </p>
      )}
    </section>
  );
}
