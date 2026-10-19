import { useState } from "react";

import { VisitRecord } from "../api/types";
import { NUMERIC_FEATURES } from "../lib/fields";

interface Props {
  patientId: string;
  visits: VisitRecord[];
  disabled?: boolean;
  onDelete: (visitId: number) => void;
}

function nonEmptyCount(v: VisitRecord): number {
  return NUMERIC_FEATURES.filter((k) => v[k] != null && v[k] !== "").length;
}

/** Read-only list of the patient's PERSISTED visits (source of truth), with delete. */
export function SavedVisits({ patientId, visits, disabled, onDelete }: Props) {
  const [confirmId, setConfirmId] = useState<number | null>(null);

  return (
    <section className="panel" aria-labelledby="saved-h" data-testid="saved-visits">
      <h3 id="saved-h">Saved visits for {patientId} ({visits.length})</h3>
      {visits.length === 0 ? (
        <p className="muted" data-testid="saved-empty">
          No saved visits yet. Add visits below and save them.
        </p>
      ) : (
        <ul className="saved-list">
          {visits.map((v) => (
            <li key={v.id} data-testid={`saved-visit-${v.id}`}>
              <span className="saved-date">{v.visit_date}</span>
              <span className="muted small">
                {" "}
                · {nonEmptyCount(v)}/{NUMERIC_FEATURES.length} numeric fields
                {v.hba1c != null ? ` · HbA1c ${v.hba1c}` : ""}
              </span>
              {confirmId === v.id ? (
                <span className="saved-confirm">
                  <button
                    type="button"
                    className="btn btn--danger btn--small"
                    onClick={() => {
                      onDelete(v.id);
                      setConfirmId(null);
                    }}
                    disabled={disabled}
                    data-testid={`confirm-delete-${v.id}`}
                  >
                    Confirm delete
                  </button>
                  <button
                    type="button"
                    className="btn btn--ghost btn--small"
                    onClick={() => setConfirmId(null)}
                  >
                    Cancel
                  </button>
                </span>
              ) : (
                <button
                  type="button"
                  className="btn btn--ghost btn--small"
                  onClick={() => setConfirmId(v.id)}
                  disabled={disabled}
                  data-testid={`delete-visit-${v.id}`}
                >
                  Delete
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
