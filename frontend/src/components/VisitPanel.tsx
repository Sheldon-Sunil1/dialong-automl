import { Visit } from "../api/types";
import {
  CATEGORICAL_FEATURES,
  CATEGORICAL_LABELS,
  CATEGORICAL_OPTIONS,
  NUMERIC_FEATURES,
  NUMERIC_LABELS,
} from "../lib/fields";

interface Props {
  index: number;
  visit: Visit;
  collapsed: boolean;
  disabled?: boolean;
  onToggle: () => void;
  onField: (key: string, value: string) => void;
  onRemove: () => void;
}

function summary(visit: Visit): string {
  const parts: string[] = [];
  if (visit.visit_date) parts.push(String(visit.visit_date));
  if (visit.hba1c !== undefined && visit.hba1c !== "" && visit.hba1c !== null) {
    parts.push(`HbA1c ${visit.hba1c}`);
  }
  return parts.length ? parts.join(" · ") : "no data yet";
}

/** A single, independently collapsible visit panel. Collapsing never mutates data. */
export function VisitPanel({
  index,
  visit,
  collapsed,
  disabled,
  onToggle,
  onField,
  onRemove,
}: Props) {
  const bodyId = `visit-body-${index}`;
  return (
    <div className="visit-card" data-testid={`visit-${index}`}>
      <div className="visit-card__head">
        <button
          type="button"
          className="visit-card__toggle"
          aria-expanded={!collapsed}
          aria-controls={bodyId}
          onClick={onToggle}
          data-testid={`visit-toggle-${index}`}
        >
          <span className="visit-card__chevron" aria-hidden="true">
            {collapsed ? "▸" : "▾"}
          </span>
          <span className="visit-card__title">Visit {index + 1}</span>
          {collapsed && <span className="visit-card__summary">{summary(visit)}</span>}
        </button>
        <button
          type="button"
          className="btn btn--danger btn--small"
          onClick={onRemove}
          disabled={disabled}
          aria-label={`Remove visit ${index + 1}`}
          data-testid={`visit-remove-${index}`}
        >
          Remove
        </button>
      </div>

      {!collapsed && (
        <div className="visit-card__body" id={bodyId}>
          <div className="field">
            <label htmlFor={`visit_date-${index}`}>Visit date *</label>
            <input
              id={`visit_date-${index}`}
              type="date"
              value={(visit.visit_date as string) ?? ""}
              onChange={(e) => onField("visit_date", e.target.value)}
              disabled={disabled}
              required
            />
          </div>

          <div className="field-grid">
            {NUMERIC_FEATURES.map((key) => (
              <div className="field" key={key}>
                <label htmlFor={`${key}-${index}`}>{NUMERIC_LABELS[key] ?? key}</label>
                <input
                  id={`${key}-${index}`}
                  type="number"
                  step="any"
                  inputMode="decimal"
                  value={(visit[key] as string | number | undefined) ?? ""}
                  onChange={(e) => onField(key, e.target.value)}
                  disabled={disabled}
                  placeholder="optional"
                />
              </div>
            ))}

            {CATEGORICAL_FEATURES.map((key) => (
              <div className="field" key={key}>
                <label htmlFor={`${key}-${index}`}>{CATEGORICAL_LABELS[key] ?? key}</label>
                <input
                  id={`${key}-${index}`}
                  list={`${key}-options`}
                  value={(visit[key] as string | undefined) ?? ""}
                  onChange={(e) => onField(key, e.target.value)}
                  disabled={disabled}
                  placeholder="optional"
                />
                <datalist id={`${key}-options`}>
                  {(CATEGORICAL_OPTIONS[key] ?? []).map((opt) => (
                    <option value={opt} key={opt} />
                  ))}
                </datalist>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
