import { useEffect, useState } from "react";

import { Visit } from "../api/types";
import { VisitPanel } from "./VisitPanel";

interface Props {
  visits: Visit[];
  onChange: (visits: Visit[]) => void;
  disabled?: boolean;
}

export function emptyVisit(): Visit {
  return { visit_date: "" };
}

export function VisitForm({ visits, onChange, disabled }: Props) {
  // Collapsed state is kept here (parallel to visits) so collapsing/expanding
  // never mutates the trajectory data. It is resized to match visits.length.
  const [collapsed, setCollapsed] = useState<boolean[]>(() => visits.map(() => false));

  // Keep the collapsed array length in sync if visits is replaced wholesale
  // (e.g. after a JSON import). New entries default to expanded.
  useEffect(() => {
    setCollapsed((prev) => {
      if (prev.length === visits.length) return prev;
      const next = visits.map((_, i) => prev[i] ?? false);
      return next;
    });
  }, [visits.length]);

  const updateField = (index: number, key: string, value: string) => {
    const next = visits.map((v, i) => (i === index ? { ...v, [key]: value } : v));
    onChange(next);
  };

  const addVisit = () => {
    onChange([...visits, emptyVisit()]);
    setCollapsed((prev) => [...prev, false]);
  };

  const removeVisit = (index: number) => {
    onChange(visits.filter((_, i) => i !== index));
    setCollapsed((prev) => prev.filter((_, i) => i !== index));
  };

  const toggle = (index: number) =>
    setCollapsed((prev) => prev.map((c, i) => (i === index ? !c : c)));

  return (
    <div className="visit-form">
      <div className="visit-form__header">
        <h3 id="visits-h">Patient visits ({visits.length})</h3>
        <button type="button" className="btn" onClick={addVisit} disabled={disabled} data-testid="add-visit">
          + Add visit
        </button>
      </div>

      {visits.length === 0 && (
        <p className="muted" data-testid="visits-empty">
          No visits yet. Add a visit or import a trajectory JSON file.
        </p>
      )}

      {visits.map((visit, i) => (
        <VisitPanel
          key={i}
          index={i}
          visit={visit}
          collapsed={collapsed[i] ?? false}
          disabled={disabled}
          onToggle={() => toggle(i)}
          onField={(key, value) => updateField(i, key, value)}
          onRemove={() => removeVisit(i)}
        />
      ))}
    </div>
  );
}
