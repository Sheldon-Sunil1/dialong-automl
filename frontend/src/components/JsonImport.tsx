import { useRef, useState } from "react";

import { Visit } from "../api/types";
import { ImportError, parseTrajectoryJson } from "../lib/importTrajectory";

interface Props {
  onImport: (patientId: string, visits: Visit[]) => void;
  disabled?: boolean;
}

export function JsonImport({ onImport, disabled }: Props) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const doImport = (raw: string) => {
    setError(null);
    setNote(null);
    try {
      const { patientId, visits } = parseTrajectoryJson(raw);
      onImport(patientId, visits);
      setNote(
        `Imported ${visits.length} visit(s) for "${patientId}". Review and correct below before predicting.`,
      );
    } catch (err) {
      if (err instanceof ImportError) setError(err.message);
      else setError(`Could not import file: ${(err as Error).message}`);
    }
  };

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const content = await file.text();
    setText(content);
    doImport(content);
    // Allow re-importing the same file.
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <section className="panel" aria-labelledby="import-h">
      <h2 id="import-h">Import trajectory JSON</h2>
      <p className="muted small">
        Accepts <code>{"{ patient_id?, visits: [...] }"}</code> or a bare array of
        visits. Metadata keys like <code>_disclaimer</code> and{" "}
        <code>visit_number</code> are stripped; clinical fields are preserved.
      </p>

      <div className="import-controls">
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          onChange={onFile}
          disabled={disabled}
          aria-label="Upload trajectory JSON file"
          data-testid="json-file-input"
        />
      </div>

      <textarea
        className="import-textarea"
        rows={6}
        value={text}
        placeholder='{"patient_id": "SYN_0001", "visits": [{"visit_date": "2020-01-01", "hba1c": 5.6}]}'
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
        aria-label="Trajectory JSON text"
        data-testid="json-textarea"
      />
      <div className="import-controls">
        <button
          type="button"
          className="btn"
          onClick={() => doImport(text)}
          disabled={disabled || text.trim() === ""}
        >
          Import from text
        </button>
      </div>

      {error && (
        <p className="error-text" role="alert" data-testid="import-error">
          {error}
        </p>
      )}
      {note && (
        <p className="note-text" data-testid="import-note">
          {note}
        </p>
      )}
    </section>
  );
}
