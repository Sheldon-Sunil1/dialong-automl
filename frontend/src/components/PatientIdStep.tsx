import { useState } from "react";

interface Props {
  patientId: string;
  confirmed: boolean;
  disabled?: boolean;
  onConfirm: (id: string) => void;
  onChangeRequest: () => void;
}

/**
 * Step 1 of the trajectory workflow: enter and confirm a Patient ID.
 *
 * The trajectory controls (import + visit editor) are revealed by the parent
 * only after an explicit confirmation of a non-empty ID — typing alone does
 * not unlock them. Once confirmed, the field becomes read-only with a "Change"
 * action that returns to editing without discarding the trajectory.
 *
 * Note: the backend has no patient persistence; this ID only labels the
 * trajectory currently being edited. It does not look up an existing record.
 */
export function PatientIdStep({
  patientId,
  confirmed,
  disabled,
  onConfirm,
  onChangeRequest,
}: Props) {
  const [draft, setDraft] = useState(patientId);
  const [touched, setTouched] = useState(false);

  const trimmed = draft.trim();
  const isEmpty = trimmed === "";

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (isEmpty) return;
    onConfirm(trimmed);
  };

  if (confirmed) {
    return (
      <div className="patient-id patient-id--confirmed" data-testid="patient-id-confirmed">
        <div className="field">
          <label htmlFor="patient_id">Patient ID</label>
          <div className="patient-id__row">
            <input id="patient_id" type="text" value={patientId} readOnly data-testid="patient-id-value" />
            <button
              type="button"
              className="btn btn--ghost"
              onClick={() => {
                setDraft(patientId);
                onChangeRequest();
              }}
              disabled={disabled}
              data-testid="patient-id-change"
            >
              Change
            </button>
          </div>
          <p className="muted small">
            This ID labels the trajectory you are editing. The backend does not store or
            look up patient records.
          </p>
        </div>
      </div>
    );
  }

  return (
    <form className="patient-id" onSubmit={submit} data-testid="patient-id-step">
      <div className="field">
        <label htmlFor="patient_id">Patient ID</label>
        <div className="patient-id__row">
          <input
            id="patient_id"
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => setTouched(true)}
            disabled={disabled}
            placeholder="e.g. SYN_0001"
            aria-invalid={touched && isEmpty}
            aria-describedby="patient_id-help"
            data-testid="patient-id-input"
          />
          <button
            type="submit"
            className="btn btn--primary"
            disabled={disabled || isEmpty}
            data-testid="patient-id-confirm"
          >
            Start trajectory
          </button>
        </div>
        <p id="patient_id-help" className="muted small">
          Enter a Patient ID and confirm to begin adding visits or importing a trajectory.
        </p>
        {touched && isEmpty && (
          <p className="error-text" role="alert" data-testid="patient-id-error">
            Patient ID cannot be empty.
          </p>
        )}
      </div>
    </form>
  );
}
