// Types mirror the backend Pydantic schemas in dialong_automl/api/schemas.py.
// Do not add fields the backend does not return.

/** GET /health */
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  detail: string;
}

/** GET /metadata (sanitized; no filesystem paths) */
export interface MetadataResponse {
  model_loaded: boolean;
  model: string | null;
  representation: string | null;
  threshold: number | null;
  observation_visits: number | null;
  synthetic_data: boolean | null;
  disclaimer: string;
  extra: Record<string, unknown>;
}

/** A single observation visit. `visit_date` is required; other clinical fields optional. */
export interface Visit {
  visit_date: string;
  [field: string]: string | number | null | undefined;
}

/** POST /predict request body */
export interface PredictRequest {
  patient_id?: string;
  visits: Visit[];
}

/** POST /predict response */
export interface PredictResponse {
  patient_id: string;
  probability: number;
  prediction: number;
  threshold: number;
  model: string;
  representation: string;
  n_visits: number;
  /** Phase 14 reliability layer; null/absent for older model artifacts. */
  reliability?: ReliabilityInfo | null;
  disclaimer: string;
}

export interface ReliabilityFlag {
  code: string;
  severity: "severe" | "moderate" | "info";
  message: string;
}

export interface ReliabilityInfo {
  level: "Low" | "Moderate" | "High";
  reasons: string[];
  flags: ReliabilityFlag[];
  calibrated_probability: number;
  calibrated_threshold: number;
  prediction_set: "no_onset" | "onset" | "undecided" | "empty";
  conformal_coverage_target: number;
  range_low: number | null;
  range_high: number | null;
  range_note: string;
  ood_percentile: number | null;
  ood_category: string | null;
  method: Record<string, unknown>;
  note: string;
}

/** A persisted visit (as returned by the patient API). */
export interface VisitRecord extends Visit {
  id: number;
  created_at?: string | null;
  updated_at?: string | null;
}

/** Result of POST /patients (find-or-create). */
export interface PatientLookupResponse {
  patient_id: string;
  exists: boolean;
  created: boolean;
  visits: VisitRecord[];
}

/** Result of GET /patients/{id}. */
export interface PatientHistoryResponse {
  patient_id: string;
  created: boolean;
  visits: VisitRecord[];
}

/** DB-backed prediction response (traces the selected visits). */
export interface SelectedPredictResponse extends PredictResponse {
  selected_visit_ids: number[];
  selected_visit_dates: string[];
}

/** A single FastAPI/Pydantic validation error entry. */
export interface PydanticErrorItem {
  loc: (string | number)[];
  msg: string;
  type: string;
}

/** Normalized API error raised by the client. */
export class ApiError extends Error {
  /** HTTP status, or 0 for network/connection failures. */
  readonly status: number;
  /** True when the backend could not be reached at all. */
  readonly isNetwork: boolean;

  constructor(message: string, status: number, isNetwork = false) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.isNetwork = isNetwork;
  }
}
