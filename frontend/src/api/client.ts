// Centralized, typed API client for the DiaLong-AutoML backend.
//
// Only the endpoints the backend actually implements are exposed here:
//   GET  /health
//   GET  /metadata
//   POST /predict
//
// Endpoints such as /train, /leaderboard, /model-info and /explain are NOT
// implemented by the backend in this build and are intentionally absent. The
// client never fabricates responses.

import { API_BASE_URL } from "../config";
import {
  ApiError,
  HealthResponse,
  MetadataResponse,
  PatientHistoryResponse,
  PatientLookupResponse,
  PredictRequest,
  PredictResponse,
  PydanticErrorItem,
  SelectedPredictResponse,
  Visit,
  VisitRecord,
} from "./types";

function joinUrl(path: string): string {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

/** Turn a FastAPI error body (`detail` may be a string or an array) into one message. */
function formatDetail(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      const items = detail as PydanticErrorItem[];
      const parts = items.map((it) => {
        const loc = Array.isArray(it.loc)
          ? it.loc.filter((p) => p !== "body").join(".")
          : "";
        return loc ? `${loc}: ${it.msg}` : it.msg;
      });
      return parts.join("; ") || `Request failed (HTTP ${status}).`;
    }
  }
  return `Request failed (HTTP ${status}).`;
}

async function request<T>(
  path: string,
  init?: RequestInit,
  signal?: AbortSignal,
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(joinUrl(path), {
      headers: { "Content-Type": "application/json" },
      signal,
      ...init,
    });
  } catch (err) {
    // fetch rejects on network/connection failure (backend down, DNS, CORS block).
    const msg =
      err instanceof DOMException && err.name === "AbortError"
        ? "Request was cancelled."
        : "Cannot reach the backend. Is the API running and the base URL correct?";
    throw new ApiError(msg, 0, true);
  }

  const text = await res.text();
  let body: unknown = undefined;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = undefined;
    }
  }

  if (!res.ok) {
    throw new ApiError(formatDetail(body, res.status), res.status);
  }
  return body as T;
}

export const apiClient = {
  baseUrl: API_BASE_URL,

  getHealth(signal?: AbortSignal): Promise<HealthResponse> {
    return request<HealthResponse>("/health", { method: "GET" }, signal);
  },

  getMetadata(signal?: AbortSignal): Promise<MetadataResponse> {
    return request<MetadataResponse>("/metadata", { method: "GET" }, signal);
  },

  predict(req: PredictRequest, signal?: AbortSignal): Promise<PredictResponse> {
    return request<PredictResponse>(
      "/predict",
      { method: "POST", body: JSON.stringify(req) },
      signal,
    );
  },

  // --- Patient persistence (Phase 10) ---

  findOrCreatePatient(patientId: string, signal?: AbortSignal): Promise<PatientLookupResponse> {
    return request<PatientLookupResponse>(
      "/patients",
      { method: "POST", body: JSON.stringify({ patient_id: patientId }) },
      signal,
    );
  },

  getPatient(patientId: string, signal?: AbortSignal): Promise<PatientHistoryResponse> {
    return request<PatientHistoryResponse>(
      `/patients/${encodeURIComponent(patientId)}`,
      { method: "GET" },
      signal,
    );
  },

  addVisit(patientId: string, visit: Visit, signal?: AbortSignal): Promise<VisitRecord> {
    return request<VisitRecord>(
      `/patients/${encodeURIComponent(patientId)}/visits`,
      { method: "POST", body: JSON.stringify(visit) },
      signal,
    );
  },

  updateVisit(
    patientId: string,
    visitId: number,
    visit: Visit,
    signal?: AbortSignal,
  ): Promise<VisitRecord> {
    return request<VisitRecord>(
      `/patients/${encodeURIComponent(patientId)}/visits/${visitId}`,
      { method: "PUT", body: JSON.stringify(visit) },
      signal,
    );
  },

  deleteVisit(patientId: string, visitId: number, signal?: AbortSignal): Promise<void> {
    return request<void>(
      `/patients/${encodeURIComponent(patientId)}/visits/${visitId}`,
      { method: "DELETE" },
      signal,
    );
  },

  predictSelected(
    patientId: string,
    visitIds: number[],
    signal?: AbortSignal,
  ): Promise<SelectedPredictResponse> {
    return request<SelectedPredictResponse>(
      `/patients/${encodeURIComponent(patientId)}/predict`,
      { method: "POST", body: JSON.stringify({ visit_ids: visitIds }) },
      signal,
    );
  },
};

export type ApiClient = typeof apiClient;
