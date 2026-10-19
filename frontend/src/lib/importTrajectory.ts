// Parse an imported patient-trajectory JSON into the editable form state.
// Accepts both the object form ({patient_id, visits:[...]}) and a bare visit
// array, matching the backend/CLI contract. Non-schema metadata keys
// (`_disclaimer`, `visit_number`) are stripped; genuine clinical fields are
// always preserved (never silently dropped).

import { Visit } from "../api/types";
import { NON_SCHEMA_IMPORT_KEYS } from "./fields";

export interface ParsedTrajectory {
  patientId: string;
  visits: Visit[];
}

export class ImportError extends Error {}

function stripNonSchemaKeys(raw: Record<string, unknown>): Visit {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(raw)) {
    if (NON_SCHEMA_IMPORT_KEYS.has(k)) continue;
    out[k] = v;
  }
  if (!("visit_date" in out)) {
    // Keep the (missing) field explicit so downstream validation flags it.
    out.visit_date = "";
  }
  return out as Visit;
}

export function parseTrajectoryJson(text: string): ParsedTrajectory {
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch (err) {
    throw new ImportError(
      `File is not valid JSON: ${(err as Error).message}`,
    );
  }

  let patientId = "UNKNOWN";
  let rawVisits: unknown;

  if (Array.isArray(payload)) {
    rawVisits = payload;
  } else if (payload && typeof payload === "object") {
    const obj = payload as Record<string, unknown>;
    if (typeof obj.patient_id === "string" && obj.patient_id.trim() !== "") {
      patientId = obj.patient_id;
    }
    if (!("visits" in obj)) {
      throw new ImportError(
        "JSON object must contain a 'visits' array (or be a bare array of visits).",
      );
    }
    rawVisits = obj.visits;
  } else {
    throw new ImportError(
      "JSON must be an object with a 'visits' array or a bare array of visit objects.",
    );
  }

  if (!Array.isArray(rawVisits) || rawVisits.length === 0) {
    throw new ImportError("The trajectory must contain at least one visit.");
  }

  const visits: Visit[] = rawVisits.map((v, i) => {
    if (!v || typeof v !== "object" || Array.isArray(v)) {
      throw new ImportError(`Visit #${i + 1} is not a JSON object.`);
    }
    return stripNonSchemaKeys(v as Record<string, unknown>);
  });

  return { patientId, visits };
}
