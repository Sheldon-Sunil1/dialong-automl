// Client-side validation performed BEFORE a prediction request is sent, so the
// user sees actionable errors early. The backend remains the source of truth
// and performs its own validation (including leakage rejection); this mirrors
// those rules but never weakens them.

import { Visit } from "../api/types";
import {
  ALL_CLINICAL_FEATURES,
  FORBIDDEN_FIELDS,
  NUMERIC_FEATURES,
  NUMERIC_RANGES,
} from "./fields";

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$/;

/** True when `value` parses to a real calendar date. */
export function isValidDate(value: unknown): boolean {
  if (typeof value !== "string" || value.trim() === "") return false;
  if (!ISO_DATE_RE.test(value.trim())) return false;
  const d = new Date(value.trim());
  return !Number.isNaN(d.getTime());
}

function hasValue(v: unknown): boolean {
  return v !== "" && v !== null && v !== undefined;
}

/**
 * Validate a list of visits. Returns all problems found (not just the first).
 *
 * Hard requirements (block prediction): at least one visit; a valid
 * `visit_date` per visit; no forbidden future/outcome fields; present numeric
 * values must be finite and within schema bounds; visits in chronological
 * order. Missing clinical features are allowed (the backend imputes them); a
 * soft completeness notice is provided separately via `visitCompleteness`.
 */
export function validateVisits(visits: Visit[]): ValidationResult {
  const errors: string[] = [];

  if (!Array.isArray(visits) || visits.length === 0) {
    return { valid: false, errors: ["Add at least one visit before predicting."] };
  }

  const dates: (number | null)[] = [];

  visits.forEach((visit, i) => {
    const label = `Visit ${i + 1}`;

    // Required: visit_date, must be a valid date.
    if (!hasValue(visit.visit_date)) {
      errors.push(`${label}: visit date is required.`);
      dates.push(null);
    } else if (!isValidDate(visit.visit_date)) {
      errors.push(`${label}: visit date "${visit.visit_date}" is not a valid date (use YYYY-MM-DD).`);
      dates.push(null);
    } else {
      dates.push(new Date(`${visit.visit_date}`.trim()).getTime());
    }

    // Forbidden future/outcome/leakage fields.
    const forbidden = Object.keys(visit).filter((k) => FORBIDDEN_FIELDS.has(k));
    if (forbidden.length > 0) {
      errors.push(`${label}: remove forbidden future/outcome field(s): ${forbidden.join(", ")}.`);
    }

    // Numeric fields that are present must be finite numbers within range.
    for (const key of NUMERIC_FEATURES) {
      if (!(key in visit)) continue;
      const v = visit[key];
      if (!hasValue(v)) continue; // omitted → allowed (backend imputes)
      const num = typeof v === "number" ? v : Number(v);
      if (!Number.isFinite(num)) {
        errors.push(`${label}: "${key}" must be a number (got "${v}").`);
        continue;
      }
      const range = NUMERIC_RANGES[key];
      if (range && (num < range.min || num > range.max)) {
        errors.push(
          `${label}: "${key}" = ${num} is outside the supported range ${range.min}–${range.max}.`,
        );
      }
    }
  });

  // Chronological ordering: each valid date must be >= the previous valid date.
  let prev: number | null = null;
  for (let i = 0; i < dates.length; i++) {
    const d = dates[i];
    if (d === null) continue;
    if (prev !== null && d < prev) {
      errors.push(
        `Visit ${i + 1}: visits must be in chronological order (its date is earlier than a previous visit).`,
      );
    }
    prev = d;
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Non-blocking completeness check: how many clinical features are present
 * across all visits. Used to show a soft "for best results, complete all
 * fields" notice without preventing a valid partial prediction.
 */
export function visitCompleteness(visits: Visit[]): {
  complete: boolean;
  missingByVisit: { index: number; missing: string[] }[];
} {
  const missingByVisit: { index: number; missing: string[] }[] = [];
  visits.forEach((visit, i) => {
    const missing = ALL_CLINICAL_FEATURES.filter((f) => !hasValue(visit[f]));
    if (missing.length > 0) missingByVisit.push({ index: i, missing });
  });
  return { complete: missingByVisit.length === 0, missingByVisit };
}

/**
 * Normalize a visit for sending: drop empty strings/null/undefined so partial
 * trajectories work (the backend imputes missing clinical fields), and coerce
 * numeric fields to numbers. Never fabricates missing values.
 */
export function cleanVisitForRequest(visit: Visit): Visit {
  const out: Visit = { visit_date: `${visit.visit_date}`.trim() };
  for (const [key, value] of Object.entries(visit)) {
    if (key === "visit_date") continue;
    if (value === "" || value === null || value === undefined) continue;
    if ((NUMERIC_FEATURES as readonly string[]).includes(key)) {
      const num = typeof value === "number" ? value : Number(value);
      out[key] = Number.isFinite(num) ? num : (value as string);
    } else {
      out[key] = value as string;
    }
  }
  return out;
}
