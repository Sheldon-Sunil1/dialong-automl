import { useCallback, useRef, useState } from "react";

import { apiClient } from "../api/client";
import { ApiError, VisitRecord } from "../api/types";

export interface PatientState {
  /** The confirmed, active patient id (null when none is active). */
  activeId: string | null;
  loading: boolean;
  /** Persisted visits for the active patient (chronological). */
  visits: VisitRecord[];
  error: string | null;
  created: boolean;
}

const EMPTY: PatientState = {
  activeId: null,
  loading: false,
  visits: [],
  error: null,
  created: false,
};

/**
 * Manages the active patient and its persisted visits with STALE-RESPONSE
 * protection: every lookup/reload is tagged with a monotonically increasing
 * request id, and only the response for the latest request is applied. This
 * guarantees a delayed response for a previous patient can never overwrite the
 * state of the currently active patient.
 */
export function usePatient() {
  const [state, setState] = useState<PatientState>(EMPTY);
  // Monotonic request counter — the "current" request id.
  const reqSeq = useRef(0);
  // The id the user intends to be active (guards against late responses).
  const intendedId = useRef<string | null>(null);

  const reset = useCallback(() => {
    // Invalidate any in-flight request and clear state.
    reqSeq.current += 1;
    intendedId.current = null;
    setState(EMPTY);
  }, []);

  /** Confirm a patient id: find-or-create, then load its history. */
  const openPatient = useCallback(async (rawId: string) => {
    const id = rawId.trim();
    if (id === "") return;

    const myReq = ++reqSeq.current;
    intendedId.current = id;
    // Immediately clear prior patient data so nothing leaks during transition.
    setState({ ...EMPTY, activeId: null, loading: true });

    try {
      const res = await apiClient.findOrCreatePatient(id);
      // Ignore if a newer request superseded this one.
      if (myReq !== reqSeq.current || intendedId.current !== id) return;
      setState({
        activeId: res.patient_id,
        loading: false,
        visits: res.visits,
        error: null,
        created: res.created,
      });
    } catch (err) {
      if (myReq !== reqSeq.current || intendedId.current !== id) return;
      const msg = err instanceof ApiError ? err.message : "Failed to open patient.";
      // On failure, do NOT show stale data: keep no active patient.
      setState({ ...EMPTY, error: msg });
    }
  }, []);

  /** Reload the active patient's history from the backend (source of truth). */
  const reload = useCallback(async () => {
    const id = state.activeId;
    if (!id) return;
    const myReq = ++reqSeq.current;
    intendedId.current = id;
    setState((s) => ({ ...s, loading: true }));
    try {
      const res = await apiClient.getPatient(id);
      if (myReq !== reqSeq.current || intendedId.current !== id) return;
      setState((s) => ({ ...s, visits: res.visits, loading: false, error: null }));
    } catch (err) {
      if (myReq !== reqSeq.current || intendedId.current !== id) return;
      const msg = err instanceof ApiError ? err.message : "Failed to reload patient.";
      setState((s) => ({ ...s, loading: false, error: msg }));
    }
  }, [state.activeId]);

  return { state, openPatient, reload, reset, intendedId, reqSeq };
}
