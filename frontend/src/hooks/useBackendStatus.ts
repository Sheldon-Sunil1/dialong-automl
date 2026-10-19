import { useCallback, useEffect, useState } from "react";

import { apiClient } from "../api/client";
import { ApiError, HealthResponse, MetadataResponse } from "../api/types";

export interface BackendStatus {
  loading: boolean;
  health: HealthResponse | null;
  metadata: MetadataResponse | null;
  error: string | null;
  /** Re-fetch health + metadata. */
  refresh: () => void;
}

/**
 * Fetches /health and /metadata on mount and on manual refresh.
 *
 * Uses a plain `cancelled` flag (rather than an AbortController) to ignore a
 * stale response after unmount/refresh. This avoids passing an AbortSignal
 * across realms — which some fetch implementations reject — and behaves
 * identically in the browser.
 */
export function useBackendStatus(): BackendStatus {
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    setError(null);

    (async () => {
      try {
        const [h, m] = await Promise.all([
          apiClient.getHealth(),
          apiClient.getMetadata(),
        ]);
        if (cancelled) return;
        setHealth(h);
        setMetadata(m);
      } catch (err) {
        if (cancelled) return;
        const msg =
          err instanceof ApiError ? err.message : "Unexpected error contacting the backend.";
        setError(msg);
        setHealth(null);
        setMetadata(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [tick]);

  return { loading, health, metadata, error, refresh };
}
