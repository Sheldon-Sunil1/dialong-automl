import { HealthResponse } from "../api/types";
import { Spinner } from "./Spinner";

interface Props {
  loading: boolean;
  health: HealthResponse | null;
  error: string | null;
  baseUrl: string;
  onRefresh: () => void;
}

export function HealthBadge({ loading, health, error, baseUrl, onRefresh }: Props) {
  let state: "loading" | "down" | "degraded" | "ok" = "loading";
  if (!loading) {
    if (error || !health) state = "down";
    else if (!health.model_loaded) state = "degraded";
    else state = "ok";
  }

  const labels: Record<typeof state, string> = {
    loading: "Checking…",
    down: "Backend unreachable",
    degraded: "Online — model not loaded",
    ok: "Online — model loaded",
  };

  return (
    <div className={`health-badge health-badge--${state}`} data-testid="health-badge">
      <span className="health-badge__dot" aria-hidden="true" />
      <span className="health-badge__label">
        {loading ? <Spinner label="Checking backend…" /> : labels[state]}
      </span>
      <span className="health-badge__url" title="API base URL">
        {baseUrl}
      </span>
      <button type="button" className="btn btn--ghost" onClick={onRefresh} disabled={loading}>
        Refresh
      </button>
      {state === "down" && error && (
        <span className="health-badge__detail">{error}</span>
      )}
      {state !== "down" && health?.detail && (
        <span className="health-badge__detail">{health.detail}</span>
      )}
    </div>
  );
}
