import { useState } from "react";

import { MetadataResponse } from "../api/types";

function fmt(value: unknown): string {
  if (value === null || value === undefined) return "Not available";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

export function MetadataPanel({ metadata }: { metadata: MetadataResponse | null }) {
  const [showExtra, setShowExtra] = useState(false);

  if (!metadata) {
    return (
      <section className="panel" aria-labelledby="metadata-h">
        <h2 id="metadata-h">Model metadata</h2>
        <p className="muted">Metadata is unavailable (backend not reachable).</p>
      </section>
    );
  }

  if (!metadata.model_loaded) {
    return (
      <section className="panel" aria-labelledby="metadata-h">
        <h2 id="metadata-h">Model metadata</h2>
        <p className="muted">
          No model artifacts are loaded on the backend. Train a model and point
          the API at its artifact directory to enable predictions.
        </p>
      </section>
    );
  }

  const extraEntries = Object.entries(metadata.extra ?? {});

  return (
    <section className="panel" aria-labelledby="metadata-h">
      <h2 id="metadata-h">Model metadata</h2>
      <dl className="kv">
        <div><dt>Model</dt><dd>{fmt(metadata.model)}</dd></div>
        <div><dt>Representation</dt><dd>{fmt(metadata.representation)}</dd></div>
        <div><dt>Decision threshold</dt><dd>{fmt(metadata.threshold)}</dd></div>
        <div><dt>Observation window (visits)</dt><dd>{fmt(metadata.observation_visits)}</dd></div>
        <div><dt>Synthetic data</dt><dd>{fmt(metadata.synthetic_data)}</dd></div>
      </dl>

      {extraEntries.length > 0 && (
        <div className="extra">
          <button
            type="button"
            className="btn btn--ghost"
            aria-expanded={showExtra}
            onClick={() => setShowExtra((s) => !s)}
          >
            {showExtra ? "Hide" : "Show"} additional metadata ({extraEntries.length})
          </button>
          {showExtra && (
            <pre className="extra__json" data-testid="metadata-extra">
              {JSON.stringify(metadata.extra, null, 2)}
            </pre>
          )}
        </div>
      )}
    </section>
  );
}
