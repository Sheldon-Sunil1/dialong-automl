// Renders a section for a capability that the CURRENT backend build does not
// expose over HTTP. We deliberately do NOT fabricate data (no fake leaderboard
// metrics, no placeholder explanation charts). The panel states the limitation
// and how the capability is available elsewhere in the project (CLI/offline).

interface Props {
  title: string;
  reason: string;
  availableVia?: string;
}

export function UnavailablePanel({ title, reason, availableVia }: Props) {
  return (
    <section className="panel panel--unavailable" aria-labelledby={`${title}-h`}>
      <h2 id={`${title}-h`}>{title}</h2>
      <p className="muted">
        <span className="badge badge--muted">Not available via API</span> {reason}
      </p>
      {availableVia && (
        <p className="muted small">Available via: {availableVia}</p>
      )}
    </section>
  );
}
