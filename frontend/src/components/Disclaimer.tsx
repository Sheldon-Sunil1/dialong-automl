export function Disclaimer({ text }: { text?: string }) {
  const message =
    text ??
    "Research/experimental system using synthetic demonstration data. This is NOT a clinical diagnostic tool. Predictions are model probabilities, not confirmed clinical outcomes, and must not be used for diagnosis or treatment decisions.";
  return (
    <div className="disclaimer" role="note" aria-label="Research-use disclaimer">
      <strong>Disclaimer:</strong> {message}
    </div>
  );
}
