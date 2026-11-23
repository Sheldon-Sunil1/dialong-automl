import { ReliabilityInfo } from "../api/types";

function pct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

const SET_TEXT: Record<ReliabilityInfo["prediction_set"], string> = {
  no_onset: "No progression",
  onset: "Progression",
  undecided: "Undecided (both outcomes plausible)",
  empty: "None (input unlike the calibration patients)",
};

const LEVEL_TEXT: Record<ReliabilityInfo["level"], string> = {
  High: "No warning signs were found for this prediction.",
  Moderate: "Treat this prediction with some caution.",
  Low: "Treat this prediction with strong caution.",
};

/** Phase 14E: how much to trust this prediction (synthetic demo model). */
export function ReliabilityPanel({ reliability }: { reliability: ReliabilityInfo }) {
  const r = reliability;
  const info = r.flags.filter((f) => f.severity === "info");
  const coverage = Math.round(r.conformal_coverage_target * 100);
  return (
    <div className="reliability" data-testid="reliability-panel">
      <h3>Reliability of this prediction</h3>
      <p className={`reliability__level reliability__level--${r.level.toLowerCase()}`}
         data-testid="reliability-level">
        Reliability: <strong>{r.level}</strong>. {LEVEL_TEXT[r.level]}
      </p>
      <dl className="kv">
        <div>
          <dt>Calibrated probability</dt>
          <dd data-testid="reliability-calibrated">{pct(r.calibrated_probability)}</dd>
        </div>
        {r.range_low !== null && r.range_high !== null && (
          <div>
            <dt>Model variability range</dt>
            <dd data-testid="reliability-range">
              {pct(r.range_low)} to {pct(r.range_high)}
            </dd>
          </div>
        )}
        <div>
          <dt>{coverage}% prediction set</dt>
          <dd data-testid="reliability-set">{SET_TEXT[r.prediction_set]}</dd>
        </div>
      </dl>
      {r.reasons.length > 0 && (
        <>
          <p className="small">Why:</p>
          <ul className="small" data-testid="reliability-reasons">
            {r.reasons.map((s) => <li key={s}>{s}</li>)}
          </ul>
        </>
      )}
      {info.length > 0 && (
        <ul className="muted small" data-testid="reliability-info">
          {info.map((f) => <li key={f.code}>{f.message}</li>)}
        </ul>
      )}
      <p className="muted small">
        The variability range is not a confidence interval for the true risk. {r.note}
      </p>
    </div>
  );
}
