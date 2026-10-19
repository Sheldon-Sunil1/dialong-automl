import { PredictResponse, SelectedPredictResponse } from "../api/types";

function pct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function selectedDates(r: PredictResponse): string[] | null {
  const sel = (r as SelectedPredictResponse).selected_visit_dates;
  return Array.isArray(sel) ? sel : null;
}

export function PredictionResult({ result }: { result: PredictResponse }) {
  const positive = result.prediction === 1;
  const dates = selectedDates(result);
  return (
    <section className="panel result" aria-labelledby="result-h" data-testid="prediction-result">
      <h2 id="result-h">Prediction</h2>

      <div className="result__headline">
        <div className={`result__risk ${positive ? "result__risk--pos" : "result__risk--neg"}`}>
          <span className="result__prob" data-testid="result-probability">
            {pct(result.probability)}
          </span>
          <span className="result__prob-label">predicted probability of progression</span>
        </div>
        <div className="result__class" data-testid="result-class">
          Predicted class:{" "}
          <strong>{positive ? "Progression (1)" : "No progression (0)"}</strong>
        </div>
      </div>

      {/* Simple, dependency-free probability bar (no charting lib needed). */}
      <div
        className="result__bar"
        role="img"
        aria-label={`Probability ${pct(result.probability)}`}
      >
        <div
          className={`result__bar-fill ${positive ? "is-pos" : "is-neg"}`}
          style={{ width: `${Math.max(0, Math.min(100, result.probability * 100))}%` }}
        />
      </div>

      <dl className="kv">
        <div>
          <dt>Patient ID</dt>
          <dd>{result.patient_id}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{result.model || "Not provided"}</dd>
        </div>
        <div>
          <dt>Representation</dt>
          <dd>{result.representation || "Not provided"}</dd>
        </div>
        <div>
          <dt>Decision threshold</dt>
          <dd>
            {typeof result.threshold === "number" ? result.threshold : "Not provided"}
          </dd>
        </div>
        <div>
          <dt>Visits used</dt>
          <dd>{result.n_visits}</dd>
        </div>
        {dates && (
          <div>
            <dt>Selected visit dates</dt>
            <dd data-testid="result-selected-dates">{dates.join(", ") || "none"}</dd>
          </div>
        )}
      </dl>

      <p className="muted small">
        This is a model probability, not a confirmed clinical outcome or
        diagnosis.
      </p>
      {result.disclaimer && <p className="muted small">{result.disclaimer}</p>}
    </section>
  );
}
