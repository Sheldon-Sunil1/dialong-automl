import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PredictResponse, ReliabilityInfo } from "../src/api/types";
import { PredictionResult } from "../src/components/PredictionResult";

const base: PredictResponse = {
  patient_id: "SYN_1", probability: 0.42, prediction: 1, threshold: 0.3,
  model: "lightgbm", representation: "wide", n_visits: 2, disclaimer: "Synthetic.",
};

const rel: ReliabilityInfo = {
  level: "Low",
  reasons: ["No HbA1c result in the visits used."],
  flags: [
    { code: "no_hba1c", severity: "severe", message: "No HbA1c result in the visits used." },
    { code: "short_history", severity: "info", message: "2 of up to 4 visits were available." },
  ],
  calibrated_probability: 0.21, calibrated_threshold: 0.15,
  prediction_set: "undecided", conformal_coverage_target: 0.9,
  range_low: 0.12, range_high: 0.33, range_note: "Model variability range.",
  ood_percentile: 40, ood_category: "typical", method: {},
  note: "Not clinical evidence.",
};

describe("ReliabilityPanel", () => {
  it("shows level, calibrated probability, range, set and reasons", () => {
    render(<PredictionResult result={{ ...base, reliability: rel }} />);
    expect(screen.getByTestId("reliability-level")).toHaveTextContent("Reliability: Low");
    expect(screen.getByTestId("reliability-calibrated")).toHaveTextContent("21.0%");
    expect(screen.getByTestId("reliability-range")).toHaveTextContent("12.0% to 33.0%");
    expect(screen.getByTestId("reliability-set")).toHaveTextContent("Undecided");
    expect(screen.getByText(/90% prediction set/)).toBeInTheDocument();
    expect(screen.getByTestId("reliability-reasons")).toHaveTextContent("No HbA1c");
    // Info flags are shown separately, never as a reason.
    expect(screen.getByTestId("reliability-info")).toHaveTextContent("2 of up to 4 visits");
    expect(screen.getByTestId("reliability-reasons")).not.toHaveTextContent("visits were");
    // Required wording: not a confidence interval; raw probability still shown.
    expect(screen.getByText(/not a confidence interval/i)).toBeInTheDocument();
    expect(screen.getByTestId("result-probability")).toHaveTextContent("42.0%");
  });

  it("handles a High level without reasons or range", () => {
    render(<PredictionResult result={{
      ...base,
      reliability: { ...rel, level: "High", reasons: [], flags: [], range_low: null,
                     range_high: null, prediction_set: "no_onset" },
    }} />);
    expect(screen.getByTestId("reliability-level")).toHaveTextContent("High");
    expect(screen.queryByTestId("reliability-reasons")).toBeNull();
    expect(screen.queryByTestId("reliability-range")).toBeNull();
  });

  it("falls back gracefully for old artifacts", () => {
    render(<PredictionResult result={{ ...base, reliability: null }} />);
    expect(screen.queryByTestId("reliability-panel")).toBeNull();
    expect(screen.getByTestId("reliability-unavailable")).toBeInTheDocument();
    render(<PredictionResult result={base} />);
    expect(screen.getAllByTestId("reliability-unavailable")).toHaveLength(2);
  });
});
