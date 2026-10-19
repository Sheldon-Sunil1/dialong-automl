import { describe, expect, it } from "vitest";

import { ImportError, parseTrajectoryJson } from "../src/lib/importTrajectory";

describe("parseTrajectoryJson", () => {
  it("parses the project's example object shape and strips metadata keys", () => {
    const raw = JSON.stringify({
      _disclaimer: "synthetic only",
      patient_id: "SYN_EXAMPLE_0001",
      visits: [
        { visit_date: "2019-02-11", visit_number: 1, hba1c: 5.6, sex: "F" },
        { visit_date: "2019-08-19", visit_number: 2, hba1c: 5.8, sex: "F" },
      ],
    });
    const { patientId, visits } = parseTrajectoryJson(raw);
    expect(patientId).toBe("SYN_EXAMPLE_0001");
    expect(visits).toHaveLength(2);
    // visit_number stripped, clinical fields preserved.
    expect("visit_number" in visits[0]).toBe(false);
    expect(visits[0].hba1c).toBe(5.6);
    expect(visits[0].sex).toBe("F");
    expect(visits[0].visit_date).toBe("2019-02-11");
  });

  it("accepts a bare array of visits", () => {
    const raw = JSON.stringify([{ visit_date: "2020-01-01", hba1c: 6.0 }]);
    const { patientId, visits } = parseTrajectoryJson(raw);
    expect(patientId).toBe("UNKNOWN");
    expect(visits).toHaveLength(1);
  });

  it("preserves clinically relevant fields (does not drop unknown clinical keys)", () => {
    const raw = JSON.stringify({ visits: [{ visit_date: "2020-01-01", hba1c: 6.0, bmi: 28 }] });
    const { visits } = parseTrajectoryJson(raw);
    expect(visits[0].bmi).toBe(28);
    expect(visits[0].hba1c).toBe(6.0);
  });

  it("rejects malformed JSON", () => {
    expect(() => parseTrajectoryJson("{ not valid")).toThrow(ImportError);
  });

  it("rejects an object without visits", () => {
    expect(() => parseTrajectoryJson(JSON.stringify({ patient_id: "P" }))).toThrow(
      /must contain a 'visits' array/i,
    );
  });

  it("rejects empty visits", () => {
    expect(() => parseTrajectoryJson(JSON.stringify({ visits: [] }))).toThrow(
      /at least one visit/i,
    );
  });

  it("rejects a non-object visit", () => {
    expect(() => parseTrajectoryJson(JSON.stringify({ visits: [42] }))).toThrow(
      /not a JSON object/i,
    );
  });

  it("marks a missing visit_date so downstream validation can flag it", () => {
    const { visits } = parseTrajectoryJson(JSON.stringify({ visits: [{ hba1c: 6.0 }] }));
    expect(visits[0].visit_date).toBe("");
  });
});
