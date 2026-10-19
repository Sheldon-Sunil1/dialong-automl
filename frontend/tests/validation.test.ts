import { describe, expect, it } from "vitest";

import { cleanVisitForRequest, isValidDate, validateVisits } from "../src/lib/validation";

describe("isValidDate", () => {
  it("accepts ISO dates", () => {
    expect(isValidDate("2020-01-01")).toBe(true);
  });
  it("rejects malformed dates", () => {
    expect(isValidDate("not-a-date")).toBe(false);
    expect(isValidDate("2020-13-40")).toBe(false);
    expect(isValidDate("")).toBe(false);
    expect(isValidDate("01/02/2020")).toBe(false);
  });
});

describe("validateVisits", () => {
  it("requires at least one visit", () => {
    const r = validateVisits([]);
    expect(r.valid).toBe(false);
    expect(r.errors[0]).toMatch(/at least one visit/i);
  });

  it("flags missing visit_date", () => {
    const r = validateVisits([{ visit_date: "" }]);
    expect(r.valid).toBe(false);
    expect(r.errors.some((e) => /visit date is required/i.test(e))).toBe(true);
  });

  it("flags malformed visit_date", () => {
    const r = validateVisits([{ visit_date: "2020-99-99" }]);
    expect(r.valid).toBe(false);
    expect(r.errors.some((e) => /not a valid date/i.test(e))).toBe(true);
  });

  it("flags forbidden leakage fields", () => {
    const r = validateVisits([{ visit_date: "2020-01-01", target: 1 } as never]);
    expect(r.valid).toBe(false);
    expect(r.errors.some((e) => /forbidden future\/outcome/i.test(e))).toBe(true);
  });

  it("flags non-numeric numeric fields", () => {
    const r = validateVisits([{ visit_date: "2020-01-01", hba1c: "abc" } as never]);
    expect(r.valid).toBe(false);
    expect(r.errors.some((e) => /must be a number/i.test(e))).toBe(true);
  });

  it("accepts a valid partial visit", () => {
    const r = validateVisits([{ visit_date: "2020-01-01", hba1c: 5.6 }]);
    expect(r.valid).toBe(true);
    expect(r.errors).toHaveLength(0);
  });
});

describe("cleanVisitForRequest", () => {
  it("drops empty fields and coerces numerics", () => {
    const out = cleanVisitForRequest({
      visit_date: "2020-01-01",
      hba1c: "5.6",
      bmi: "",
      sex: "F",
    } as never);
    expect(out).toEqual({ visit_date: "2020-01-01", hba1c: 5.6, sex: "F" });
    expect("bmi" in out).toBe(false);
  });
});
