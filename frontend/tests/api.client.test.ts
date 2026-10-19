import { describe, expect, it } from "vitest";

import { apiClient } from "../src/api/client";
import { ApiError } from "../src/api/types";
import { server } from "./msw/server";
import {
  healthLoaded,
  metadataLoaded,
  predict422Pydantic,
  predict422String,
  predict503,
  predictNetworkError,
  predictSuccess,
} from "./msw/handlers";

describe("apiClient", () => {
  it("fetches health", async () => {
    server.use(healthLoaded);
    const h = await apiClient.getHealth();
    expect(h.status).toBe("ok");
    expect(h.model_loaded).toBe(true);
  });

  it("fetches metadata", async () => {
    server.use(metadataLoaded);
    const m = await apiClient.getMetadata();
    expect(m.model).toBe("lightgbm");
    expect(m.extra.seed).toBe(42);
  });

  it("predicts successfully", async () => {
    server.use(predictSuccess);
    const r = await apiClient.predict({ patient_id: "P", visits: [{ visit_date: "2020-01-01" }] });
    expect(r.probability).toBeCloseTo(0.1234);
    expect(r.prediction).toBe(0);
  });

  it("normalizes a string 422 detail", async () => {
    server.use(predict422String);
    await expect(
      apiClient.predict({ visits: [{ visit_date: "2020-01-01" }] }),
    ).rejects.toMatchObject({ status: 422 });
    try {
      await apiClient.predict({ visits: [{ visit_date: "2020-01-01" }] });
    } catch (e) {
      expect((e as ApiError).message).toContain("forbidden future/outcome");
    }
  });

  it("normalizes a Pydantic array 422 detail", async () => {
    server.use(predict422Pydantic);
    try {
      await apiClient.predict({ visits: [{ visit_date: "2020-01-01" }] });
      throw new Error("should have thrown");
    } catch (e) {
      const err = e as ApiError;
      expect(err.status).toBe(422);
      expect(err.message).toContain("visits.0.visit_date");
      expect(err.message).toContain("Field required");
    }
  });

  it("maps 503 to an ApiError", async () => {
    server.use(predict503);
    await expect(
      apiClient.predict({ visits: [{ visit_date: "2020-01-01" }] }),
    ).rejects.toMatchObject({ status: 503 });
  });

  it("flags network failures", async () => {
    server.use(predictNetworkError);
    try {
      await apiClient.predict({ visits: [{ visit_date: "2020-01-01" }] });
      throw new Error("should have thrown");
    } catch (e) {
      const err = e as ApiError;
      expect(err.isNetwork).toBe(true);
      expect(err.status).toBe(0);
    }
  });
});
