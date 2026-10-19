import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse, delay } from "msw";
import { describe, expect, it } from "vitest";

import { usePatient } from "../src/hooks/usePatient";
import { server } from "./msw/server";

const BASE = "http://localhost:8000";

describe("usePatient — stale-response protection", () => {
  it("a late response for a previous patient does not overwrite the active one", async () => {
    // SYN_0001 responds slowly; SYN_0002 responds fast. The user opens 0001
    // then immediately 0002. The late 0001 response must be ignored.
    server.use(
      http.post(`${BASE}/patients`, async ({ request }) => {
        const body = (await request.json()) as { patient_id: string };
        if (body.patient_id === "SYN_0001") {
          await delay(80);
          return HttpResponse.json({
            patient_id: "SYN_0001",
            exists: false,
            created: true,
            visits: [{ id: 1, visit_date: "2018-01-01" }],
          });
        }
        return HttpResponse.json({
          patient_id: "SYN_0002",
          exists: false,
          created: true,
          visits: [{ id: 2, visit_date: "2020-01-01" }],
        });
      }),
    );

    const { result } = renderHook(() => usePatient());

    await act(async () => {
      // Fire 0001 (slow) then 0002 (fast) back-to-back.
      result.current.openPatient("SYN_0001");
      result.current.openPatient("SYN_0002");
      await new Promise((r) => setTimeout(r, 200));
    });

    await waitFor(() => expect(result.current.state.activeId).toBe("SYN_0002"));
    // The late SYN_0001 response must NOT have replaced SYN_0002.
    expect(result.current.state.activeId).toBe("SYN_0002");
    expect(result.current.state.visits.map((v) => v.visit_date)).toEqual(["2020-01-01"]);
  });

  it("reset clears the active patient and invalidates in-flight requests", async () => {
    server.use(
      http.post(`${BASE}/patients`, () =>
        HttpResponse.json({ patient_id: "SYN_0001", exists: false, created: true, visits: [] }),
      ),
    );
    const { result } = renderHook(() => usePatient());
    await act(async () => {
      await result.current.openPatient("SYN_0001");
    });
    expect(result.current.state.activeId).toBe("SYN_0001");
    act(() => result.current.reset());
    expect(result.current.state.activeId).toBeNull();
    expect(result.current.state.visits).toEqual([]);
  });
});
