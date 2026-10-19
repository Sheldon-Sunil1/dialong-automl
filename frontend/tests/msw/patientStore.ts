import { http, HttpResponse } from "msw";

const BASE = "http://localhost:8000";

interface StoredVisit {
  id: number;
  visit_date: string;
  [k: string]: unknown;
}

/**
 * A stateful in-memory stand-in for the backend patient store, used ONLY in
 * tests. It mimics the real API's isolation/ownership semantics so the
 * frontend's patient-switching behavior can be exercised deterministically.
 */
export function makePatientStore(options?: { observationLimit?: number }) {
  const patients = new Map<string, StoredVisit[]>();
  let nextId = 1;
  const limit = options?.observationLimit ?? 4;

  const norm = (id: string) => id.trim();

  const handlers = [
    http.get(`${BASE}/health`, () =>
      HttpResponse.json({ status: "ok", model_loaded: true, detail: "Model artifacts loaded." }),
    ),
    http.get(`${BASE}/metadata`, () =>
      HttpResponse.json({
        model_loaded: true,
        model: "gru",
        representation: "sequence",
        threshold: 0.5,
        observation_visits: limit,
        synthetic_data: true,
        disclaimer: "Synthetic demonstration model. Not clinical evidence.",
        extra: {},
      }),
    ),

    http.post(`${BASE}/patients`, async ({ request }) => {
      const body = (await request.json()) as { patient_id?: string };
      const id = (body.patient_id ?? "").trim();
      if (!id) return HttpResponse.json({ detail: "patient_id is required." }, { status: 422 });
      const existed = patients.has(id);
      if (!existed) patients.set(id, []);
      return HttpResponse.json({
        patient_id: id,
        exists: existed,
        created: !existed,
        visits: patients.get(id),
      });
    }),

    http.get(`${BASE}/patients/:pid`, ({ params }) => {
      const id = norm(String(params.pid));
      if (!patients.has(id))
        return HttpResponse.json({ detail: "not found" }, { status: 404 });
      return HttpResponse.json({ patient_id: id, created: false, visits: patients.get(id) });
    }),

    http.post(`${BASE}/patients/:pid/visits`, async ({ params, request }) => {
      const id = norm(String(params.pid));
      if (!patients.has(id))
        return HttpResponse.json({ detail: "not found" }, { status: 404 });
      const body = (await request.json()) as Record<string, unknown>;
      const visit: StoredVisit = { id: nextId++, visit_date: String(body.visit_date), ...body };
      patients.get(id)!.push(visit);
      patients.get(id)!.sort((a, b) => a.visit_date.localeCompare(b.visit_date));
      return HttpResponse.json(visit, { status: 201 });
    }),

    http.delete(`${BASE}/patients/:pid/visits/:vid`, ({ params }) => {
      const id = norm(String(params.pid));
      const vid = Number(params.vid);
      const list = patients.get(id);
      if (!list) return HttpResponse.json({ detail: "not found" }, { status: 404 });
      const idx = list.findIndex((v) => v.id === vid);
      if (idx < 0) return HttpResponse.json({ detail: "not found" }, { status: 404 });
      list.splice(idx, 1);
      return new HttpResponse(null, { status: 204 });
    }),

    http.post(`${BASE}/patients/:pid/predict`, async ({ params, request }) => {
      const id = norm(String(params.pid));
      const list = patients.get(id);
      if (!list) return HttpResponse.json({ detail: "not found" }, { status: 404 });
      const body = (await request.json()) as { visit_ids: number[] };
      const owned = list.filter((v) => body.visit_ids.includes(v.id));
      if (owned.length !== body.visit_ids.length)
        return HttpResponse.json(
          { detail: "One or more selected visits do not belong to this patient." },
          { status: 404 },
        );
      if (owned.length > limit)
        return HttpResponse.json(
          { detail: `The model accepts at most ${limit} visits (observation window).` },
          { status: 422 },
        );
      const sorted = [...owned].sort((a, b) => a.visit_date.localeCompare(b.visit_date));
      return HttpResponse.json({
        patient_id: id,
        probability: 0.3,
        prediction: 0,
        threshold: 0.5,
        model: "gru",
        representation: "sequence",
        n_visits: sorted.length,
        disclaimer: "Synthetic demonstration model. Not clinical evidence.",
        selected_visit_ids: sorted.map((v) => v.id),
        selected_visit_dates: sorted.map((v) => v.visit_date),
      });
    }),
  ];

  return { handlers, patients };
}
