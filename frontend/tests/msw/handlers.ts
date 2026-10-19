import { http, HttpResponse } from "msw";

const BASE = "http://localhost:8000";

export const healthLoaded = http.get(`${BASE}/health`, () =>
  HttpResponse.json({ status: "ok", model_loaded: true, detail: "Model artifacts loaded." }),
);

export const healthNotLoaded = http.get(`${BASE}/health`, () =>
  HttpResponse.json({
    status: "ok",
    model_loaded: false,
    detail: "Model artifacts not loaded; prediction is unavailable.",
  }),
);

export const metadataLoaded = http.get(`${BASE}/metadata`, () =>
  HttpResponse.json({
    model_loaded: true,
    model: "lightgbm",
    representation: "wide",
    threshold: 0.5,
    observation_visits: 4,
    synthetic_data: true,
    disclaimer: "Synthetic demonstration model. Not clinical evidence.",
    extra: { optuna_n_trials_completed: 40, seed: 42 },
  }),
);

export const metadataNotLoaded = http.get(`${BASE}/metadata`, () =>
  HttpResponse.json({
    model_loaded: false,
    model: null,
    representation: null,
    threshold: null,
    observation_visits: null,
    synthetic_data: null,
    disclaimer: "Synthetic demonstration model. Not clinical evidence.",
    extra: {},
  }),
);

export const predictSuccess = http.post(`${BASE}/predict`, async () =>
  HttpResponse.json({
    patient_id: "SYN_0001",
    probability: 0.1234,
    prediction: 0,
    threshold: 0.5,
    model: "lightgbm",
    representation: "wide",
    n_visits: 2,
    disclaimer: "Synthetic demonstration model. Not clinical evidence.",
  }),
);

export const predict503 = http.post(`${BASE}/predict`, () =>
  HttpResponse.json(
    { detail: "Model artifacts are not available; prediction is unavailable." },
    { status: 503 },
  ),
);

export const predict422String = http.post(`${BASE}/predict`, () =>
  HttpResponse.json(
    {
      detail:
        "Visit #0 contains forbidden future/outcome field(s) ['target']. Submit observation data only.",
    },
    { status: 422 },
  ),
);

export const predict422Pydantic = http.post(`${BASE}/predict`, () =>
  HttpResponse.json(
    {
      detail: [
        { loc: ["body", "visits", 0, "visit_date"], msg: "Field required", type: "missing" },
      ],
    },
    { status: 422 },
  ),
);

export const healthNetworkError = http.get(`${BASE}/health`, () => HttpResponse.error());
export const metadataNetworkError = http.get(`${BASE}/metadata`, () => HttpResponse.error());
export const predictNetworkError = http.post(`${BASE}/predict`, () => HttpResponse.error());
