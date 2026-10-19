// Canonical feature names and value domains, mirroring the backend
// (dialong_automl/features/wide.py NUMERIC_FEATURES / CATEGORICAL_FEATURES and
// the leakage set). Categorical options are *suggestions* taken from the
// synthetic generator; the model maps unknown categories to an <UNK> slot, so
// these are not hard constraints — but we surface them for convenience.

export const NUMERIC_FEATURES = [
  "hba1c",
  "fasting_glucose",
  "bmi",
  "systolic_bp",
  "diastolic_bp",
  "ldl_cholesterol",
  "hdl_cholesterol",
  "triglycerides",
  "heart_rate",
  "age_at_visit",
] as const;

export const CATEGORICAL_FEATURES = [
  "sex",
  "smoking_status",
  "physical_activity_level",
  "family_history_diabetes",
  "metformin_use",
  "hypertension_diagnosis",
] as const;

export const CATEGORICAL_OPTIONS: Record<string, string[]> = {
  sex: ["M", "F"],
  smoking_status: ["never", "former", "current"],
  physical_activity_level: ["sedentary", "moderate", "active"],
  family_history_diabetes: ["yes", "no"],
  metformin_use: ["yes", "no"],
  hypertension_diagnosis: ["yes", "no"],
};

// Forbidden future/outcome/leakage fields — must match the backend set in
// dialong_automl/features/wide.py (_REPRESENTATION_LEAKAGE_COLUMNS).
export const FORBIDDEN_FIELDS = new Set<string>([
  "target",
  "is_future_visit",
  "future_hba1c",
  "future_glucose",
  "future_diabetes",
  "diabetes_progression",
  "diagnosis_after",
  "post_outcome",
  "outcome",
]);

// Keys present in the project's example JSON that are not part of the model's
// feature schema. `_disclaimer` is metadata; `visit_number` is positional only.
// Neither is a leakage field, so stripping them is safe and keeps requests clean.
export const NON_SCHEMA_IMPORT_KEYS = new Set<string>([
  "_disclaimer",
  "visit_number",
]);

export const NUMERIC_LABELS: Record<string, string> = {
  hba1c: "HbA1c (%)",
  fasting_glucose: "Fasting glucose (mg/dL)",
  bmi: "BMI (kg/m²)",
  systolic_bp: "Systolic BP (mmHg)",
  diastolic_bp: "Diastolic BP (mmHg)",
  ldl_cholesterol: "LDL cholesterol (mg/dL)",
  hdl_cholesterol: "HDL cholesterol (mg/dL)",
  triglycerides: "Triglycerides (mg/dL)",
  heart_rate: "Heart rate (bpm)",
  age_at_visit: "Age at visit (years)",
};

export const CATEGORICAL_LABELS: Record<string, string> = {
  sex: "Sex",
  smoking_status: "Smoking status",
  physical_activity_level: "Physical activity",
  family_history_diabetes: "Family history of diabetes",
  metformin_use: "Metformin use",
  hypertension_diagnosis: "Hypertension diagnosis",
};

// Plausible value ranges (inclusive) for numeric features. These mirror the
// synthetic generator's clip bounds in dialong_automl/data/synthetic_generator.py
// and are used ONLY to catch obviously out-of-range entries before submission.
// They are schema sanity bounds, not clinical reference ranges.
export const NUMERIC_RANGES: Record<string, { min: number; max: number }> = {
  hba1c: { min: 3.0, max: 20.0 },
  fasting_glucose: { min: 40, max: 500 },
  bmi: { min: 10, max: 70 },
  systolic_bp: { min: 70, max: 220 },
  diastolic_bp: { min: 40, max: 140 },
  ldl_cholesterol: { min: 20, max: 300 },
  hdl_cholesterol: { min: 10, max: 150 },
  triglycerides: { min: 20, max: 700 },
  heart_rate: { min: 30, max: 160 },
  age_at_visit: { min: 0, max: 120 },
};

// `visit_date` is the only hard-required field per visit. Clinical features are
// optional: the backend's fitted preprocessor imputes genuinely-missing values
// using training-time statistics (identical to how it handles missing values
// during training). The UI surfaces a soft "completeness" notice rather than
// blocking prediction, so partial trajectories remain usable but encouraged to
// be complete.
export const ALL_CLINICAL_FEATURES: string[] = [
  ...NUMERIC_FEATURES,
  ...CATEGORICAL_FEATURES,
];
