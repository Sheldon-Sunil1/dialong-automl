// Centralized frontend configuration.
// The API base URL comes from a Vite env variable so it is configurable per
// environment without code changes. Falls back to the local dev backend.

const RAW_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000";

// Strip a trailing slash so path joins are predictable.
export const API_BASE_URL = RAW_BASE.replace(/\/+$/, "");
