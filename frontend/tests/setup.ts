import "@testing-library/jest-dom/vitest";
import { afterAll, afterEach, beforeAll } from "vitest";

import { server } from "./msw/server";

// Start the MSW mock server for the whole test run. Network is mocked ONLY in
// tests; the running application always talks to the real backend.
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
