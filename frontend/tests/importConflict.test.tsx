import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import App from "../src/App";
import { server } from "./msw/server";
import { makePatientStore } from "./msw/patientStore";

async function openPatient(id: string) {
  const input = await screen.findByTestId("patient-id-input");
  await userEvent.clear(input);
  await userEvent.type(input, id);
  await userEvent.click(screen.getByTestId("patient-id-confirm"));
  await screen.findByTestId("trajectory-section");
}

async function pasteJson(text: string) {
  const area = screen.getByTestId("json-textarea") as HTMLTextAreaElement;
  await userEvent.clear(area);
  await userEvent.click(area);
  await userEvent.paste(text);
  await userEvent.click(screen.getByRole("button", { name: /import from text/i }));
}

describe("App — JSON import into the active patient", () => {
  it("imports matching-id visits as unsaved drafts (not auto-saved)", async () => {
    const { handlers, patients } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await openPatient("SYN_0001");

    await pasteJson(
      JSON.stringify({
        patient_id: "SYN_0001",
        visits: [
          { visit_date: "2020-01-01", visit_number: 1, hba1c: 5.6, _disclaimer: "x" },
          { visit_date: "2020-06-01", hba1c: 5.9 },
        ],
      }),
    );

    // Imported visits are drafts (unsaved); not yet persisted.
    expect(screen.getByTestId("unsaved-badge")).toBeInTheDocument();
    expect(patients.get("SYN_0001")).toHaveLength(0);
    // Metadata key stripped; clinical field preserved in the draft editor.
    expect(document.getElementById("hba1c-0")).toHaveValue(5.6);

    // Saving persists them to the active patient only.
    await userEvent.click(screen.getByTestId("save-visits"));
    await waitFor(() => expect(patients.get("SYN_0001")).toHaveLength(2));
  });

  it("flags a conflicting patient_id and requires explicit resolution", async () => {
    const { handlers } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await openPatient("SYN_0001");

    await pasteJson(
      JSON.stringify({ patient_id: "OTHER_9999", visits: [{ visit_date: "2020-01-01", hba1c: 6.1 }] }),
    );

    const conflict = await screen.findByTestId("import-conflict");
    expect(conflict).toHaveTextContent(/OTHER_9999/);
    expect(conflict).toHaveTextContent(/SYN_0001/);
    // No drafts added yet (awaiting resolution).
    expect(screen.queryByTestId("unsaved-badge")).not.toBeInTheDocument();

    // Resolve: add to the active patient.
    await userEvent.click(screen.getByRole("button", { name: /add to SYN_0001/i }));
    expect(screen.queryByTestId("import-conflict")).not.toBeInTheDocument();
    expect(screen.getByTestId("unsaved-badge")).toBeInTheDocument();
  });

  it("cancels a conflicting import without changing the patient", async () => {
    const { handlers } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await openPatient("SYN_0001");
    await pasteJson(
      JSON.stringify({ patient_id: "OTHER", visits: [{ visit_date: "2020-01-01" }] }),
    );
    await screen.findByTestId("import-conflict");
    await userEvent.click(screen.getByRole("button", { name: /cancel import/i }));
    expect(screen.queryByTestId("import-conflict")).not.toBeInTheDocument();
    expect(screen.queryByTestId("unsaved-badge")).not.toBeInTheDocument();
  });

  it("shows a readable error for malformed JSON", async () => {
    const { handlers } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await openPatient("SYN_0001");
    await pasteJson("{ not valid json");
    expect(await screen.findByTestId("import-error")).toHaveTextContent(/not valid json/i);
  });
});
