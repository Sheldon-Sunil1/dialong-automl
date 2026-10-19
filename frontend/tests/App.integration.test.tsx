import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import App from "../src/App";
import { server } from "./msw/server";
import {
  healthLoaded,
  healthNetworkError,
  metadataLoaded,
  metadataNetworkError,
} from "./msw/handlers";
import { makePatientStore } from "./msw/patientStore";

async function openPatient(id: string) {
  const input = await screen.findByTestId("patient-id-input");
  await userEvent.clear(input);
  await userEvent.type(input, id);
  await userEvent.click(screen.getByTestId("patient-id-confirm"));
  await screen.findByTestId("trajectory-section");
}

async function addAndSaveVisit(dateValue: string, hba1c?: string) {
  await userEvent.click(screen.getByTestId("add-visit"));
  // The newly added draft visit is the last visit panel; find its date input.
  const dateInputs = document.querySelectorAll('input[id^="visit_date-"]');
  const last = dateInputs[dateInputs.length - 1] as HTMLInputElement;
  const idx = last.id.split("-")[1];
  await userEvent.type(last, dateValue);
  if (hba1c !== undefined) {
    await userEvent.type(document.getElementById(`hba1c-${idx}`) as HTMLElement, hba1c);
  }
}

describe("App — status & metadata", () => {
  it("shows loaded health + metadata and the disclaimer", async () => {
    server.use(healthLoaded, metadataLoaded);
    render(<App />);
    await waitFor(() =>
      expect(screen.getByTestId("health-badge")).toHaveTextContent(/model loaded/i),
    );
    expect(
      screen.getAllByText(/not a clinical diagnostic tool|not clinical evidence/i).length,
    ).toBeGreaterThan(0);
  });

  it("shows backend-unreachable on network failure", async () => {
    server.use(healthNetworkError, metadataNetworkError);
    render(<App />);
    await waitFor(() =>
      expect(screen.getByTestId("health-badge")).toHaveTextContent(/unreachable/i),
    );
  });
});

describe("App — Patient ID first step", () => {
  it("initially shows only the Patient ID step", async () => {
    const { handlers } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await screen.findByTestId("patient-id-step");
    expect(screen.queryByTestId("trajectory-section")).not.toBeInTheDocument();
    expect(screen.queryByTestId("saved-visits")).not.toBeInTheDocument();
  });

  it("does not unlock for an empty/whitespace ID", async () => {
    const { handlers } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await screen.findByTestId("patient-id-step");
    expect(screen.getByTestId("patient-id-confirm")).toBeDisabled();
    await userEvent.type(screen.getByTestId("patient-id-input"), "   ");
    expect(screen.getByTestId("patient-id-confirm")).toBeDisabled();
  });

  it("creates a new patient with an empty history and reveals controls", async () => {
    const { handlers } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await openPatient("SYN_NEW");
    expect(screen.getByTestId("new-patient-note")).toBeInTheDocument();
    expect(screen.getByTestId("saved-empty")).toBeInTheDocument();
    expect(screen.getByTestId("add-visit")).toBeInTheDocument();
  });
});

describe("App — visit persistence", () => {
  it("saves drafts to the backend and shows them as saved visits", async () => {
    const { handlers, patients } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await openPatient("SYN_0001");

    await addAndSaveVisit("2020-01-01", "5.6");
    expect(screen.getByTestId("unsaved-badge")).toBeInTheDocument();
    await userEvent.click(screen.getByTestId("save-visits"));

    // Saved visit appears; backend store has it.
    await waitFor(() => expect(screen.getByTestId("saved-visits")).toHaveTextContent("2020-01-01"));
    expect(patients.get("SYN_0001")).toHaveLength(1);
    // Draft cleared after save.
    expect(screen.queryByTestId("unsaved-badge")).not.toBeInTheDocument();
  });
});

describe("App — visit selection & prediction", () => {
  it("selects saved visits and predicts using only the selection", async () => {
    const { handlers } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await openPatient("SYN_0001");

    await addAndSaveVisit("2019-02-11", "5.6");
    await userEvent.click(screen.getByTestId("save-visits"));
    await addAndSaveVisit("2019-08-19", "5.8");
    await userEvent.click(screen.getByTestId("save-visits"));
    await waitFor(() => expect(screen.getByTestId("selector-count")).toHaveTextContent("2 of 2"));

    await userEvent.click(screen.getByTestId("predict-button"));

    const result = await screen.findByTestId("prediction-result");
    expect(within(result).getByTestId("result-probability")).toBeInTheDocument();
    expect(within(result).getByTestId("result-selected-dates")).toHaveTextContent(
      "2019-02-11, 2019-08-19",
    );
  });

  it("clears the prediction when the selection changes", async () => {
    const { handlers } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await openPatient("SYN_0001");
    await addAndSaveVisit("2019-02-11", "5.6");
    await userEvent.click(screen.getByTestId("save-visits"));
    await addAndSaveVisit("2019-08-19", "5.8");
    await userEvent.click(screen.getByTestId("save-visits"));
    await waitFor(() => expect(screen.getByTestId("selector-count")).toHaveTextContent("2 of 2"));

    await userEvent.click(screen.getByTestId("predict-button"));
    await screen.findByTestId("prediction-result");

    // Change the selection (deselect all) → prediction clears immediately.
    await userEvent.click(screen.getByRole("button", { name: /deselect all/i }));
    await waitFor(() =>
      expect(screen.queryByTestId("prediction-result")).not.toBeInTheDocument(),
    );
  });

  it("enforces the observation window (no silent truncation)", async () => {
    const { handlers } = makePatientStore({ observationLimit: 2 });
    server.use(...handlers);
    render(<App />);
    await openPatient("SYN_0001");
    for (const d of ["2019-01-01", "2019-02-01", "2019-03-01"]) {
      await addAndSaveVisit(d, "5.6");
      await userEvent.click(screen.getByTestId("save-visits"));
    }
    await waitFor(() => expect(screen.getByTestId("selector-count")).toHaveTextContent("3 of 3"));
    // All 3 selected > limit 2 → predict disabled + limit note.
    expect(screen.getByTestId("predict-button")).toBeDisabled();
    expect(screen.getByTestId("limit-note")).toHaveTextContent(/observation window/i);
  });
});

describe("App — unavailable features", () => {
  it("documents unavailable features without fabricating data", async () => {
    const { handlers } = makePatientStore();
    server.use(...handlers);
    render(<App />);
    await waitFor(() => expect(screen.getByTestId("health-badge")).toBeInTheDocument());
    await userEvent.click(screen.getByText(/experiments & explainability/i));
    expect(screen.getByRole("heading", { name: /experiment leaderboard/i })).toBeInTheDocument();
    expect(screen.getAllByText(/not available via api/i).length).toBeGreaterThanOrEqual(3);
  });
});
