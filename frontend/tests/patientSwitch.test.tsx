import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import App from "../src/App";
import { server } from "./msw/server";
import { makePatientStore } from "./msw/patientStore";

async function openPatient(id: string) {
  // Works for both the first step and the "Change" flow.
  const change = screen.queryByTestId("patient-id-change");
  if (change) await userEvent.click(change);
  const input = await screen.findByTestId("patient-id-input");
  await userEvent.clear(input);
  await userEvent.type(input, id);
  await userEvent.click(screen.getByTestId("patient-id-confirm"));
  await screen.findByTestId("trajectory-section");
}

async function addAndSaveVisit(dateValue: string, hba1c: string) {
  await userEvent.click(screen.getByTestId("add-visit"));
  const dateInputs = document.querySelectorAll('input[id^="visit_date-"]');
  const last = dateInputs[dateInputs.length - 1] as HTMLInputElement;
  const idx = last.id.split("-")[1];
  await userEvent.type(last, dateValue);
  await userEvent.type(document.getElementById(`hba1c-${idx}`) as HTMLElement, hba1c);
  await userEvent.click(screen.getByTestId("save-visits"));
  await waitFor(() => expect(screen.queryByTestId("unsaved-badge")).not.toBeInTheDocument());
}

/**
 * Mandatory regression: SYN_0001 → SYN_0002 → SYN_0001.
 * Data isolation, prediction clearing on switch, and persistence (shared store
 * across the simulated "reload"). Uses the stateful MSW patient store.
 */
describe("Mandatory patient-switch sequence", () => {
  it("isolates histories, clears predictions, and persists across reload", async () => {
    const store = makePatientStore();
    server.use(...store.handlers);

    // --- Step A: open SYN_0001 (new, empty) and give it one saved visit ---
    const { unmount } = render(<App />);
    await openPatient("SYN_0001");
    expect(screen.getByTestId("saved-empty")).toBeInTheDocument();
    await addAndSaveVisit("2018-01-01", "5.4");
    expect(screen.getByTestId("saved-visits")).toHaveTextContent("2018-01-01");

    // Predict for SYN_0001 so we can verify it clears on switch.
    await waitFor(() => expect(screen.getByTestId("predict-button")).toBeEnabled());
    await userEvent.click(screen.getByTestId("predict-button"));
    await screen.findByTestId("prediction-result");

    // --- Step B: switch to SYN_0002 (new, empty) ---
    await openPatient("SYN_0002");
    // No SYN_0001 visit leaks; prediction cleared.
    expect(screen.getByTestId("saved-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("prediction-result")).not.toBeInTheDocument();
    expect(screen.getByTestId("saved-visits")).not.toHaveTextContent("2018-01-01");

    // --- Step C: add two visits to SYN_0002 ---
    await addAndSaveVisit("2020-01-01", "6.8");
    await addAndSaveVisit("2020-06-01", "7.1");
    expect(store.patients.get("SYN_0002")).toHaveLength(2);
    // SYN_0001 still has exactly its one visit in the store.
    expect(store.patients.get("SYN_0001")).toHaveLength(1);

    // --- Step D: switch back to SYN_0001 — original visit restored, no SYN_0002 ---
    await openPatient("SYN_0001");
    await waitFor(() => expect(screen.getByTestId("saved-visits")).toHaveTextContent("2018-01-01"));
    expect(screen.getByTestId("saved-visits")).not.toHaveTextContent("2020-01-01");
    expect(screen.queryByTestId("prediction-result")).not.toBeInTheDocument();

    // --- Step E: "reload" the app (unmount + remount, same backing store) ---
    unmount();
    render(<App />);
    await openPatient("SYN_0001");
    await waitFor(() => expect(screen.getByTestId("saved-visits")).toHaveTextContent("2018-01-01"));
    expect(screen.getByTestId("saved-visits")).not.toHaveTextContent("2020-01-01");

    await openPatient("SYN_0002");
    await waitFor(() => expect(screen.getByTestId("saved-visits")).toHaveTextContent("2020-01-01"));
    expect(screen.getByTestId("saved-visits")).toHaveTextContent("2020-06-01");
    expect(screen.getByTestId("saved-visits")).not.toHaveTextContent("2018-01-01");
  });
});
