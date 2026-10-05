import { test, expect } from "vitest";
import { parseRouteState } from "../../src/lib/urlState";
test("invalid_duplicate_and_unknown_URL_state_recovers", () => {
  expect(
    parseRouteState(new URLSearchParams("model=unknown")).warning,
  ).toBeTruthy();
  expect(
    parseRouteState(new URLSearchParams("model=qwen36-moe&depth=expert")),
  ).toMatchObject({ modelId: "qwen36-moe", depth: "expert" });
  expect(
    parseRouteState(new URLSearchParams("depth=expert&depth=beginner")).warning,
  ).toBeTruthy();
});
import {
  parseExperimentParams,
  serializeExperimentParams,
} from "../../src/lib/urlState";
test("experiment_params_recover_and_roundtrip", () => {
  expect(
    parseExperimentParams(
      "kv-cache",
      new URLSearchParams("tokens=4096&batch=2&bytes=1"),
    ).values.tokens,
  ).toBe(4096);
  expect(
    parseExperimentParams(
      "kv-cache",
      new URLSearchParams("tokens=1e99&batch=-1&bytes=3"),
    ).warning,
  ).toBeTruthy();
  expect(
    parseExperimentParams(
      "paged-attention",
      new URLSearchParams("capacity=2&capacity=4"),
    ).warning,
  ).toBeTruthy();
  expect(
    parseExperimentParams(
      "kv-cache",
      serializeExperimentParams("kv-cache", {
        tokens: 8192,
        batch: 4,
        bytes: 0.5,
      }),
    ).values,
  ).toEqual({ tokens: 8192, batch: 4, bytes: 0.5 });
});
