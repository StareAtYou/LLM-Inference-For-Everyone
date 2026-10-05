import { test, expect } from "vitest";
import { simulateBatching } from "../../src/simulation/batching";
test("arrival_work_and_slot_invariants", () => {
  const requests = [
    { id: "a", arrival: 0, outputTokens: 4 },
    { id: "b", arrival: 0, outputTokens: 1 },
    { id: "late", arrival: 3, outputTokens: 2 },
  ];
  for (const mode of ["static", "continuous"] as const) {
    const result = simulateBatching(requests, 2, mode);
    expect(Object.keys(result.completions).sort()).toEqual(["a", "b", "late"]);
    for (const req of requests) {
      expect(
        result.frames.flatMap((f) => f.slots.filter((id) => id === req.id)),
      ).toHaveLength(req.outputTokens);
      expect(result.completions[req.id]).toBeGreaterThanOrEqual(
        req.arrival + req.outputTokens,
      );
    }
    for (const f of result.frames) {
      expect(new Set(f.slots.filter(Boolean)).size).toBe(
        f.slots.filter(Boolean).length,
      );
      if (f.tick < 3) expect(f.slots).not.toContain("late");
    }
  }
});
test("one_slot_is_equal_and_empty_workload_is_finite", () => {
  const r = [
    { id: "a", arrival: 0, outputTokens: 3 },
    { id: "b", arrival: 0, outputTokens: 2 },
  ];
  expect(simulateBatching(r, 1, "static")).toEqual(
    simulateBatching(r, 1, "continuous"),
  );
  expect(simulateBatching([], 2, "continuous").totalSlots).toBe(0);
  expect(() =>
    simulateBatching(
      [{ id: "bad", arrival: -1, outputTokens: 3 }],
      2,
      "continuous",
    ),
  ).toThrow();
});
