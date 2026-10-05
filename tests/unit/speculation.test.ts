import { test, expect } from "vitest";
import { simulateSpeculation } from "../../src/simulation/speculation";
test("first_rejection_discards_remaining_draft_and_corrects", () => {
  const r = simulateSpeculation(["a", "b", "c"], ["a", "x", "c"], 3, {
    draft: 1,
    verify: 2,
  });
  expect(r.rounds[0].accepted).toEqual(["a"]);
  expect(r.rounds[0].rejected).toEqual(["x", "c"]);
  expect(r.rounds[0].correction).toBe("b");
  expect(r.output).toEqual(["a", "b", "c"]);
});
test("all_accept_and_all_reject_preserve_target_and_high_cost_can_lose", () => {
  const target = ["a", "b", "c", "d"];
  for (const draft of [target, ["x", "x", "x", "x"]])
    expect(
      simulateSpeculation(target, draft, 3, { draft: 0.1, verify: 1 }).output,
    ).toEqual(target);
  const r = simulateSpeculation(target, target, 3, { draft: 5, verify: 1 });
  expect(r.baselineCost / r.speculativeCost).toBeLessThan(1);
  expect(() =>
    simulateSpeculation(target, target, 0, { draft: 0.1, verify: 1 }),
  ).toThrow();
  expect(() =>
    simulateSpeculation(target, target, 2, { draft: 0, verify: 1 }),
  ).toThrow();
});
