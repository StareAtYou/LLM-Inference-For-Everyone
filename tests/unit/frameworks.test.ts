import { expect, test } from "vitest";
import { frameworks } from "../../src/content/frameworks";
import { topics } from "../../src/content/topics";
test("framework_graphs_are_versioned_and_internally_connected", () => {
  expect(frameworks).toHaveLength(2);
  for (const f of frameworks) {
    expect(f.modules.length).toBeGreaterThanOrEqual(6);
    expect(f.commit).toMatch(/^[a-f0-9]{40}$/);
    for (const edge of f.edges) {
      expect(f.modules.some((m) => m.id === edge.from)).toBe(true);
      expect(f.modules.some((m) => m.id === edge.to)).toBe(true);
    }
    for (const m of f.modules) {
      expect(m.snippet.code.length).toBeGreaterThan(10);
      for (const t of m.relatedTopics)
        expect(topics.some((x) => x.id === t)).toBe(true);
      for (const s of m.sources) {
        expect(s.url).toContain(f.commit);
        expect(s.path).toBeTruthy();
        expect(s.symbol).toBeTruthy();
        expect(s.startLine).toBeGreaterThan(0);
      }
    }
  }
});
test("cache_explanations_distinguish_block_hash_and_radix", () => {
  expect(
    frameworks
      .find((f) => f.id === "vllm")
      ?.modules.find((m) => m.id === "blocks")?.responsibility,
  ).toContain("块哈希");
  expect(
    frameworks
      .find((f) => f.id === "sglang")
      ?.modules.find((m) => m.id === "cache")?.responsibility,
  ).toContain("Radix");
});
