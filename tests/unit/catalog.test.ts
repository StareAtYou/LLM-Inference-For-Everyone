import { test, expect } from "vitest";
import { models } from "../../src/content/models";
import { topics } from "../../src/content/topics";
import { searchTopics } from "../../src/lib/search";
test("real_Qwen_shapes_use_hybrid_layers_and_shared_experts", () => {
  const d = models.find((m) => m.id === "qwen38-dense")!;
  const m = models.find((m) => m.id === "qwen36-moe")!;
  expect(d?.layerTypes.length).toBe(64);
  expect(d?.layerTypes.filter((t) => t === "full_attention")).toHaveLength(16);
  expect(d?.headDim).toBe(256);
  expect(m?.expertCount).toBe(256);
  expect(m?.activeExperts).toBe(8);
  expect(m?.sharedExperts).toBe(1);
});
test("search_resolves_English_terms_and_catalog_references", () => {
  expect(topics.length).toBeGreaterThanOrEqual(20);
  expect(
    searchTopics(topics, "continuous batching", "beginner").some(
      (t) => t.id === "batching",
    ),
  ).toBe(true);
  const ids = new Set(topics.map((t) => t.id));
  for (const t of topics) {
    for (const id of t.prerequisites) expect(ids.has(id)).toBe(true);
    expect(t.levels.beginner.explanation).not.toBe(t.levels.expert.explanation);
  }
});

test("depths_add_distinct_information_in_every_topic", () => {
  for (const topic of topics) {
    const paragraphs = (depth: "beginner" | "advanced" | "expert") =>
      [topic.levels[depth].explanation, ...topic.levels[depth].details].sort();
    expect(paragraphs("advanced"), topic.id).not.toEqual(
      paragraphs("beginner"),
    );
    expect(
      topic.levels.expert.details.some(
        (x) =>
          x ===
          "核对实现时，先追踪输入输出张量、状态生命周期与适用条件，再用固定负载比较效果。",
      ),
      topic.id,
    ).toBe(false);
  }
});
