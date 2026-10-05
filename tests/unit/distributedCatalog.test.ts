import { expect, test } from "vitest";
import { topics } from "../../src/content/topics";
import { searchTopics } from "../../src/lib/search";

test("distributed strategies and collectives are independently discoverable with valid prerequisites", () => {
  const ids = [
    "tp",
    "dp",
    "ep",
    "pp",
    "cp",
    "sp",
    "all-reduce",
    "all-gather",
    "reduce-scatter",
    "all-to-all",
    "broadcast",
    "reduce",
    "gather",
    "scatter",
    "send-recv",
  ];
  const all = new Set(topics.map((t) => t.id));
  for (const id of ids) {
    const topic = topics.find((t) => t.id === id);
    expect(topic, id).toBeDefined();
    expect(topic!.sources.length).toBeGreaterThan(0);
    expect(
      topic!.links.some((link) => link.to.startsWith("/distributed")),
    ).toBe(true);
    for (const prerequisite of topic!.prerequisites)
      expect(all.has(prerequisite)).toBe(true);
  }
  expect(
    searchTopics(topics, "reduce scatter", "beginner").some(
      (t) => t.id === "reduce-scatter",
    ),
  ).toBe(true);
  expect(
    searchTopics(topics, "context parallel", "expert").some(
      (t) => t.id === "cp",
    ),
  ).toBe(true);
});
