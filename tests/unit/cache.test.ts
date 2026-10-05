import { test, expect } from "vitest";
import { getModel } from "../../src/content/models";
import { estimateFullAttentionKvBytes } from "../../src/simulation/memory";
import { createPagedState, stepPaged } from "../../src/simulation/paging";
import { buildPrefixTree } from "../../src/simulation/prefix";
test("hybrid_kv_only_counts_full_attention", () => {
  expect(
    estimateFullAttentionKvBytes(getModel("qwen38-dense"), 1024, 1, 2),
  ).toBe(67108864);
  expect(estimateFullAttentionKvBytes(getModel("qwen36-moe"), 1024, 1, 2)).toBe(
    20971520,
  );
  expect(() =>
    estimateFullAttentionKvBytes(getModel("teaching"), Infinity, 1, 2),
  ).toThrow();
});
test("paging_conserves_unique_blocks_and_is_atomic_on_oom", () => {
  const empty = createPagedState(3, 4);
  const a = stepPaged(empty, { type: "allocate", id: "a", tokens: 5 });
  expect(a.requests.a.blocks).toHaveLength(2);
  expect(empty.freeBlocks).toHaveLength(3);
  expect(() => stepPaged(a, { type: "allocate", id: "b", tokens: 8 })).toThrow(
    "不足",
  );
  const b = stepPaged(a, { type: "allocate", id: "b", tokens: 4 });
  expect(
    new Set([...b.requests.a.blocks, ...b.requests.b.blocks, ...b.freeBlocks])
      .size,
  ).toBe(3);
  const released = stepPaged(b, { type: "release", id: "a" });
  expect(released.freeBlocks).toHaveLength(2);
  expect(
    stepPaged(released, { type: "allocate", id: "c", tokens: 5 }).requests.c
      .blocks,
  ).toHaveLength(2);
  expect(() =>
    stepPaged(empty, { type: "allocate", id: "__proto__", tokens: 1 }),
  ).toThrow();
});
test("prefix_tree_shares_only_equal_leading_tokens", () => {
  const tree = buildPrefixTree([
    ["a", "b", "x"],
    ["a", "b", "y"],
    ["a", "c"],
  ]);
  expect(tree.children[0].count).toBe(3);
  expect(tree.children[0].children[0].count).toBe(2);
});
