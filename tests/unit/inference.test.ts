import { expect, test } from "vitest";
import { softmax, filterDistribution } from "../../src/simulation/sampling";
import { causalAttention } from "../../src/simulation/attention";
import { buildPipelineFrames } from "../../src/simulation/pipeline";
test("causal_mask_never_sees_future", () => {
  const rows = causalAttention([
    [1, 2, 3],
    [3, 4, 5],
    [2, 1, 4],
  ]);
  expect(rows[0]).toEqual([1, 0, 0]);
  rows.forEach((r, i) => {
    expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(r.slice(i + 1).every((v) => v === 0)).toBe(true);
  });
});
test("sampling_normalizes_filtered_distribution", () => {
  expect(filterDistribution([0.1, 0.7, 0.2], 1, 1)).toEqual([0, 1, 0]);
  expect(
    filterDistribution([0.1, 0.7, 0.2], 3, 0.8).reduce((a, b) => a + b, 0),
  ).toBeCloseTo(1);
  expect(softmax([10000, 9999], 0.001).every(Number.isFinite)).toBe(true);
  expect(() => softmax([], 1)).toThrow();
  expect(() => filterDistribution([0, 0], 2, 1)).toThrow();
});
test("zero_temperature_returns_argmax", () =>
  expect(softmax([1, 3, 2], 0)).toEqual([0, 1, 0]));
test("first_token_comes_from_prefill", () => {
  const frames = buildPipelineFrames("qwen38-dense", "sky", {
    temperature: 1,
    topK: 5,
    topP: 1,
  });
  expect(frames.find((f) => f.outputTokens.length)?.stage).toBe("Prefill");
  expect(frames.at(-1)?.stage).toBe("停止");
});
