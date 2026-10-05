import { test, expect } from "vitest";
import { routeExperts } from "../../src/simulation/experts";
import { getModel } from "../../src/content/models";
test("top_k_routes_are_unique_normalized_and_stable", () => {
  expect(routeExperts([1, 4, 2, 3], 2).map((x) => x.id)).toEqual([1, 3]);
  const r = routeExperts(
    Array.from({ length: 256 }, (_, i) => Math.sin(i)),
    8,
  );
  expect(new Set(r.map((x) => x.id)).size).toBe(8);
  expect(r.reduce((s, x) => s + x.weight, 0)).toBeCloseTo(1);
  expect(routeExperts([0, 0, 0], 2).map((x) => x.id)).toEqual([0, 1]);
  expect(() => routeExperts([1], 2)).toThrow();
});
test("real_hybrid_shape_comes_from_config", () => {
  const m = getModel("qwen38-dense");
  expect(m.qHeads * m.headDim).toBe(6144);
  expect(
    m.layerTypes.every(
      (t, i) => t === (i % 4 === 3 ? "full_attention" : "linear_attention"),
    ),
  ).toBe(true);
  expect(getModel("qwen36-moe").sharedExperts).toBe(1);
});
