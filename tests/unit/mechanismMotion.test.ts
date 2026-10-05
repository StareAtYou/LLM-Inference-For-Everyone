import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";
import ContinuousMechanismScene from "../../src/mechanisms/ContinuousMechanismScene";
import { buildMechanismFrames, mechanisms } from "../../src/mechanisms/catalog";

const render = (
  id: string,
  position: number,
  params?: Record<string, number>,
) =>
  renderToStaticMarkup(
    createElement(ContinuousMechanismScene, {
      definition: mechanisms.find((m) => m.id === id)!,
      frames: buildMechanismFrames(id, params),
      position,
    }),
  );
const objects = (html: string) =>
  [
    ...html.matchAll(/data-motion-object="([^"]+)" data-geometry="([^"]+)"/g),
  ].map((m) => ({
    id: m[1],
    geometry: JSON.parse(m[2].replaceAll("&quot;", '"')) as number[],
  }));

// Catches fractional position being ignored, stage remounts, and geometry resets at boundaries.
describe("continuous mechanism object choreography", () => {
  test("every module transforms persistent objects during every operation", () => {
    for (const definition of mechanisms) {
      const frames = buildMechanismFrames(definition.id);
      for (let i = 0; i < frames.length - 1; i++) {
        const before = objects(render(definition.id, i + 0.2));
        const after = objects(render(definition.id, i + 0.8));
        expect(before.length, definition.id).toBeGreaterThan(0);
        expect(
          after.map((o) => o.id),
          definition.id,
        ).toEqual(before.map((o) => o.id));
        expect(
          after.some((o, j) =>
            o.geometry.some((v, k) => Math.abs(v - before[j].geometry[k]) > 1),
          ),
          `${definition.id} operation ${i}`,
        ).toBe(true);
      }
    }
  });
  test("handoffs remain continuous on both sides of every integer boundary", () => {
    for (const definition of mechanisms) {
      const frames = buildMechanismFrames(definition.id);
      for (let i = 1; i < frames.length - 1; i++) {
        const before = objects(render(definition.id, i - 0.00001));
        const after = objects(render(definition.id, i + 0.00001));
        expect(before.length).toBeGreaterThan(0);
        expect(after.map((o) => o.id)).toEqual(before.map((o) => o.id));
        for (let j = 0; j < before.length; j++)
          for (let k = 0; k < before[j].geometry.length; k++) {
            expect(
              Math.abs(after[j].geometry[k] - before[j].geometry[k]),
              `${definition.id} boundary ${i}`,
            ).toBeLessThan(0.01);
          }
      }
    }
  });
  test("RoPE continuously rotates the same vector while preserving its visual length", () => {
    const first = render("rope", 1.2),
      later = render("rope", 1.8);
    const angle = (s: string) =>
      Number(s.match(/data-rotation-angle="([^"]+)"/)?.[1]);
    expect(angle(first)).toBeGreaterThan(0);
    expect(angle(later)).toBeGreaterThan(angle(first));
    for (const s of [first, later]) {
      const tip =
        s
          .match(/data-rotation-tip="([^"]+)"/)?.[1]
          ?.split(",")
          .map(Number) ?? [];
      expect(tip).toHaveLength(2);
      expect(Math.hypot(...tip)).toBeCloseTo(76, 8);
    }
  });
  test("MoE branches split through selected experts and converge into one result", () => {
    const split = objects(render("moe", 1.8));
    const merged = objects(render("moe", 3));
    const spread = (os: typeof split) =>
      Math.max(...os.map((o) => o.geometry[1])) -
      Math.min(...os.map((o) => o.geometry[1]));
    expect(split.length).toBeGreaterThan(1);
    expect(spread(split)).toBeGreaterThan(80);
    expect(spread(merged)).toBeLessThan(40);
  });
  test("seeking is deterministic and completed cache writes do not disappear at a subsequent operation", () => {
    expect(objects(render("kv-cache", 1.4))).toEqual(
      objects(render("kv-cache", 1.4)),
    );
    const count = (p: number) =>
      Number(
        render("kv-cache", p).match(/data-cache-occupancy="([^"]+)"/)?.[1],
      );
    expect(count(0)).toBe(0);
    expect(count(1)).toBe(3);
    expect(count(2)).toBe(3);
    expect(count(3)).toBe(4);
    expect(count(4)).toBe(0);
  });
  test("speculative rejection sends only the rejected suffix into the rollback lane", () => {
    const before = objects(render("speculation", 0.8));
    const after = objects(render("speculation", 2.8));
    expect(after).toHaveLength(4);
    expect(after[0].geometry[1]).toBeLessThan(220);
    expect(after[1].geometry[1]).toBeLessThan(220);
    expect(after[2].geometry[1]).toBeGreaterThan(240);
    expect(after[2].geometry[1]).toBeGreaterThan(before[2].geometry[1]);
  });
  test("shared prefix and completed prefill blocks remain available in later operations", () => {
    const occupancy = (id: string, p: number) =>
      Number(render(id, p).match(/data-cache-occupancy="([^"]+)"/)?.[1]);
    expect(occupancy("prefix-caching", 2)).toBe(3);
    expect(occupancy("prefix-caching", 3)).toBe(3);
    expect(occupancy("chunked-prefill", 3)).toBe(6);
    expect(occupancy("chunked-prefill", 4)).toBe(6);
  });
});

test("RoPE position metadata never becomes an output channel or overflows the scene", () => {
  const before = objects(render("rope", 1, { position: 0 })),
    after = objects(render("rope", 1, { position: 8 }));
  expect(after.map((o) => o.geometry)).toEqual(before.map((o) => o.geometry));
  for (const p of [0, 0.5, 1, 1.5, 2])
    for (const o of objects(render("rope", p, { position: 8 }))) {
      expect(o.geometry[1] + o.geometry[3]).toBeLessThan(355);
    }
});
