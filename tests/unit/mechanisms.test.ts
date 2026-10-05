import { describe, expect, test } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { buildMechanismFrames, mechanisms } from "../../src/mechanisms/catalog";
import MechanismPlayer from "../../src/mechanisms/MechanismPlayer";

const expectedIds = [
  "tokenization",
  "embedding",
  "attention",
  "rope",
  "ffn",
  "gqa",
  "gated-deltanet",
  "moe",
  "prefill",
  "decode",
  "sampling",
  "kv-cache",
  "paged-attention",
  "prefix-caching",
  "batching",
  "quantization",
  "speculation",
  "flash-attention",
  "parallelism",
  "chunked-prefill",
  "cuda-graphs",
  "metrics",
  "frameworks",
  "rmsnorm",
  "qkv",
  "residual",
  "lm-head",
  "softmax",
  "attention-output",
];
const last = (id: string, params?: Record<string, number>) =>
  buildMechanismFrames(id, params).at(-1)!;

describe("independent principle calculations", () => {
  test("every public principle has its own complete causal explanation", () => {
    expect(mechanisms.map((m) => m.id).sort()).toEqual(expectedIds.sort());
    const signatures = new Set<string>();
    for (const m of mechanisms) {
      const frames = buildMechanismFrames(m.id);
      expect(frames.length).toBeGreaterThanOrEqual(3);
      expect(frames.length).toBeLessThanOrEqual(8);
      expect(new Set(frames.map((f) => f.caption)).size).toBe(frames.length);
      expect(
        frames.every((f) => f.panels.length > 0 && f.caption.length > 15),
      ).toBe(true);
      signatures.add(frames.map((f) => f.stage).join("|"));
      const numbers = frames.flatMap((f) => [
        ...(f.output ?? []),
        ...Object.values(f.readouts ?? {}),
      ]);
      expect(numbers.every(Number.isFinite)).toBe(true);
    }
    expect(signatures.size).toBe(expectedIds.length);
  });
  test("RMS normalization computes root mean square and remains finite for zero input", () => {
    const f = last("rmsnorm");
    expect(f.readouts.rms).toBeCloseTo(Math.sqrt(2.5 + 1e-6));
    expect(f.output).toEqual(
      [1, -1, 2, -2].map((x) => x / Math.sqrt(2.5 + 1e-6)),
    );
    expect(last("rmsnorm", { amplitude: 0 }).output).toEqual([0, 0, 0, 0]);
  });
  test("QKV projections expose actual distinct matrix products", () => {
    const f = last("qkv");
    expect(f.output).toEqual([2, 0, 1, -1, 1, 1]);
    expect(
      buildMechanismFrames("qkv").some((x) => x.readouts.contribution === 2),
    ).toBe(true);
  });
  test("RoPE preserves pair length and changes with position", () => {
    const zero = last("rope", { position: 0 }).output!;
    const rotated = last("rope", { position: 3 }).output!;
    expect(zero).toEqual([1, 0]);
    expect(rotated[0] ** 2 + rotated[1] ** 2).toBeCloseTo(1);
    expect(rotated).not.toEqual(zero);
  });
  test("attention masks future tokens before softmax then weights V", () => {
    const f = last("attention", { row: 1 });
    const weights = f.panels.find((p) => p.label === "因果注意力权重")!.matrix!;
    expect(weights[1][2]).toBe(0);
    expect(weights[1].reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(f.output![0]).toBeCloseTo(weights[1][0] * 1 + weights[1][1] * 0);
    expect(f.output![1]).toBeCloseTo(weights[1][0] * 0 + weights[1][1] * 2);
  });
  test("softmax is normalized at normal and zero temperature", () => {
    expect(last("softmax").output!.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(last("softmax", { temperature: 0 }).output).toEqual([0, 0, 1, 0]);
    expect(last("softmax", { temperature: 2 }).output).not.toEqual(
      last("softmax").output,
    );
  });
  test("residual addition and output projection produce visible calculated vectors", () => {
    expect(last("residual").output).toEqual([1.2, -0.5, 1.7, -1.9]);
    expect(last("attention-output").output).toEqual([1, 3]);
    expect(last("lm-head").output).toEqual([1, -1, 1.5, 0.5]);
  });
  test("quantization configuration changes codes and errors with finite all-zero handling", () => {
    const four = last("quantization", { bits: 4 });
    const eight = last("quantization", { bits: 8 });
    expect(eight.readouts.mse).toBeLessThan(four.readouts.mse);
    expect(eight.readouts.payloadBytes).toBe(6);
    expect(four.readouts.payloadBytes).toBe(3);
    expect(last("quantization", { amplitude: 0 }).output).toEqual([
      0, 0, 0, 0, 0, 0,
    ]);
  });
  test("paging recomputes block allocations and batching shows waiting then reuse", () => {
    expect(last("paged-attention", { blockSize: 2 }).readouts.blocks).toBe(4);
    expect(last("paged-attention", { blockSize: 4 }).readouts.blocks).toBe(3);
    const frames = buildMechanismFrames("batching", { slots: 1 });
    expect(frames.some((f) => f.readouts.waiting > 0)).toBe(true);
    expect(frames.at(-1)!.readouts.completed).toBe(3);
    expect(frames.at(-1)!.readouts.allocated).toBe(0);
    expect(
      buildMechanismFrames("batching", { slots: 2 })[1].readouts.waiting,
    ).toBeLessThan(frames[1].readouts.waiting);
  });
  test("KV cache appends consumed tokens only and releases after completion", () => {
    const frames = buildMechanismFrames("kv-cache");
    expect(
      frames.find((f) => f.stage === "采样尚未写入")!.readouts.cachedTokens,
    ).toBe(3);
    expect(
      frames.find((f) => f.stage === "下一轮写入")!.readouts.cachedTokens,
    ).toBe(4);
    expect(frames.at(-1)!.readouts.cachedTokens).toBe(0);
  });
  test("bounded parameters reject nonfinite numbers and unknown principles return no frames", () => {
    expect(() => buildMechanismFrames("rmsnorm", { amplitude: NaN })).toThrow();
    expect(() =>
      buildMechanismFrames("paged-attention", { blockSize: 0 }),
    ).toThrow();
    expect(() => buildMechanismFrames("quantization", { bits: 6 })).toThrow();
    expect(() => buildMechanismFrames("gqa", { kvHeads: 3 })).toThrow();
    expect(() => buildMechanismFrames("attention", { row: 1.5 })).toThrow();
    expect(buildMechanismFrames("unknown")).toEqual([]);
  });
  test("player exposes local controls and explicit unknown principle fallback", () => {
    const html = renderToStaticMarkup(
      createElement(MechanismPlayer, { id: "rmsnorm" }),
    );
    expect(html).toContain('data-mechanism-id="rmsnorm"');
    expect(html).toContain('aria-label="播放原理动图"');
    expect(html).toContain('aria-label="原理动画进度"');
    expect(
      renderToStaticMarkup(createElement(MechanismPlayer, { id: "unknown" })),
    ).toContain("尚未收录这个原理");
  });
});
