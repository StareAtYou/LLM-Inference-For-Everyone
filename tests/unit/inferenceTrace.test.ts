import { describe, expect, it } from "vitest";
import {
  buildInferenceTrace,
  type TraceConfig,
  type TraceFrame,
} from "../../src/simulation/inferenceTrace";

const config: TraceConfig = {
  modelId: "qwen38-dense",
  scenarioId: "sky",
  mode: "single",
  capacity: 2,
  temperature: 1,
  topK: 6,
  topP: 1,
  outputLimit: 3,
};
const request = (frame: TraceFrame, id = "R1") =>
  frame.requests.find((r) => r.id === id)!;
const samples = (frames: TraceFrame[], id = "R1") =>
  frames.filter((f) => f.stage === "sample" && f.requestIds.includes(id));

describe("computed inference trace", () => {
  it("distinguishes attention and FFN normalization/residual branches", () => {
    const trace = buildInferenceTrace({ ...config, outputLimit: 1 });
    const firstLayer = trace.filter((frame) => frame.layer === 0);
    expect(
      firstLayer.filter((f) => f.stage === "rmsnorm").map((f) => f.blockPart),
    ).toEqual(["attention", "ffn"]);
    expect(
      firstLayer.filter((f) => f.stage === "residual").map((f) => f.blockPart),
    ).toEqual(["attention", "ffn"]);
  });

  it("exposes selected expert weights and outputs that reconstruct the MoE weighted sum", () => {
    const trace = buildInferenceTrace({
      ...config,
      modelId: "qwen36-moe",
      outputLimit: 1,
    });
    const frame = trace.find((f) => f.stage === "moe")!;
    const selected = frame.tensors.find(
      (t) => t.name === "选中专家 [ID, 权重]",
    );
    expect(selected).toBeDefined();
    const experts = frame.tensors.find((t) => t.name === "选中专家输出")!;
    const routed = frame.tensors.find((t) => t.name === "路由加权和")!;
    const shared = frame.tensors.find((t) => t.name === "共享专家")!;
    const combined = frame.tensors
      .find((t) => t.name === "MoE 输出")!
      .values.slice(-4);
    const [firstWeight, secondWeight] = [
      selected!.values[1],
      selected!.values[3],
    ];
    expect(firstWeight + secondWeight).toBeCloseTo(1, 12);
    for (let j = 0; j < 4; j++) {
      expect(routed.values[j]).toBeCloseTo(
        firstWeight * experts.values[j] + secondWeight * experts.values[4 + j],
        12,
      );
      expect(combined[j]).toBeCloseTo(routed.values[j] + shared.values[j], 12);
    }
  });

  it("keeps causal convolution history and fixed-size linear states separate from KV", () => {
    const trace = buildInferenceTrace({ ...config, outputLimit: 2 });
    const linear = trace.find((f) => f.stage === "linear")!;
    const history = linear.tensors.find((t) => t.name === "卷积历史");
    expect(history).toBeDefined();
    expect(history!.rows).toBe(2);
    expect(history!.cols).toBe(8);
    expect(
      linear.tensors
        .filter((t) => t.name.startsWith("递归状态"))
        .map((t) => t.shape),
    ).toEqual(["2 × 2", "2 × 2"]);
    expect(linear.tensors.some((t) => t.name.startsWith("KV："))).toBe(false);
    const decode = trace.find(
      (f) => f.stage === "linear" && f.pass === "decode",
    )!;
    expect(decode.tensors.find((t) => t.name === "卷积历史")!.rows).toBe(2);
  });

  it("runs every real layer type in both model topologies", () => {
    for (const [modelId, layers, full, linear] of [
      ["qwen38-dense", 64, 16, 48],
      ["qwen36-moe", 40, 10, 30],
    ] as const) {
      const trace = buildInferenceTrace({ ...config, modelId, outputLimit: 1 });
      expect(
        new Set(trace.filter((f) => f.stage === "qkv").map((f) => f.layer))
          .size,
      ).toBe(layers);
      expect(trace.filter((f) => f.stage === "attention")).toHaveLength(full);
      expect(trace.filter((f) => f.stage === "linear")).toHaveLength(linear);
      expect(
        trace.filter(
          (f) => f.stage === (modelId === "qwen36-moe" ? "moe" : "ffn"),
        ),
      ).toHaveLength(layers);
    }
  });

  it("computes finite miniature tensors and RMS normalization rather than illustrative placeholders", () => {
    const trace = buildInferenceTrace({ ...config, outputLimit: 1 });
    for (const frame of trace)
      for (const tensor of frame.tensors) {
        expect(tensor.values).toHaveLength(tensor.rows * tensor.cols);
        expect(tensor.values.length).toBeGreaterThan(0);
        expect(tensor.values.every(Number.isFinite)).toBe(true);
        expect(tensor.shape).toBe(`${tensor.rows} × ${tensor.cols}`);
      }
    const norm = trace.find((f) => f.stage === "rmsnorm")!;
    const input = norm.tensors
      .find((t) => t.name === "输入 x")!
      .values.slice(-4);
    const output = norm.tensors
      .find((t) => t.name === "归一化 x̂")!
      .values.slice(-4);
    const denominator = Math.sqrt(
      input.reduce((sum, x) => sum + x * x, 0) / 4 + 1e-6,
    );
    output.forEach((value, i) =>
      expect(value).toBeCloseTo(input[i] / denominator, 10),
    );
    const attention = trace.find((f) => f.stage === "attention")!;
    const probabilities = attention.tensors.find(
      (t) => t.name === "注意力概率",
    )!;
    for (let row = 0; row < probabilities.rows; row++) {
      expect(
        probabilities.values
          .slice(row * probabilities.cols, (row + 1) * probabilities.cols)
          .reduce((a, b) => a + b, 0),
      ).toBeCloseTo(1, 10);
    }
  });

  it("puts a generated token into cache on the next decode pass and releases blocks after stop", () => {
    const trace = buildInferenceTrace(config);
    const sampling = samples(trace);
    expect(sampling).toHaveLength(3);
    expect(sampling.map((f) => request(f).cacheTokens)).toEqual([6, 7, 8]);
    expect(sampling.map((f) => request(f).outputTokens.length)).toEqual([
      0, 1, 2,
    ]);
    const emits = trace.filter((f) => f.stage === "emit");
    expect(emits.map((f) => request(f).outputTokens.length)).toEqual([1, 2, 3]);
    expect(emits.map((f) => request(f).cacheTokens)).toEqual([6, 7, 8]);
    const stopped = trace.find((f) => f.stage === "finish")!;
    expect(request(stopped).blocks.length).toBe(2);
    expect(request(stopped).status).toBe("done");
    expect(request(trace.at(-1)!).blocks).toEqual([]);
    expect(request(trace.at(-1)!).cacheTokens).toBe(0);
    expect(request(trace.at(-1)!).processedTokens).toBe(8);
    expect(request(trace.at(-1)!).outputTokens).toHaveLength(3);
  });

  it("fills a free batch slot with the queued arrival and keeps request computations isolated", () => {
    const trace = buildInferenceTrace({ ...config, mode: "batch" });
    const rounds = trace.filter((f) => f.stage === "schedule");
    expect(rounds[0].requestIds).toEqual(["R1", "R2"]);
    const firstR3 = rounds.find((f) => f.requestIds.includes("R3"))!;
    expect(firstR3.tick).toBe(2);
    expect(request(firstR3, "R2").status).toBe("done");
    for (const frame of trace) {
      expect(
        frame.requests.filter(
          (r) => r.status === "prefill" || r.status === "decode",
        ).length,
      ).toBeLessThanOrEqual(2);
      const allBlocks = frame.requests.flatMap((r) => r.blocks);
      expect(new Set(allBlocks).size).toBe(allBlocks.length);
    }
    for (const [id, scenarioId, outputLimit] of [
      ["R1", "sky", 3],
      ["R2", "code", 2],
      ["R3", "shared", 3],
    ] as const) {
      const solo = buildInferenceTrace({ ...config, scenarioId, outputLimit });
      expect(samples(trace, id).map((f) => f.probabilities)).toEqual(
        samples(solo).map((f) => f.probabilities),
      );
      expect(request(trace.at(-1)!, id).outputTokens).toEqual(
        request(solo.at(-1)!).outputTokens,
      );
      expect(request(trace.at(-1)!, id).blocks).toEqual([]);
    }
  });

  it("uses hidden-state logits, applies sampling controls, and replays frozen snapshots deterministically", () => {
    const trace = buildInferenceTrace(config);
    expect(buildInferenceTrace(config)).toEqual(trace);
    expect(Object.isFrozen(trace[0])).toBe(true);
    expect(Object.isFrozen(trace[0].requests[0].inputIds)).toBe(true);
    expect(request(trace[0]).outputTokens).toEqual([]);
    const logits = trace
      .find((f) => f.stage === "lm-head")!
      .tensors.find((t) => t.name === "Logits")!;
    expect(logits.values).not.toEqual([3.2, 2.4, 1.8, 1.1, 0.2, -0.7]);
    const argmax = buildInferenceTrace({ ...config, temperature: 0 });
    expect(samples(argmax)[0].probabilities.filter((p) => p > 0)).toHaveLength(
      1,
    );
    expect(samples(argmax)[0].probabilities).not.toEqual(
      samples(trace)[0].probabilities,
    );
    const topOne = buildInferenceTrace({ ...config, topK: 1 });
    expect(samples(topOne)[0].probabilities.filter((p) => p > 0)).toHaveLength(
      1,
    );
    const other = buildInferenceTrace({ ...config, scenarioId: "code" });
    expect(samples(other)[0].probabilities).not.toEqual(
      samples(trace)[0].probabilities,
    );
  });

  it.each([
    { capacity: 0 },
    { capacity: 4 },
    { capacity: 1.5 },
    { outputLimit: 0 },
    { outputLimit: 5 },
    { outputLimit: NaN },
    { temperature: Infinity },
    { temperature: -1 },
    { topK: 0 },
    { topP: 0 },
    { topP: 1.1 },
    { scenarioId: "missing" },
    { modelId: "missing" },
    { mode: "missing" },
  ])("rejects invalid configuration %j", (override) => {
    expect(() =>
      buildInferenceTrace({ ...config, ...override } as TraceConfig),
    ).toThrow();
  });
});

it("labels a packed prefill/decode batch as mixed", () => {
  const frames = buildInferenceTrace({ ...config, mode: "batch", capacity: 3 });
  const mixed = frames.find((f) => f.stage === "schedule" && f.tick === 1)!;
  expect(mixed.pass).toBe("mixed");
  expect(
    mixed.requests
      .filter((r) => mixed.requestIds.includes(r.id))
      .map((r) => r.status),
  ).toContain("prefill");
  expect(
    mixed.requests
      .filter((r) => mixed.requestIds.includes(r.id))
      .map((r) => r.status),
  ).toContain("decode");
});
