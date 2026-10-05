import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { models } from "../../src/content/models";
import {
  buildInferenceTrace,
  type TraceConfig,
} from "../../src/simulation/inferenceTrace";
import { mechanisms, buildMechanismFrames } from "../../src/mechanisms/catalog";
import {
  deriveTraceShapes,
  deriveMechanismShapes,
  modelShapeSteps,
} from "../../src/education/shapeDerivations";
import TensorInspector from "../../src/visualizations/TensorInspector";
import MechanismPlayer from "../../src/mechanisms/MechanismPlayer";
const config: TraceConfig = {
  modelId: "qwen38-dense",
  scenarioId: "sky",
  mode: "batch",
  capacity: 3,
  temperature: 1,
  topK: 6,
  topP: 1,
  outputLimit: 3,
};
const object = (lesson: ReturnType<typeof deriveTraceShapes>, name: string) =>
  lesson.steps
    .flatMap((s) => [...s.inputs, ...s.outputs])
    .find((o) => o.name === name)!;
describe("shape teaching follows actual computation", () => {
  it.each([
    ["qwen38-dense", 24, 4, 5120, 12288],
    ["qwen36-moe", 16, 2, 2048, 8192],
  ] as const)(
    "derives gated full projections for %s",
    (id, qh, kh, h, gated) => {
      const model = models.find((m) => m.id === id)!;
      const frames = buildInferenceTrace({ ...config, modelId: id });
      const frame = frames.find(
        (f) => f.stage === "qkv" && f.layer === 3 && f.pass === "prefill",
      )!;
      const lesson = deriveTraceShapes(frame, "R1", model);
      expect(object(lesson, "W_Q+gate").dimensions).toEqual([h, gated]);
      expect(object(lesson, "Q").dimensions).toEqual([1, qh, 6, 256]);
      expect(object(lesson, "K 新位置").dimensions).toEqual([1, kh, 6, 256]);
      expect(lesson.queryLength).toBe(6);
      expect(lesson.packedTokens).toBe(10);
      expect(lesson.steps.some((s) => s.explanation.includes("收缩"))).toBe(
        true,
      );
    },
  );
  it("uses one decode query and includes current token in per-layer KV despite deferred counters", () => {
    const model = models[0],
      frames = buildInferenceTrace(config);
    const f = frames.find(
      (f) => f.stage === "attention" && f.pass === "mixed",
    )!;
    const lesson = deriveTraceShapes(f, "R1", model);
    expect(lesson.queryLength).toBe(1);
    expect(lesson.keyLength).toBe(7);
    expect(object(lesson, "分数 / 概率").dimensions).toEqual([1, 24, 1, 7]);
    expect(object(lesson, "K 缓存").dimensions).toEqual([1, 4, 7, 256]);
  });
  it("packs ragged mixed prefill/decode without cross-request attention", () => {
    const f = buildInferenceTrace(config).find(
      (f) => f.stage === "embedding" && f.pass === "mixed",
    )!;
    const lesson = deriveTraceShapes(f, "R3", models[0]);
    expect(lesson.batchRequests.map((r) => r.queryLength)).toEqual([1, 1, 8]);
    expect(lesson.packedTokens).toBe(10);
    expect(lesson.queryLength).toBe(8);
    expect(lesson.notes.join(" ")).toContain("不跨请求");
  });
  it.each([
    ["qwen38-dense", 48, 10240, 17408],
    ["qwen36-moe", 32, 8192, 512],
  ] as const)(
    "derives linear state and FFN intermediate widths for %s",
    (id, hv, width, ffn) => {
      const model = models.find((m) => m.id === id)!;
      const frames = buildInferenceTrace({ ...config, modelId: id });
      const linear = deriveTraceShapes(
        frames.find((f) => f.stage === "linear")!,
        "R1",
        model,
      );
      expect(object(linear, "S 固定状态").dimensions).toEqual([
        1,
        hv,
        128,
        128,
      ]);
      expect(object(linear, "线性 QKV").dimensions).toEqual([1, 6, width]);
      expect(object(linear, "输出门 z").dimensions).toEqual([1, 6, hv, 128]);
      const lesson = deriveTraceShapes(
        frames.find((f) => f.stage === (id === "qwen36-moe" ? "moe" : "ffn"))!,
        "R1",
        model,
      );
      expect(object(lesson, "W_gate / W_up").dimensions).toEqual([
        model.hiddenSize,
        ffn,
      ]);
      if (id === "qwen36-moe") {
        expect(object(lesson, "路由 logits").dimensions).toEqual([6, 256]);
        expect(object(lesson, "路由 logits").axes).toEqual(["q", "E"]);
        expect(object(lesson, "top-k ID / 权重").dimensions).toEqual([6, 8]);
        expect(object(lesson, "专家 e 输入").dimensions).toEqual(["Nₑ", 2048]);
        expect(object(lesson, "共享门").dimensions).toEqual([6, 1]);
      }
    },
  );
  it("has stage-specific derivations and truthful current object dimensions across all 29 mechanisms and parameter boundaries", () => {
    expect(mechanisms).toHaveLength(29);
    for (const definition of mechanisms) {
      const paramsList = [
        {},
        ...definition.parameters.flatMap((p) => [
          { [p.key]: p.min },
          { [p.key]: p.max },
        ]),
      ];
      for (const params of paramsList) {
        const frames = buildMechanismFrames(definition.id, params);
        const lessons = frames.map((f, i) =>
          deriveMechanismShapes(definition.id, i, params),
        );
        expect(new Set(lessons.map((l) => l.stage)).size).toBe(frames.length);
        lessons.forEach((lesson, i) => {
          expect(lesson.stage).toBe(frames[i].stage);
          expect(lesson.steps.length).toBeGreaterThan(0);
          expect(
            lesson.steps.every(
              (s) =>
                s.inputs.length > 0 &&
                s.outputs.length > 0 &&
                s.explanation.length > 10 &&
                s.substitution.length > 0,
            ),
          ).toBe(true);
          expect(lesson.currentObjects).toHaveLength(frames[i].panels.length);
          lesson.currentObjects.forEach((o, j) => {
            const panel = frames[i].panels[j];
            if (panel.matrix)
              expect(o.dimensions).toEqual([
                panel.matrix.length,
                panel.matrix[0].length,
              ]);
            else if (panel.values)
              expect(o.dimensions).toEqual([panel.values.length]);
            else {
              expect(o.kind).toBe("structure");
              expect(o.count).toBe(panel.items?.length ?? 0);
            }
          });
        });
      }
    }
  });
  it("makes all linear gate weight widths explicit and keeps each per-head scalar gate separate", () => {
    const dense = modelShapeSteps("linear", models[0], 1, 9, true);
    const objects = dense.flatMap((step) => [...step.inputs, ...step.outputs]);
    expect(objects.find((o) => o.name === "W_z")?.dimensions).toEqual([
      5120, 6144,
    ]);
    expect(objects.find((o) => o.name === "W_a / W_b")?.dimensions).toEqual([
      5120, 48,
    ]);
    expect(objects.find((o) => o.name === "α / β 门")?.dimensions).toEqual([
      1, 1, 48,
    ]);
  });
  it("uses catalog defaults and current chunk budgets for engineering objects", () => {
    const pages = deriveMechanismShapes("paged-attention", 1);
    expect(
      pages.steps[0].outputs.find((o) => o.name === "分配物理块")?.count,
    ).toBe(4);
    const chunks = deriveMechanismShapes("chunked-prefill", 2, { chunk: 3 });
    expect(chunks.steps[0].outputs[0].count).toBe(4);
    const tail = deriveMechanismShapes("chunked-prefill", 3, { chunk: 2 });
    expect(tail.steps[0].outputs[0].count).toBe(3);
  });
  it("does not invent network query context during request lifecycle stages", () => {
    const stages = new Set([
      "receive",
      "tokenize",
      "schedule",
      "emit",
      "feedback",
      "finish",
      "release",
    ]);
    const frames = buildInferenceTrace(config).filter((f) =>
      stages.has(f.stage),
    );
    expect(new Set(frames.map((f) => f.stage)).size).toBe(stages.size);
    for (const frame of frames) {
      const lesson = deriveTraceShapes(
        frame,
        frame.requestIds[0] ?? "R1",
        models[0],
      );
      expect(lesson.batchRequests).toEqual([]);
      expect([
        lesson.queryLength,
        lesson.keyLength,
        lesson.packedTokens,
      ]).toEqual([0, 0, 0]);
      expect(
        lesson.axes.some((a) => a.symbol === "t" || a.symbol === "N"),
      ).toBe(false);
      const html = renderToStaticMarkup(
        createElement(TensorInspector, {
          frame,
          requestId: frame.requestIds[0] ?? "R1",
          model: models[0],
        }),
      );
      expect(html).not.toContain('class="shape-context"');
    }
  });
  it("distinguishes ID sequence axes and scheduler metadata from activation channels", () => {
    const frames = buildInferenceTrace(config);
    for (const [stage, name, dimensions] of [
      ["tokenize", "输入 Token IDs", [1, 6]],
      ["feedback", "待反馈 ID", [1, 1]],
    ] as const) {
      const frame = frames.find(
        (f) => f.stage === stage && f.requestIds.includes("R1"),
      )!;
      const ids = deriveTraceShapes(frame, "R1", models[0]).currentObjects.find(
        (o) => o.name === name,
      )!;
      expect(ids.axes).toEqual(["b", "q"]);
      expect(ids.dimensions).toEqual(dimensions);
    }
    const frame = frames.find((f) => f.stage === "schedule")!;
    const batch = deriveTraceShapes(frame, "R1", models[0]).currentObjects.find(
      (o) => o.name.includes("批次"),
    )!;
    expect(batch.kind).toBe("structure");
    expect(batch.dimensions).toEqual([2, 2]);
    expect(batch.count).toBe(2);
    expect(batch.detail).toContain("请求");
  });
  it.each(["qwen38-dense", "qwen36-moe"] as const)(
    "accounts for every K/V input and preserves element counts when splitting heads for %s",
    (id) => {
      const steps = modelShapeSteps(
        "qkv",
        models.find((m) => m.id === id)!,
        6,
        6,
        false,
      );
      const projection = steps[0],
        reshape = steps[1];
      expect(reshape.inputs.map((o) => o.name)).toEqual(
        projection.outputs.map((o) => o.name),
      );
      expect(reshape.inputs).toHaveLength(3);
      const elements = (objects: typeof reshape.inputs) =>
        objects.reduce(
          (sum, o) =>
            sum + o.dimensions.reduce<number>((n, d) => n * Number(d), 1),
          0,
        );
      expect(elements(reshape.inputs)).toBe(elements(reshape.outputs));
      const merge = modelShapeSteps("moe", models[1], 6, 6, false).at(-1)!;
      expect(
        merge.inputs.find((o) => o.name === "共享分支 X")?.dimensions,
      ).toEqual([6, 2048]);
      expect(
        merge.inputs.find((o) => o.name === "SharedFFN 输出")?.dimensions,
      ).toEqual([6, 2048]);
    },
  );
  it("uses semantic history, vocabulary, scalar, expert and projection axes", () => {
    const axes = (id: string, index: number, name: string) =>
      object(deriveMechanismShapes(id, index), name).axes;
    expect(axes("flash-attention", 0, "分数行")).toEqual(["t"]);
    expect(axes("flash-attention", 0, "V")).toEqual(["t"]);
    expect(axes("flash-attention", 1, "V tile")).toEqual(["t_tile"]);
    expect(axes("flash-attention", 3, "分母 l")).toEqual(["标量"]);
    expect(axes("flash-attention", 3, "输出")).toEqual(["dv"]);
    expect(axes("softmax", 2, "概率")).toEqual(["V"]);
    expect(axes("softmax", 2, "归约分母")).toEqual(["标量"]);
    expect(axes("ffn", 0, "W_up / W_gate")).toEqual(["H", "I"]);
    expect(axes("ffn", 3, "W_down")).toEqual(["I", "H"]);
    expect(axes("attention", 0, "Kᵀ")).toEqual(["d", "t"]);
    expect(axes("attention", 3, "当前查询概率")).toEqual(["q", "t"]);
    expect(axes("gated-deltanet", 0, "k / q")).toEqual(["dk"]);
    expect(axes("moe", 2, "选中专家输出")).toEqual(["k", "H"]);
    expect(axes("embedding", 0, "E")).toEqual(["V", "H"]);
    expect(axes("parallelism", 2, "rank 部分结果")).toEqual(["rank", "F"]);
    expect(axes("parallelism", 0, "本 rank W")).toEqual(["H_local", "F"]);
    const displayed = (id: string, i: number, name: string) =>
      deriveMechanismShapes(id, i).currentObjects.find((o) => o.name === name)
        ?.axes;
    expect(displayed("attention", 2, "选中行概率")).toEqual(["t"]);
    expect(displayed("attention", 3, "当前查询的输出")).toEqual(["d"]);
    expect(displayed("prefill", 1, "各位置可见性")).toEqual(["q", "t"]);
    expect(displayed("decode", 1, "当前 Q")).toEqual(["d"]);
    expect(displayed("decode", 2, "这一行注意力")).toEqual(["t"]);
    expect(displayed("lm-head", 0, "最后位置 h")).toEqual(["H"]);
    expect(displayed("lm-head", 1, "候选 2 的贡献")).toEqual(["H"]);
    expect(displayed("attention-output", 2, "输出投影结果")).toEqual(["H"]);
    const flash = deriveMechanismShapes("flash-attention", 1);
    expect(
      flash.currentObjects.find((o) => o.name === "当前 V tile")?.axes,
    ).toEqual(["t_tile"]);
    expect(
      flash.currentObjects.find((o) => o.name === "在线累积 m / l / o")?.kind,
    ).toBe("structure");
    for (const m of mechanisms)
      for (const [i] of buildMechanismFrames(m.id).entries()) {
        const lesson = deriveMechanismShapes(m.id, i);
        for (const o of [
          ...lesson.currentObjects,
          ...lesson.steps.flatMap((s) => [...s.inputs, ...s.outputs]),
        ]) {
          if (o.kind === "tensor")
            expect(o.axes.length).toBe(o.dimensions.length);
        }
      }
  });
  it("rejects nonexistent stages rather than inventing a shape", () => {
    expect(() => deriveMechanismShapes("missing", 0)).toThrow();
    expect(() => deriveMechanismShapes("attention", 99)).toThrow();
  });
  it("react rendering exposes the current derivation on both inspectors", () => {
    const frame = buildInferenceTrace(config).find(
      (f) => f.stage === "attention",
    )!;
    const html = renderToStaticMarkup(
      createElement(TensorInspector, {
        frame,
        requestId: "R1",
        model: models[0],
      }),
    );
    expect(html).toContain('data-testid="shape-teaching"');
    expect(html).toContain("形状是怎样推出来的");
    for (const mechanism of mechanisms) {
      const html = renderToStaticMarkup(
        createElement(MechanismPlayer, { id: mechanism.id }),
      );
      expect(html).toContain('data-testid="shape-teaching"');
    }
  });
});
