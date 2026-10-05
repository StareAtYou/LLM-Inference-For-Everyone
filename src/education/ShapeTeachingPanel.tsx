import { useState } from "react";
import { models } from "../content/models";
import type { TraceStage } from "../simulation/inferenceTrace";
import {
  formatShape,
  modelShapeSteps,
  type ShapeLesson,
  type ShapeObject,
} from "./shapeDerivations";
import "./shapes.css";

function ObjectList({ objects }: { objects: ShapeObject[] }) {
  return (
    <ul className="shape-object-list">
      {objects.map((object, i) => (
        <li key={object.name + i}>
          <span>{object.name}</span>
          <code>{formatShape(object)}</code>
          <small>
            {object.kind === "tensor" ? object.axes.join(" × ") : "结构对象"}
            {object.detail ? ` · ${object.detail}` : ""}
          </small>
        </li>
      ))}
    </ul>
  );
}
export default function ShapeTeachingPanel({
  lesson,
  showCurrentObjects = true,
}: {
  lesson: ShapeLesson;
  showCurrentObjects?: boolean;
}) {
  return (
    <section
      className="shape-teaching"
      data-testid="shape-teaching"
      data-shape-stage={lesson.stage}
      aria-label="当前阶段形状推导"
    >
      <div className="shape-kicker">INPUT → OPERATION → OUTPUT</div>
      <h3>形状是怎样推出来的？</h3>
      <p className="shape-scope">
        {lesson.scope} · {lesson.stage}
      </p>
      {lesson.batchRequests.length > 0 && (
        <div className="shape-context">
          <span>
            本轮 q=<strong>{lesson.queryLength}</strong>
          </span>
          <span>
            Full KV t=<strong>{lesson.keyLength}</strong>
          </span>
          <span>
            整批 N=<strong>{lesson.packedTokens}</strong>
          </span>
        </div>
      )}
      {lesson.steps.map((step, i) => (
        <details className="shape-step" key={step.title + i} open={i === 0}>
          <summary>
            <span>{String(i + 1).padStart(2, "0")}</span>
            {step.title}
          </summary>
          <div className="shape-step-body">
            <div className="shape-flow">
              <div>
                <h4>输入</h4>
                <ObjectList objects={step.inputs} />
              </div>
              <span className="shape-flow-arrow" aria-hidden="true">
                →
              </span>
              <div>
                <h4>输出 / 中间对象</h4>
                <ObjectList objects={step.outputs} />
              </div>
            </div>
            <div
              className="shape-formula"
              tabIndex={0}
              aria-label="形状推导公式"
            >
              <code>{step.formula}</code>
            </div>
            <p>{step.explanation}</p>
            <p className="shape-substitution">
              <span>数值代入</span>
              {step.substitution}
            </p>
          </div>
        </details>
      ))}
      <details className="shape-axis-details">
        <summary>轴图例与当前批次</summary>
        <dl className="shape-axes">
          {lesson.axes.map((axis) => (
            <div key={axis.symbol}>
              <dt>
                {axis.symbol}
                {axis.value !== undefined ? ` = ${axis.value}` : ""}
              </dt>
              <dd>{axis.meaning}</dd>
            </div>
          ))}
        </dl>
        {lesson.batchRequests.length > 1 && (
          <ul className="shape-batch">
            {lesson.batchRequests.map((r) => (
              <li key={r.id}>
                {r.id}：q={r.queryLength}，历史={r.historyLength}
              </li>
            ))}
          </ul>
        )}
      </details>
      {showCurrentObjects && lesson.currentObjects.length > 0 && (
        <details className="shape-axis-details">
          <summary>当前教学读数的实际形状</summary>
          <ObjectList objects={lesson.currentObjects} />
        </details>
      )}
      <details className="shape-axis-details">
        <summary>维度约定与教学边界</summary>
        <ul className="shape-notes">
          {lesson.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
        {lesson.sources && (
          <p className="shape-sources">
            依据：
            {lesson.sources.map((source, i) => (
              <a
                key={source.url}
                href={source.url}
                target="_blank"
                rel="noreferrer"
              >
                {i > 0 ? " · " : ""}
                {source.title} ↗
              </a>
            ))}
          </p>
        )}
      </details>
    </section>
  );
}
const modelStages: Record<string, TraceStage> = {
  embedding: "embedding",
  rmsnorm: "rmsnorm",
  qkv: "qkv",
  rope: "rope",
  attention: "attention",
  gqa: "attention",
  "attention-output": "attention-output",
  residual: "residual",
  ffn: "ffn",
  moe: "moe",
  "gated-deltanet": "linear",
  prefill: "attention",
  decode: "attention",
  "lm-head": "lm-head",
  sampling: "sample",
  softmax: "sample",
};
export function MechanismModelComparison({
  id,
  params,
}: {
  id: string;
  params: Record<string, number>;
}) {
  const [modelId, setModelId] = useState(
    id === "moe" ? models[1].id : models[0].id,
  );
  const stage = modelStages[id];
  if (!stage) return null;
  const model = models.find((m) => m.id === modelId)!;
  const q = id === "prefill" ? (params.tokens ?? 4) : 1,
    t = id === "decode" ? 4 : q;
  const incompatible = id === "moe" && model.family !== "moe";
  const linear = id === "gated-deltanet";
  const steps = incompatible ? [] : modelShapeSteps(stage, model, q, t, linear);
  const lesson: ShapeLesson = {
    stage: "该模块完整逻辑通路",
    scope: model.name,
    axes: [],
    steps,
    notes: [
      "模型对照使用配置尺寸；不对大矩阵进行计算。当前原理小数组与完整模型门控 / 投影可不同。",
      "示例 b=1；以本原理的位置参数代入，未指定位置长度的算子以 q=1 展示。",
      "Full 的 Q 投影包含独立输出门，因此投影输出是 2×nQ×d；输出宽度 nQ×d 无须等于 H。",
      "线性模型输出门采用 SiLU；Full 激活请核对对应实现，dense 配置指定 swish 而基础 Qwen3.5 实现使用 sigmoid。",
    ],
    currentObjects: [],
    queryLength: q,
    keyLength: t,
    packedTokens: q,
    batchRequests: [],
    sources: [
      ...model.sources,
      {
        title: "Qwen3.5 / MoE 实现（核查 2026-10-05）",
        url: "https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5_moe",
      },
    ],
  };
  return (
    <details className="shape-model-comparison">
      <summary>对照 Qwen 模型的逻辑维度</summary>
      <label className="shape-model-select">
        模型
        <select
          aria-label="形状对照模型"
          value={modelId}
          onChange={(e) => setModelId(e.target.value as typeof modelId)}
        >
          {models
            .filter((m) => m.id !== "teaching")
            .map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
        </select>
      </label>
      {incompatible ? (
        <p>
          所选模型使用 Dense FFN，没有路由专家。选择 MoE
          模型可查看真实专家维度。
        </p>
      ) : (
        <ShapeTeachingPanel lesson={lesson} showCurrentObjects={false} />
      )}
    </details>
  );
}
