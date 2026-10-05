import type { Depth, ModelId } from "../types";
import { isDepth } from "./preferences";
export const isModelId = (value: unknown): value is ModelId =>
  ["qwen38-dense", "qwen36-moe", "teaching"].includes(String(value));
export function parseRouteState(params: URLSearchParams): {
  depth: Depth;
  modelId: ModelId;
  warning?: string;
} {
  const ds = params.getAll("depth"),
    ms = params.getAll("model");
  const invalidDepth = ds.length > 1 || (ds.length === 1 && !isDepth(ds[0]));
  const invalidModel = ms.length > 1 || (ms.length === 1 && !isModelId(ms[0]));
  return {
    depth: !invalidDepth && isDepth(ds[0]) ? ds[0] : "beginner",
    modelId: !invalidModel && isModelId(ms[0]) ? ms[0] : "qwen38-dense",
    ...(invalidDepth || invalidModel
      ? { warning: "链接中的部分参数无效，已恢复为默认配置。" }
      : {}),
  };
}
type Rule = {
  default: number | string;
  min?: number;
  max?: number;
  integer?: boolean;
  choices?: (number | string)[];
};
const schemas: Record<string, Record<string, Rule>> = {
  "continuous-batching": {
    capacity: { default: 2, min: 1, max: 4, integer: true },
    scenario: { default: "mixed", choices: ["mixed", "equal", "staggered"] },
  },
  quantization: {
    bits: { default: 4, choices: [4, 8] },
    scenario: { default: "balanced", choices: ["balanced", "outlier", "zero"] },
  },
  "speculative-decoding": {
    draftLength: { default: 3, min: 1, max: 6, integer: true },
    draftCost: { default: 0.2, min: 0.1, max: 3 },
    verifyCost: { default: 2, min: 1, max: 5 },
    scenario: { default: "partial", choices: ["partial", "all", "low"] },
  },
  "kv-cache": {
    tokens: { default: 1024, min: 1, max: 262144, integer: true },
    batch: { default: 1, min: 1, max: 64, integer: true },
    bytes: { default: 2, choices: [0.5, 1, 2, 4] },
  },
  "paged-attention": {
    tokens: { default: 5, min: 1, max: 262144, integer: true },
    capacity: { default: 12, min: 1, max: 128, integer: true },
    blockSize: { default: 4, min: 1, max: 64, integer: true },
  },
};
export function parseExperimentParams(
  experiment: string,
  params: URLSearchParams,
): { values: Record<string, number | string>; warning?: string } {
  const values: Record<string, number | string> = {};
  let invalid = false;
  for (const [key, rule] of Object.entries(schemas[experiment] ?? {})) {
    const raw = params.getAll(key);
    let value: number | string = rule.default;
    if (raw.length) {
      value = typeof rule.default === "number" ? Number(raw[0]) : raw[0];
      const valid =
        raw.length === 1 &&
        raw[0].trim() !== "" &&
        (typeof value === "string" || Number.isFinite(value)) &&
        (!rule.integer || Number.isInteger(value)) &&
        (!rule.choices || rule.choices.includes(value)) &&
        (rule.min === undefined || Number(value) >= rule.min) &&
        (rule.max === undefined || Number(value) <= rule.max);
      if (!valid) {
        invalid = true;
        value = rule.default;
      }
    }
    values[key] = value;
  }
  return {
    values,
    warning: invalid ? "部分实验参数无效，已恢复为默认值。" : undefined,
  };
}
export function serializeExperimentParams(
  experiment: string,
  values: Record<string, number | string>,
): URLSearchParams {
  const draft = new URLSearchParams();
  for (const key of Object.keys(schemas[experiment] ?? {}))
    if (values[key] !== undefined) draft.set(key, String(values[key]));
  const checked = parseExperimentParams(experiment, draft);
  return new URLSearchParams(
    Object.entries(checked.values).map(([k, v]) => [k, String(v)]),
  );
}
