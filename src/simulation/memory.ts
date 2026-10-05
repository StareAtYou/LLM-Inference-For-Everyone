import type { ModelPreset } from "../types";
export function estimateFullAttentionKvBytes(
  model: ModelPreset,
  tokens: number,
  batch: number,
  bytesPerElement: number,
): number {
  if (
    !Number.isInteger(tokens) ||
    tokens < 1 ||
    tokens > 262144 ||
    !Number.isInteger(batch) ||
    batch < 1 ||
    batch > 64 ||
    ![0.5, 1, 2, 4].includes(bytesPerElement)
  )
    throw new Error("缓存参数超出教学范围");
  return (
    2 *
    model.layerTypes.filter((t) => t === "full_attention").length *
    tokens *
    batch *
    model.kvHeads *
    model.headDim *
    bytesPerElement
  );
}
export const formatBytes = (bytes: number) =>
  bytes >= 2 ** 30
    ? (bytes / 2 ** 30).toFixed(2) + " GiB"
    : bytes >= 2 ** 20
      ? (bytes / 2 ** 20).toFixed(2) + " MiB"
      : (bytes / 1024).toFixed(2) + " KiB";
