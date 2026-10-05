export type QuantizedVector = {
  scale: number;
  codes: number[];
  reconstructed: number[];
  meanSquaredError: number;
  payloadBytes: number;
};
export function quantizeSymmetric(
  weights: number[],
  bits: 4 | 8,
): QuantizedVector {
  if (
    !weights.length ||
    weights.length > 4096 ||
    !weights.every(Number.isFinite) ||
    ![4, 8].includes(bits)
  )
    throw new Error("量化输入必须是非空有限向量，位宽为 4 或 8");
  const bound = 2 ** (bits - 1) - 1;
  const max = Math.max(...weights.map(Math.abs));
  const scale = max === 0 ? 1 : max / bound;
  if (scale === 0) throw new Error("权重过小，无法用当前数值精度演示");
  const codes = weights.map(
    (w) => Math.max(-bound, Math.min(bound, Math.round(w / scale))) || 0,
  );
  const reconstructed = codes.map((c) => c * scale);
  const meanSquaredError = weights.reduce(
    (s, w, i) => s + (w - reconstructed[i]) ** 2 / weights.length,
    0,
  );
  if (!Number.isFinite(meanSquaredError))
    throw new Error("权重范围过大，误差估算超出数值精度");
  return {
    scale,
    codes,
    reconstructed,
    meanSquaredError,
    payloadBytes: Math.ceil((weights.length * bits) / 8),
  };
}
