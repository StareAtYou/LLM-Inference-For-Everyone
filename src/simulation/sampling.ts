export function softmax(logits: number[], temperature: number): number[] {
  if (
    !logits.length ||
    !logits.every(Number.isFinite) ||
    !Number.isFinite(temperature) ||
    temperature < 0
  )
    throw new Error("需要有限 logits 与非负温度");
  const max = Math.max(...logits);
  if (temperature === 0)
    return logits.map((_, i) => (i === logits.indexOf(max) ? 1 : 0));
  const exp = logits.map((v) => Math.exp((v - max) / temperature));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}
export function filterDistribution(
  probs: number[],
  topK: number,
  topP: number,
): number[] {
  if (
    !probs.length ||
    probs.some((p) => !Number.isFinite(p) || p < 0) ||
    !Number.isInteger(topK) ||
    topK < 1 ||
    !Number.isFinite(topP) ||
    topP <= 0 ||
    topP > 1
  )
    throw new Error("概率与采样参数无效");
  const sorted = probs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p || a.i - b.i)
    .slice(0, topK);
  const mass = sorted.reduce((s, x) => s + x.p, 0);
  if (mass <= 0) throw new Error("概率质量必须大于零");
  let sum = 0;
  const kept = new Set<number>();
  for (const x of sorted) {
    if (x.p > 0) {
      kept.add(x.i);
      sum += x.p / mass;
    }
    if (sum >= topP - 1e-12) break;
  }
  const total = probs.reduce((s, p, i) => s + (kept.has(i) ? p : 0), 0);
  return probs.map((p, i) => (kept.has(i) ? p / total : 0));
}
