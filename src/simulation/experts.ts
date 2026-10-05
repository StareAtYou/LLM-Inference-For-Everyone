import { softmax } from "./sampling";
export function routeExperts(
  logits: number[],
  activeCount: number,
): { id: number; weight: number }[] {
  if (
    !Number.isInteger(activeCount) ||
    activeCount < 1 ||
    activeCount > logits.length ||
    !logits.every(Number.isFinite)
  )
    throw new Error("专家数量或路由分数无效");
  const top = logits
    .map((logit, id) => ({ logit, id }))
    .sort((a, b) => b.logit - a.logit || a.id - b.id)
    .slice(0, activeCount);
  const weights = softmax(
    top.map((x) => x.logit),
    1,
  );
  return top.map((x, i) => ({ id: x.id, weight: weights[i] }));
}
