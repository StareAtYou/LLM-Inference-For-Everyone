import { softmax } from "./sampling";
export function causalAttention(scores: number[][]): number[][] {
  if (
    !scores.length ||
    scores.some(
      (row) => row.length !== scores.length || !row.every(Number.isFinite),
    )
  )
    throw new Error("注意力分数需为非空有限方阵");
  return scores.map((r, i) => [
    ...softmax(r.slice(0, i + 1), 1),
    ...Array(scores.length - i - 1).fill(0),
  ]);
}
