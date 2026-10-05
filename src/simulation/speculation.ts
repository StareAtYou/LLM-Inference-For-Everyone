export type SpeculationResult = {
  rounds: {
    draft: string[];
    accepted: string[];
    rejected: string[];
    correction?: string;
  }[];
  output: string[];
  baselineCost: number;
  speculativeCost: number;
};
export function simulateSpeculation(
  targetTokens: string[],
  draftTokens: string[],
  draftLength: number,
  costs: { draft: number; verify: number },
): SpeculationResult {
  if (
    !Number.isInteger(draftLength) ||
    draftLength < 1 ||
    draftLength > 16 ||
    targetTokens.length > 256 ||
    draftTokens.length !== targetTokens.length ||
    [...targetTokens, ...draftTokens].some(
      (t) => typeof t !== "string" || !t.length || t.length > 100,
    ) ||
    [costs.draft, costs.verify].some(
      (c) => !Number.isFinite(c) || c <= 0 || c > 1000,
    )
  )
    throw new Error("草稿、长度或成本无效");
  const result: SpeculationResult = {
    rounds: [],
    output: [],
    baselineCost: targetTokens.length * costs.verify,
    speculativeCost: 0,
  };
  while (result.output.length < targetTokens.length) {
    const start = result.output.length;
    const draft = draftTokens.slice(start, start + draftLength);
    const accepted: string[] = [];
    let i = 0;
    while (i < draft.length && draft[i] === targetTokens[start + i])
      accepted.push(draft[i++]);
    const rejected = draft.slice(i);
    const correction = targetTokens[start + i];
    result.rounds.push({ draft, accepted, rejected, correction });
    result.output.push(...accepted);
    if (correction !== undefined) result.output.push(correction);
    result.speculativeCost += draft.length * costs.draft + costs.verify;
  }
  return result;
}
