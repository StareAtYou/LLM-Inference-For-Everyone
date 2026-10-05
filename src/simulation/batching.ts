export type Request = { id: string; arrival: number; outputTokens: number };
export type BatchResult = {
  frames: { tick: number; slots: (string | null)[] }[];
  completions: Record<string, number>;
  busySlots: number;
  totalSlots: number;
};
export function simulateBatching(
  requests: Request[],
  capacity: number,
  mode: "static" | "continuous",
): BatchResult {
  if (
    !Number.isInteger(capacity) ||
    capacity < 1 ||
    capacity > 16 ||
    requests.length > 64 ||
    !["static", "continuous"].includes(mode) ||
    new Set(requests.map((r) => r.id)).size !== requests.length ||
    requests.some(
      (r) =>
        !/^\w[\w-]{0,30}$/.test(r.id) ||
        ["__proto__", "constructor", "prototype"].includes(r.id) ||
        !Number.isInteger(r.arrival) ||
        r.arrival < 0 ||
        r.arrival > 256 ||
        !Number.isInteger(r.outputTokens) ||
        r.outputTokens < 1 ||
        r.outputTokens > 128,
    )
  )
    throw new Error("批处理请求或容量无效");
  const pending = requests
    .map((r, i) => ({ ...r, order: i }))
    .sort((a, b) => a.arrival - b.arrival || a.order - b.order);
  const slots: (Request | null)[] = Array(capacity).fill(null);
  const remaining = new Map(requests.map((r) => [r.id, r.outputTokens]));
  const result: BatchResult = {
    frames: [],
    completions: {},
    busySlots: 0,
    totalSlots: 0,
  };
  let tick = 0;
  while (pending.length || slots.some(Boolean)) {
    if (mode === "continuous" || !slots.some(Boolean)) {
      for (let i = 0; i < capacity; i++) {
        if (!slots[i] && pending[0]?.arrival <= tick)
          slots[i] = pending.shift()!;
      }
    }
    result.frames.push({ tick, slots: slots.map((r) => r?.id ?? null) });
    for (let i = 0; i < capacity; i++) {
      const req = slots[i];
      if (!req) continue;
      result.busySlots++;
      const left = remaining.get(req.id)! - 1;
      remaining.set(req.id, left);
      if (left === 0) {
        result.completions[req.id] = tick + 1;
        slots[i] = null;
      }
    }
    tick++;
  }
  result.totalSlots = result.frames.length * capacity;
  return result;
}
