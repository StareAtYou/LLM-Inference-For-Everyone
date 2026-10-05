import type { TraceFrame, TraceStage } from "../simulation/inferenceTrace";

export type JourneyPoint = { x: number; y: number };
export type JourneyTrack = {
  id: string;
  points: JourneyPoint[];
  tangents: JourneyPoint[];
  channelValues: number[][];
  carriedTokens: (string | null)[];
};
export const journeyNodes: {
  stage: TraceStage;
  part?: "ffn";
  x: number;
  y: number;
  title: string;
  module: string;
}[] = [
  { stage: "receive", x: 60, y: 92, title: "文本请求", module: "frameworks" },
  {
    stage: "tokenize",
    x: 65,
    y: 190,
    title: "Token IDs",
    module: "tokenization",
  },
  { stage: "schedule", x: 110, y: 305, title: "调度入槽", module: "batching" },
  {
    stage: "embedding",
    x: 200,
    y: 178,
    title: "Embedding",
    module: "embedding",
  },
  { stage: "rmsnorm", x: 310, y: 170, title: "RMSNorm", module: "rmsnorm" },
  { stage: "qkv", x: 420, y: 130, title: "Q / K / V", module: "qkv" },
  { stage: "rope", x: 540, y: 130, title: "RoPE", module: "rope" },
  {
    stage: "attention",
    x: 645,
    y: 175,
    title: "Full Attention",
    module: "attention",
  },
  {
    stage: "linear",
    x: 540,
    y: 195,
    title: "DeltaNet",
    module: "gated-deltanet",
  },
  {
    stage: "attention-output",
    x: 645,
    y: 265,
    title: "输出投影",
    module: "attention-output",
  },
  { stage: "residual", x: 565, y: 325, title: "+ 残差", module: "residual" },
  {
    stage: "rmsnorm",
    part: "ffn",
    x: 455,
    y: 325,
    title: "RMSNorm",
    module: "rmsnorm",
  },
  { stage: "ffn", x: 355, y: 325, title: "Dense FFN", module: "ffn" },
  { stage: "moe", x: 355, y: 325, title: "MoE 专家", module: "moe" },
  {
    stage: "residual",
    part: "ffn",
    x: 290,
    y: 265,
    title: "+ 残差",
    module: "residual",
  },
  {
    stage: "final-norm",
    x: 720,
    y: 265,
    title: "Final Norm",
    module: "rmsnorm",
  },
  { stage: "lm-head", x: 810, y: 265, title: "LM Head", module: "lm-head" },
  { stage: "sample", x: 865, y: 340, title: "采样", module: "sampling" },
  { stage: "emit", x: 775, y: 425, title: "新 Token", module: "decode" },
  { stage: "feedback", x: 180, y: 455, title: "回送下一轮", module: "decode" },
  { stage: "finish", x: 870, y: 440, title: "结束", module: "decode" },
  { stage: "release", x: 895, y: 455, title: "释放", module: "kv-cache" },
];
export function journeyNode(frame: TraceFrame) {
  return (
    journeyNodes.find(
      (n) =>
        n.stage === frame.stage &&
        (n.part ?? "attention") === (frame.blockPart ?? "attention"),
    ) ?? journeyNodes[0]
  );
}
function tangents(points: JourneyPoint[]) {
  return points.map((point, i) => {
    const before = points[Math.max(0, i - 1)],
      after = points[Math.min(points.length - 1, i + 1)];
    // Hold a parked request still. Otherwise use the same bounded tangent on
    // both sides of an event, so neither position nor velocity resets there.
    if (
      Math.hypot(point.x - before.x, point.y - before.y) < 0.001 ||
      Math.hypot(after.x - point.x, after.y - point.y) < 0.001
    )
      return { x: 0, y: 0 };
    return {
      x: Math.max(-150, Math.min(150, (after.x - before.x) / 2)),
      y: Math.max(-70, Math.min(70, (after.y - before.y) / 2)),
    };
  });
}
export function buildJourneyTracks(
  trace: readonly TraceFrame[],
): JourneyTrack[] {
  return (trace[0]?.requests ?? []).map((request, lane) => {
    let last: JourneyPoint = { x: 60, y: 92 + lane * 14 };
    let values = [0.3, 0.7, 0.45, 0.8];
    const channelValues: number[][] = [];
    let token: string | null = null;
    const carriedTokens: (string | null)[] = [];
    const points = trace.map((frame) => {
      if (frame.requestIds.includes(request.id)) {
        if (frame.stage === "embedding") token = null;
        if (frame.stage === "emit")
          token =
            frame.requests
              .find((r) => r.id === request.id)!
              .outputTokens.at(-1) ?? null;
        const node = journeyNode(frame);
        last = { x: node.x, y: node.y + lane * 14 };
        const tensor = frame.tensors
          .filter((t) => t.requestId === request.id)
          .at(-1);
        if (tensor) values = tensor.values.slice(-tensor.cols).slice(0, 6);
      }
      channelValues.push(values);
      carriedTokens.push(token);
      return last;
    });
    return {
      id: request.id,
      points,
      tangents: tangents(points),
      channelValues,
      carriedTokens,
    };
  });
}
export function journeyPose(
  track: JourneyTrack,
  position: number,
): JourneyPoint {
  const bounded = Math.max(
    0,
    Math.min(track.points.length - 1, Number.isFinite(position) ? position : 0),
  );
  const i = Math.floor(bounded),
    t = bounded - i,
    j = Math.min(i + 1, track.points.length - 1);
  const a = track.points[i],
    b = track.points[j],
    u = track.tangents[i],
    v = track.tangents[j];
  const h00 = 2 * t * t * t - 3 * t * t + 1,
    h10 = t * t * t - 2 * t * t + t,
    h01 = -2 * t * t * t + 3 * t * t,
    h11 = t * t * t - t * t;
  return {
    x: h00 * a.x + h10 * u.x + h01 * b.x + h11 * v.x,
    y: h00 * a.y + h10 * u.y + h01 * b.y + h11 * v.y,
  };
}
export function journeySegment(
  track: JourneyTrack,
  first: number,
  last: number,
) {
  const p = track.points[first];
  let path = `M${p.x} ${p.y}`;
  for (let i = first; i < last; i++) {
    const a = track.points[i],
      b = track.points[i + 1],
      u = track.tangents[i],
      v = track.tangents[i + 1];
    path += ` C${a.x + u.x / 3} ${a.y + u.y / 3},${b.x - v.x / 3} ${b.y - v.y / 3},${b.x} ${b.y}`;
  }
  return path;
}
export function journeyTrail(track: JourneyTrack, position: number) {
  return Array.from({ length: 14 }, (_, i) =>
    journeyPose(track, Math.max(0, position - (13 - i) * 0.045)),
  )
    .map((p, i) => `${i ? "L" : "M"}${p.x} ${p.y}`)
    .join(" ");
}
export function nodeActivity(
  node: (typeof journeyNodes)[number],
  current: TraceFrame,
  next: TraceFrame,
  fraction: number,
) {
  const match = (f: TraceFrame) =>
    f.stage === node.stage &&
    (node.part ?? "attention") === (f.blockPart ?? "attention");
  const t = fraction * fraction * (3 - 2 * fraction);
  return (match(current) ? 1 - t : 0) + (match(next) ? t : 0);
}
