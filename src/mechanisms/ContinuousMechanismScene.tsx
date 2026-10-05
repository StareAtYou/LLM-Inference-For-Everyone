import type {
  MechanismDefinition,
  MechanismFrame,
  MechanismPanel,
} from "./catalog";
import "./continuous.css";

/** Whole-path percentage, independent of the fraction within a single stage. */
export function continuousProgress(
  position: number,
  frameCount: number,
): number {
  if (!Number.isFinite(position)) return 0;
  return frameCount > 1
    ? Math.max(0, Math.min(1, position / (frameCount - 1)))
    : 1;
}

const compactNumber = (value: number) =>
  Number.isInteger(value) ? String(value) : Number(value.toFixed(3)).toString();

/** Geometry visualizes transport only; operands always come directly from catalog snapshots. */
function PanelGlyph({ panel }: { panel: MechanismPanel }) {
  const values = panel.values ?? [];
  const rows = panel.matrix ?? [];
  const bound = Math.max(
    1,
    ...values.map(Math.abs),
    ...rows.flat().map(Math.abs),
  );
  const selected = (index: number) => panel.selected?.includes(index) ?? false;
  const numeric = [...values, ...rows.flat()];
  const summary = numeric.length
    ? numeric.map(compactNumber).join(", ")
    : (panel.items ?? []).join(" · ");
  const columns = Math.max(1, rows[0]?.length ?? 1);
  return (
    <div
      className={`cm-glyph cm-glyph-${panel.kind}`}
      data-glyph-kind={panel.kind}
    >
      <span className="cm-glyph-label" title={panel.label}>
        {panel.label}
      </span>
      <svg
        viewBox="0 0 120 56"
        role="img"
        aria-label={`${panel.label}：${summary}`}
        data-values={JSON.stringify(numeric)}
        data-rows={rows.length}
        data-columns={columns}
      >
        <title>{`${panel.label}：${summary}${panel.note ? `。${panel.note}` : ""}`}</title>
        {panel.kind === "matrix" &&
          rows.flatMap((row, r) =>
            row.map((value, c) => {
              const masked = Boolean(panel.mask && c > r);
              const cellW = 108 / columns,
                cellH = 46 / Math.max(1, rows.length);
              return (
                <g key={`${r}-${c}`} data-value={value} data-masked={masked}>
                  <rect
                    x={6 + c * cellW}
                    y={5 + r * cellH}
                    width={Math.max(1, cellW - 3)}
                    height={Math.max(1, cellH - 3)}
                    rx="2"
                    className={
                      masked
                        ? "cm-masked"
                        : selected(r * columns + c)
                          ? "cm-selected"
                          : "cm-data"
                    }
                    opacity={
                      masked ? 0.45 : 0.35 + (Math.abs(value) / bound) * 0.65
                    }
                  />
                  {masked && (
                    <path
                      d={`M${8 + c * cellW} ${7 + r * cellH}l${Math.max(2, cellW - 8)} ${Math.max(2, cellH - 8)}`}
                      className="cm-mask-line"
                    />
                  )}
                </g>
              );
            }),
          )}
        {panel.kind === "vector" &&
          values.map((value, i) => {
            const width = 108 / Math.max(1, values.length);
            return (
              <g key={i} data-value={value}>
                <rect
                  x={6 + i * width}
                  y="10"
                  width={Math.max(1, width - 3)}
                  height="34"
                  rx="3"
                  className={
                    selected(i)
                      ? "cm-selected"
                      : value < 0
                        ? "cm-negative"
                        : "cm-data"
                  }
                  opacity={0.35 + (Math.abs(value) / bound) * 0.65}
                />
                {values.length <= 6 && (
                  <text
                    x={6 + (i + 0.5) * width}
                    y="31"
                    textAnchor="middle"
                    className="cm-value-text"
                  >
                    {compactNumber(value)}
                  </text>
                )}
              </g>
            );
          })}
        {panel.kind === "bars" &&
          values.map((value, i) => {
            const width = 108 / Math.max(1, values.length);
            const height = (Math.abs(value) / bound) * 38;
            return (
              <g key={i} data-value={value}>
                <rect
                  x={6 + i * width}
                  y={47 - height}
                  width={Math.max(1, width - 4)}
                  height={Math.max(1, height)}
                  rx="2"
                  className={
                    selected(i)
                      ? "cm-selected"
                      : value < 0
                        ? "cm-negative"
                        : "cm-data"
                  }
                />
                <text
                  x={6 + (i + 0.5) * width}
                  y="55"
                  textAnchor="middle"
                  className="cm-index-text"
                >
                  {i}
                </text>
              </g>
            );
          })}
        {panel.kind === "rotation" && (
          <>
            <circle cx="60" cy="28" r="23" className="cm-ring" />
            <path d="M29 28H91M60 3V53" className="cm-axis" />
            <path d="M60 28H83" className="cm-reference" />
            <line
              x1="60"
              y1="28"
              x2={
                60 +
                ((values[0] ?? 0) /
                  Math.max(1, Math.hypot(values[0] ?? 0, values[1] ?? 0))) *
                  23
              }
              y2={
                28 -
                ((values[1] ?? 0) /
                  Math.max(1, Math.hypot(values[0] ?? 0, values[1] ?? 0))) *
                  23
              }
              className="cm-rotation-vector"
            />
            <circle
              cx={
                60 +
                ((values[0] ?? 0) /
                  Math.max(1, Math.hypot(values[0] ?? 0, values[1] ?? 0))) *
                  23
              }
              cy={
                28 -
                ((values[1] ?? 0) /
                  Math.max(1, Math.hypot(values[0] ?? 0, values[1] ?? 0))) *
                  23
              }
              r="3"
              className="cm-dot"
            />
          </>
        )}
        {panel.kind === "plot" && (
          <>
            <path d="M6 28H114" className="cm-axis" />
            {(rows[0] ?? []).map((value, i) => {
              const x = 12 + (i * 96) / Math.max(1, (rows[0]?.length ?? 1) - 1);
              const y = 28 - (value / bound) * 21,
                reconstructedY = 28 - ((rows[1]?.[i] ?? value) / bound) * 21;
              return (
                <g key={i} data-value={value}>
                  <line
                    x1={x}
                    x2={x}
                    y1={y}
                    y2={reconstructedY}
                    className="cm-error"
                  />
                  <circle cx={x} cy={y} r="2.5" className="cm-original" />
                  <rect
                    x={x - 2}
                    y={reconstructedY - 2}
                    width="4"
                    height="4"
                    className="cm-data"
                  />
                </g>
              );
            })}
          </>
        )}
        {panel.kind === "experts" &&
          (panel.items ?? []).map((item, i) => {
            const width = 108 / Math.max(1, panel.items?.length ?? 1);
            return (
              <g
                key={i}
                data-selected={selected(i)}
                data-value={values[i] ?? 0}
              >
                <path
                  d={`M60 3L${6 + (i + 0.5) * width} 19`}
                  className={selected(i) ? "cm-route-selected" : "cm-route"}
                />
                <rect
                  x={6 + i * width}
                  y="20"
                  width={Math.max(1, width - 4)}
                  height="30"
                  rx="4"
                  className={selected(i) ? "cm-selected" : "cm-sleeping"}
                />
                <text
                  x={6 + (i + 0.5) * width}
                  y="38"
                  textAnchor="middle"
                  className="cm-value-text"
                >
                  E{i}
                </text>
                <title>{`${item}：${values[i] ?? 0}${selected(i) ? " · 参与计算" : " · 本轮休眠"}`}</title>
              </g>
            );
          })}
        {["tokens", "cache", "queue", "graph"].includes(panel.kind) &&
          (panel.items ?? []).map((item, i) => {
            const count = panel.items?.length ?? 1;
            const cols = Math.min(4, count),
              width = 108 / Math.max(1, cols),
              height = 44 / Math.max(1, Math.ceil(count / cols));
            const x = 6 + (i % cols) * width,
              y = 6 + Math.floor(i / cols) * height;
            return (
              <g key={i} data-selected={selected(i)}>
                {panel.kind === "graph" && i > 0 && (
                  <path
                    d={`M${x - 3} ${y + height / 2}H${x}`}
                    className="cm-route-selected"
                  />
                )}
                <rect
                  x={x}
                  y={y}
                  width={Math.max(1, width - 4)}
                  height={Math.max(1, height - 4)}
                  rx={panel.kind === "cache" ? 1 : 4}
                  className={
                    selected(i)
                      ? "cm-selected"
                      : panel.kind === "queue"
                        ? "cm-queue"
                        : "cm-token"
                  }
                />
                <text
                  x={x + (width - 4) / 2}
                  y={y + height / 2 + 2}
                  textAnchor="middle"
                  className="cm-token-text"
                >
                  {item.slice(0, 5)}
                </text>
                <title>{item}</title>
              </g>
            );
          })}
      </svg>
    </div>
  );
}

type MotionObject = {
  x: number;
  y: number;
  w: number;
  h: number;
  opacity: number;
  tone: number;
};
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const ease = (t: number) => t * t * (3 - 2 * t);
const cacheIds = new Set([
  "kv-cache",
  "prefill",
  "decode",
  "paged-attention",
  "prefix-caching",
  "chunked-prefill",
]);
const matrixIds = new Set([
  "attention",
  "flash-attention",
  "gated-deltanet",
  "embedding",
]);
const branchIds = new Set([
  "qkv",
  "gqa",
  "parallelism",
  "residual",
  "attention-output",
  "ffn",
]);

function numbers(frame: MechanismFrame): number[] {
  if (frame.output?.length) return frame.output;
  const panel =
    frame.panels.find((p) => p.kind === "bars" || p.kind === "vector") ??
    frame.panels.find((p) => p.matrix?.length);
  return panel?.values ?? panel?.matrix?.flat() ?? [];
}
function family(id: string) {
  if (id === "rope") return "rotation";
  if (id === "moe") return "routing";
  if (cacheIds.has(id)) return "cache";
  if (id === "batching") return "queue";
  if (id === "tokenization") return "merge";
  if (id === "quantization") return "quantization";
  if (id === "speculation") return "verification";
  if (id === "cuda-graphs" || id === "frameworks") return "graph";
  if (id === "metrics") return "timing";
  if (matrixIds.has(id)) return "matrix";
  if (branchIds.has(id)) return "branches";
  return "channels";
}
function cacheOccupancy(frame: MechanismFrame): number {
  const stored = frame.panels.find((p) => p.kind === "cache");
  return (
    frame.readouts.cachedTokens ??
    frame.readouts.processedInput ??
    frame.readouts.reusedTokens ??
    frame.readouts.matchedTokens ??
    frame.readouts.blocks ??
    stored?.selected?.length ??
    stored?.items?.filter((v) => !/(未分配|空闲|无共同|0 个)/.test(v)).length ??
    0
  );
}

function objectCount(id: string, frames: readonly MechanismFrame[]) {
  if (id === "moe" || id === "batching") return 3;
  if (id === "rope") return 2;
  if (id === "qkv") return 6;
  if (id === "tokenization") return 5;
  if (cacheIds.has(id))
    return Math.min(12, Math.max(4, ...frames.map(cacheOccupancy)) + 1);
  const numeric = Math.max(0, ...frames.map((f) => numbers(f).length));
  const items = Math.max(
    0,
    ...frames.flatMap((f) => f.panels.map((p) => p.items?.length ?? 0)),
  );
  return Math.min(16, Math.max(4, numeric, items) + (cacheIds.has(id) ? 1 : 0));
}

/** Stable slots represent channels/tokens throughout the movie; only their geometry changes.
 * These are display coordinates, never interpolated mathematical outputs. */
function keyObject(
  id: string,
  frames: readonly MechanismFrame[],
  index: number,
  slot: number,
  count: number,
): MotionObject {
  const frame = frames[index],
    progress = index / Math.max(1, frames.length - 1);
  const vals = numbers(frame),
    value = vals[slot % Math.max(1, vals.length)] ?? 0;
  const bound = Math.max(1, ...frames.flatMap(numbers).map(Math.abs));
  const kind = family(id);
  const base = {
    x: 145 + progress * 530,
    y: 194,
    w: 28,
    h: 28,
    opacity: 1,
    tone: value < 0 ? 1 : 0,
  };
  if (kind === "routing")
    return {
      ...base,
      y:
        index === 0
          ? 192 + slot * 3
          : index === frames.length - 1
            ? 188 + slot * 8
            : 108 + slot * 84,
      w: index === frames.length - 1 ? 46 : 34,
      h: 26,
    };
  if (kind === "branches") {
    const lanes = id === "qkv" ? 3 : id === "residual" ? 2 : 4;
    const lane = Math.floor((slot * lanes) / count);
    return {
      ...base,
      x: base.x + (slot % Math.max(1, count / lanes)) * 22,
      y:
        index === 0
          ? 178 + (slot % 2) * 32
          : 96 + lane * (190 / Math.max(1, lanes - 1)),
      w: 24,
      h: 16 + (Math.abs(value) / bound) * 24,
    };
  }
  if (kind === "cache") {
    const occupied = cacheOccupancy(frame);
    const stored = slot < occupied;
    return {
      ...base,
      x: stored
        ? 360 + (slot % 4) * 62
        : 104 + progress * 520 + (slot % 4) * 22,
      y: stored
        ? 152 + Math.floor(slot / 4) * 64
        : 122 + Math.floor(slot / 4) * 53,
      w: stored ? 48 : 25,
      h: stored ? 44 : 25,
      opacity: stored || slot === occupied || slot < 3 ? 1 : 0.12,
      tone: stored ? 0 : 0.65,
    };
  }
  if (kind === "queue") {
    const request = ["A", "B", "C"][slot];
    const completed = frames
      .slice(1, index + 1)
      .some((f) =>
        f.panels.some(
          (p) =>
            /完成|已完成/.test(p.label) && p.items?.some((v) => v === request),
        ),
      );
    const running = frame.panels
      .find((p) => /执行/.test(p.label))
      ?.items?.some((v) => v.startsWith(request));
    return {
      ...base,
      x: completed
        ? 640 + progress * 70
        : running
          ? 392 + index * 10
          : 115 + slot * 30,
      y: 116 + slot * 76,
      w: 54,
      h: 35,
      tone: completed ? 0 : running ? 0.25 : 0.9,
    };
  }
  if (kind === "merge") {
    const groups = frame.panels.find((p) => p.kind === "tokens")?.items ?? [
      "l",
      "o",
      "w",
      "e",
      "r",
    ];
    const chars = ["l", "o", "w", "e", "r"];
    const group = Math.max(
      0,
      groups.findIndex((v) => v.includes(chars[slot])),
    );
    const charsBefore = groups.slice(0, group).join("").length;
    return {
      ...base,
      x: 122 + progress * 235 + group * 90 + (slot - charsBefore) * 23,
      y: 189 - (index === 4 ? 14 : 0),
      w: 23,
      h: 39,
      tone: 0.7 - progress * 0.7,
    };
  }
  if (kind === "matrix") {
    const matrices = frame.panels.filter((p) => p.matrix?.length);
    const panel =
      id === "attention" && index === 0 ? matrices.at(-1) : matrices[0];
    const matrix = panel?.matrix ?? [],
      columns = matrix[0]?.length ?? 3;
    const v = matrix.flat()[slot] ?? vals[slot] ?? 0;
    const finalOutput = index === frames.length - 1 && frame.output?.length;
    return {
      ...base,
      x: finalOutput
        ? 651 + (slot % 2) * 40
        : 235 + progress * 175 + (slot % columns) * 43,
      y: finalOutput ? 178 : 105 + Math.floor(slot / columns) * 43,
      w: 34,
      h: finalOutput ? 28 + Math.abs(v) * 20 : 34,
      opacity: finalOutput
        ? slot < frame.output!.length
          ? 1
          : 0.04
        : slot < matrix.flat().length
          ? panel?.mask && slot % columns > Math.floor(slot / columns)
            ? 0.12
            : 0.35 + Math.min(1, Math.abs(v)) * 0.65
          : 0.06,
      tone: v < 0 ? 1 : 0,
    };
  }
  if (kind === "rotation") {
    // Position/angle metadata are not vector channels. Hold the input pair
    // until rotation starts, then morph into the catalog's computed pair.
    const pair = (index === frames.length - 1
      ? frames.at(-1)
      : frames[0]
    )?.panels.find((p) => p.kind === "rotation")?.values ?? [1, 0];
    const channel = pair[slot] ?? 0;
    return {
      ...base,
      x:
        (index === 0 ? 110 : index === frames.length - 1 ? 650 : 310) +
        slot * 45,
      y: 185,
      w: 30,
      h: 24 + Math.min(1, Math.abs(channel)) * 45,
      tone: channel < 0 ? 1 : 0,
    };
  }
  if (kind === "verification") {
    const accepted = frames[1].readouts.accepted ?? 0;
    const corrected = slot === 3,
      rejected = !corrected && slot >= accepted;
    return {
      ...base,
      x: 126 + slot * 62 + progress * 305,
      y: index < 2 ? 162 : rejected ? 262 : 162,
      w: 47,
      h: 35,
      opacity: corrected && index < 2 ? 0.08 : 1,
      tone: rejected && index >= 1 ? 1 : 0,
    };
  }
  if (kind === "graph") {
    return {
      ...base,
      x: 115 + progress * 523 + slot * 19,
      y:
        index === 0
          ? 123 + slot * 39
          : index === 1
            ? 168 + (slot % 2) * 35
            : index === 2
              ? 194
              : 126 + (slot % 2) * 96,
      w: index === 2 ? 19 : 27,
      h: index === 1 ? 35 : 23,
    };
  }
  if (kind === "timing") {
    const times = frames[2].panels.find((p) => p.kind === "vector")?.values ?? [
      3, 5, 7,
    ];
    const total = times.at(-1) ?? 7;
    const time = slot === 0 ? 0 : (times[slot - 1] ?? total);
    return {
      ...base,
      x:
        index < 2
          ? 112 + progress * 225 + slot * 18
          : 112 + (time / total) * 550 + progress * 8,
      y: index === 0 ? 138 : 184 + (slot % 2) * 50,
      w: index === 3 ? 26 : 15,
      h: 28,
      opacity: index < 2 && slot > index ? 0.1 : 1,
    };
  }
  if (kind === "quantization") {
    const plot = frame.panels.find((p) => p.kind === "plot");
    const source = frames
      .flatMap((f) => f.panels)
      .find((p) => p.kind === "plot")?.matrix;
    const values =
      plot?.matrix?.[1] ??
      frame.panels.find((p) => p.kind === "vector")?.values ??
      source?.[0] ??
      [];
    return {
      ...base,
      x: 168 + slot * 79 + progress * 28,
      y: 199 - ((values[slot] ?? 0) / bound) * 73,
      w: 16,
      h: 16,
    };
  }
  const row = slot < count / 2 ? 0 : 1,
    col = slot % Math.ceil(count / 2);
  const height = 12 + (Math.abs(value) / bound) * 76;
  return {
    ...base,
    x: 134 + col * (420 / Math.max(3, Math.ceil(count / 2))) + progress * 106,
    y: 185 + row * 75 - (value >= 0 ? height : 0),
    w: Math.min(46, 260 / count),
    h: height,
    opacity: slot < Math.max(1, vals.length) ? 1 : 0.12,
  };
}

const captions: Record<string, [string, string, string]> = {
  rotation: ["通道对", "位置旋转 · 长度不变", "旋转后的 Q / K"],
  routing: ["同一个 Token", "Top-2 专家 + 共享分支", "加权融合"],
  cache: ["新输入", "处理后写入 · 历史复用", "继续 / 释放"],
  queue: ["等待队列", "活动槽位 · 每轮推进", "完成并释放"],
  merge: ["字符片段", "相邻片段逐次合并", "Token ID"],
  matrix: ["输入张量", "选行 / 遮罩 / 加权读取", "输出张量"],
  branches: ["输入通道", "分支投影 / 分组计算", "交付下一算子"],
  quantization: ["原始权重", "缩放 → 离散 → 重建", "观察重建误差"],
  channels: ["输入通道", "逐通道运算与重组", "输出通道"],
  verification: ["草稿候选", "目标模型验证", "确认 / 回滚"],
  graph: ["输入 / 请求", "依赖节点与执行路径", "输出与反馈"],
  timing: ["请求到达", "等待 → 首 Token → 间隔", "观测窗口"],
};

export default function ContinuousMechanismScene({
  definition,
  frames,
  position,
}: {
  definition: MechanismDefinition;
  frames: readonly MechanismFrame[];
  position: number;
}) {
  if (!frames.length) return null;
  const last = frames.length - 1;
  const bounded = Math.max(
    0,
    Math.min(last, Number.isFinite(position) ? position : 0),
  );
  const index = Math.floor(bounded),
    next = Math.min(last, index + 1),
    t = ease(bounded - index);
  const progress = continuousProgress(bounded, frames.length);
  const current = frames[index],
    kind = family(definition.id),
    count = objectCount(definition.id, frames);
  const objects = Array.from({ length: count }, (_, slot) => {
    const a = keyObject(definition.id, frames, index, slot, count),
      b = keyObject(definition.id, frames, next, slot, count);
    return Object.fromEntries(
      Object.keys(a).map((k) => [
        k,
        lerp(a[k as keyof MotionObject], b[k as keyof MotionObject], t),
      ]),
    ) as MotionObject;
  });
  const rotation =
    frames
      .flatMap((f) => f.panels)
      .filter((p) => p.kind === "rotation")
      .at(-1)?.values?.[2] ?? 0;
  const angle = rotation * ease(Math.max(0, Math.min(1, bounded - 1)));
  const tip = [76 * Math.cos(angle), -76 * Math.sin(angle)];
  const selected = frames
    .flatMap((f) => f.panels)
    .find((p) => p.kind === "experts")?.selected ?? [0, 1];
  const routeWeights = frames
    .flatMap((f) => f.panels)
    .find((p) => p.label === "选中专家的权重")?.values ?? [0.5, 0.5];
  const cacheLabels =
    frames
      .flatMap((f) => f.panels)
      .filter(
        (p) =>
          p.kind === "cache" &&
          p.items?.some((v) => !/(空闲|未分配|无共同|0 个)/.test(v)),
      )
      .sort((a, b) => (b.items?.length ?? 0) - (a.items?.length ?? 0))[0]
      ?.items ?? [];
  const scan = 245 + progress * 270;
  return (
    <div
      className={`cm-scene cm-view-${definition.view}`}
      data-testid="continuous-mechanism-scene"
      data-view={definition.view}
      data-progress={progress}
      data-position={bounded}
      aria-label={`${definition.title}连续运算过程`}
    >
      <div className="cm-heading">
        <span>连续运算 · 同一组数据对象</span>
        <output>{Math.round(progress * 100)}%</output>
      </div>
      <p className="cm-motion-note">
        形状、位置与颜色连续插值，用于理解运算过程；数值保持关键帧的精确计算结果。
      </p>
      <div
        className="cm-viewport"
        role="region"
        aria-label="连续原理通路，可横向滚动"
        tabIndex={0}
      >
        <svg
          className="cm-canvas"
          viewBox="0 0 820 355"
          role="img"
          aria-label={`${definition.title}：输入经过${frames.map((f) => f.stage).join("、")}形成输出`}
          data-motion-family={kind}
        >
          <defs>
            <linearGradient id={`cm-path-${definition.id}`}>
              <stop stopColor="#9acdb0" stopOpacity=".12" />
              <stop offset=".55" stopColor="#9acdb0" stopOpacity=".5" />
              <stop offset="1" stopColor="#9acdb0" stopOpacity=".12" />
            </linearGradient>
          </defs>
          <path
            className="cm-main-path"
            stroke={`url(#cm-path-${definition.id})`}
            d="M48 195H770"
          />
          {[90, 410, 730].map((x, i) => (
            <g key={x}>
              <text x={x} y="39" textAnchor="middle" className="cm-zone-index">
                0{i + 1}
              </text>
              <text x={x} y="61" textAnchor="middle" className="cm-zone-title">
                {captions[kind][i]}
              </text>
            </g>
          ))}
          <rect
            x="224"
            y="79"
            width="367"
            height="221"
            rx="18"
            className="cm-workspace"
          />
          {kind === "rotation" && (
            <g
              className="cm-rotation-apparatus"
              data-rotation-angle={angle}
              data-rotation-tip={tip.join(",")}
            >
              <circle cx="393" cy="189" r="76" className="cm-ring" />
              <path d="M301 189H486M393 97V281" className="cm-axis" />
              <path d="M393 189H469" className="cm-reference" />
              <path
                d={`M469 189 A76 76 0 ${angle > Math.PI ? 1 : 0} 0 ${393 + tip[0]} ${189 + tip[1]}`}
                className="cm-angle-arc"
              />
              <line
                x1="393"
                y1="189"
                x2={393 + tip[0]}
                y2={189 + tip[1]}
                className="cm-rotation-vector"
              />
              <circle
                cx={393 + tip[0]}
                cy={189 + tip[1]}
                r="6"
                className="cm-dot"
              />
              <text
                x="393"
                y="319"
                textAnchor="middle"
                className="cm-scene-label"
              >
                同一向量旋转 · 弧线表示已走过的角度
              </text>
            </g>
          )}
          {kind === "routing" && (
            <g>
              {[0, 1, 2].map((lane) => (
                <g
                  key={lane}
                  data-routed-expert={lane === 2 ? "shared" : selected[lane]}
                >
                  <path
                    strokeWidth={
                      lane === 2 ? 1.5 : 1 + 3 * (routeWeights[lane] ?? 0)
                    }
                    d={`M150 195C250 195 270 ${121 + lane * 84} 380 ${121 + lane * 84}S580 195 715 195`}
                    className="cm-route-selected"
                    opacity={lane === 2 ? 0.5 : 0.8}
                  />
                  <rect
                    x="425"
                    y={100 + lane * 84}
                    width="97"
                    height="42"
                    rx="7"
                    className="cm-apparatus-cell"
                  />
                  <text
                    x="437"
                    y={126 + lane * 84}
                    textAnchor="start"
                    className="cm-scene-label"
                  >
                    {lane === 2 ? "共享专家" : `专家 E${selected[lane]}`}
                  </text>
                </g>
              ))}
            </g>
          )}
          {kind === "branches" && (
            <g>
              {Array.from(
                { length: definition.id === "qkv" ? 3 : 4 },
                (_, lane) => (
                  <g key={lane}>
                    <path
                      d={`M115 195C235 195 240 ${104 + lane * 62} 385 ${104 + lane * 62}H734`}
                      className="cm-route"
                    />
                    <text x="416" y={92 + lane * 62} className="cm-scene-label">
                      {definition.id === "qkv"
                        ? ["Q = xWQ", "K = xWK", "V = xWV"][lane]
                        : `分支 ${lane + 1}`}
                    </text>
                  </g>
                ),
              )}
            </g>
          )}
          {kind === "cache" && (
            <g data-cache-occupancy={cacheOccupancy(current)}>
              {Array.from({ length: Math.max(8, count) }, (_, i) => (
                <g key={i}>
                  <rect
                    x={353 + (i % 4) * 62}
                    y={145 + Math.floor(i / 4) * 64}
                    width="56"
                    height="52"
                    rx="4"
                    className="cm-cache-slot"
                  />
                  <text
                    x={381 + (i % 4) * 62}
                    y={140 + Math.floor(i / 4) * 64}
                    textAnchor="middle"
                    className="cm-small-label"
                  >
                    KV {i}
                  </text>
                </g>
              ))}
              <path
                d="M123 205C240 205 260 162 350 162M600 230C680 230 670 264 741 264"
                className="cm-route-selected"
              />
              <text
                x="474"
                y="318"
                textAnchor="middle"
                className="cm-scene-label"
              >
                先处理输入，再写入 K / V
              </text>
            </g>
          )}
          {kind === "queue" && (
            <g>
              {["A", "B", "C"].map((label, i) => (
                <g key={label}>
                  <rect
                    x="278"
                    y={103 + i * 76}
                    width="292"
                    height="52"
                    rx="8"
                    className="cm-cache-slot"
                  />
                  <path d={`M76 ${133 + i * 76}H749`} className="cm-route" />
                  <text x="294" y={95 + i * 76} className="cm-small-label">
                    请求 {label} · 独立缓存
                  </text>
                </g>
              ))}
            </g>
          )}
          {kind === "matrix" && (
            <g>
              <rect
                x={scan}
                y="98"
                width="12"
                height="165"
                className="cm-matrix-scan"
              />
              <path
                d="M252 281H557M252 279V286M557 279V286"
                className="cm-axis"
              />
              <text
                x="404"
                y="319"
                textAnchor="middle"
                className="cm-scene-label"
              >
                扫描张量 → 权重与内容汇合
              </text>
            </g>
          )}
          {kind === "quantization" && (
            <g>
              <path d="M137 199H701" className="cm-axis" />
              {frames
                .flatMap((f) => f.panels)
                .find((p) => p.kind === "plot")
                ?.matrix?.[0]?.map((v, i) => {
                  const bound = Math.max(
                    1,
                    ...frames.flatMap(numbers).map(Math.abs),
                  );
                  const x = 168 + i * 79,
                    y = 199 - (v / bound) * 73;
                  return (
                    <g key={i}>
                      <circle cx={x} cy={y} r="6" className="cm-original" />
                      <path
                        d={`M${x} ${y}L${objects[i]?.x ?? x} ${objects[i]?.y ?? y}`}
                        className="cm-error"
                      />
                    </g>
                  );
                })}
              <text
                x="410"
                y="319"
                textAnchor="middle"
                className="cm-scene-label"
              >
                圆点：原始权重 · 方块：缩放 / 量化 / 重建
              </text>
            </g>
          )}
          {kind === "channels" && (
            <g>
              <path d="M126 185H668M126 260H668" className="cm-axis" />
              <text
                x="410"
                y="319"
                textAnchor="middle"
                className="cm-scene-label"
              >
                通道高度与颜色跟随运算变换
              </text>
            </g>
          )}
          {kind === "verification" && (
            <g>
              <path
                d="M95 184H724M334 212L375 279H713"
                className="cm-route-selected"
              />
              <rect
                x="302"
                y="98"
                width="257"
                height="105"
                rx="12"
                className="cm-cache-slot"
              />
              <text
                x="430"
                y="118"
                textAnchor="middle"
                className="cm-small-label"
              >
                目标模型并行验证候选
              </text>
              <text
                x="544"
                y="306"
                textAnchor="middle"
                className="cm-scene-label"
              >
                首次拒绝后的草稿 → 回滚
              </text>
            </g>
          )}
          {kind === "graph" && (
            <g>
              {[0, 1, 2].map((i) => (
                <g key={i}>
                  <rect
                    x={247 + i * 109}
                    y="147"
                    width="91"
                    height="88"
                    rx="10"
                    className="cm-cache-slot"
                  />
                  <text
                    x={293 + i * 109}
                    y="143"
                    textAnchor="middle"
                    className="cm-small-label"
                  >
                    {definition.id === "cuda-graphs"
                      ? ["QKV", "Attention", "FFN"][i]
                      : ["调度器", "执行器", "Kernel"][i]}
                  </text>
                  {i < 2 && (
                    <path
                      d={`M${338 + i * 109} 191h18`}
                      className="cm-route-selected"
                    />
                  )}
                </g>
              ))}
              <path d="M302 255C302 279 559 279 559 255" className="cm-route" />
              <text
                x="431"
                y="315"
                textAnchor="middle"
                className="cm-scene-label"
              >
                {definition.id === "cuda-graphs"
                  ? "捕获依赖与缓冲区 → 输入更新 → 图重放"
                  : "请求入队 → 批次执行 → 流式输出 / 反馈"}
              </text>
            </g>
          )}
          {kind === "timing" && (
            <g>
              <path d="M108 246H742" className="cm-axis" />
              {[0, 1, 2, 3].map((i) => (
                <g key={i}>
                  <path d={`M${112 + i * 184} 95V250`} className="cm-route" />
                  <text
                    x={112 + i * 184}
                    y="278"
                    textAnchor="middle"
                    className="cm-small-label"
                  >
                    {["到达", "首 Token", "Token 2", "Token 3"][i]}
                  </text>
                </g>
              ))}
              <path
                d={`M112 303H${112 + progress * 558}`}
                className="cm-route-selected"
              />
              <text
                x="414"
                y="322"
                textAnchor="middle"
                className="cm-scene-label"
              >
                演示步骤时间轴 · 观测窗口逐渐延伸
              </text>
            </g>
          )}
          <g className="cm-persistent-objects">
            {objects.map((o, i) => (
              <g
                key={i}
                data-motion-object={`${definition.id}-${i}`}
                data-geometry={JSON.stringify([
                  o.x,
                  o.y,
                  o.w,
                  o.h,
                  o.opacity,
                  o.tone,
                ])}
                transform={`translate(${o.x} ${o.y})`}
                opacity={o.opacity}
              >
                <rect
                  x="0"
                  y="0"
                  width={o.w}
                  height={o.h}
                  rx={kind === "merge" ? 2 : 5}
                  fill={`rgb(${lerp(150, 178, o.tone)},${lerp(211, 153, o.tone)},${lerp(168, 228, o.tone)})`}
                  className="cm-moving-object"
                />
                {(kind === "merge" ||
                  kind === "queue" ||
                  kind === "cache" ||
                  kind === "routing" ||
                  kind === "verification") && (
                  <text
                    x={o.w / 2}
                    y={o.h / 2 + 4}
                    textAnchor="middle"
                    className="cm-object-label"
                  >
                    {kind === "merge"
                      ? ["l", "o", "w", "e", "r"][i]
                      : kind === "queue"
                        ? ["A", "B", "C"][i]
                        : kind === "routing"
                          ? i === 2
                            ? "共享"
                            : "x"
                          : kind === "verification"
                            ? ["猫", "坐", "垫", "修正"][i]
                            : (cacheLabels[i] ?? "新输入").slice(0, 4)}
                  </text>
                )}
              </g>
            ))}
          </g>
          <g className="cm-stage-crossfade">
            {frames.map((f, i) => (
              <text
                key={f.stage}
                x={410 + (i - bounded) * 230}
                y="345"
                textAnchor="middle"
                className="cm-operation-label"
                opacity={Math.max(0, 1 - Math.abs(bounded - i))}
              >
                {f.stage}
              </text>
            ))}
          </g>
        </svg>
      </div>
      <p className="cm-pan-hint">窄屏可左右滚动，查看完整通路。</p>
      <div className="cm-sequence" aria-label="完整运算顺序">
        {frames.map((f, i) => (
          <span
            key={f.stage}
            data-stage-index={i}
            data-state={
              i === index ? "active" : i < index ? "complete" : "pending"
            }
          >
            <small>{String(i + 1).padStart(2, "0")}</small>
            {f.stage}
            {i < last && <b data-flow-edge={i}>→</b>}
          </span>
        ))}
      </div>
      {!!current.output?.length && (
        <div
          className="cm-exact-output"
          data-output={JSON.stringify(current.output)}
        >
          <span>已到达关键帧 · 精确结果</span>
          <div>
            {current.output.map((v, i) => (
              <output key={i} data-value={v}>
                {compactNumber(v)}
              </output>
            ))}
          </div>
        </div>
      )}
      <details className="cm-snapshots">
        <summary>展开全部关键帧的精确数据</summary>
        <div className="cm-snapshot-grid">
          {frames.map((f, i) => (
            <section key={f.stage}>
              <h4>
                {i + 1}. {f.stage}
              </h4>
              <p>{f.formula}</p>
              <div className="cm-stage-glyphs">
                {f.panels.map((p, j) => (
                  <PanelGlyph key={`${i}-${j}`} panel={p} />
                ))}
              </div>
            </section>
          ))}
        </div>
      </details>
    </div>
  );
}
