import { useMemo } from "react";
import {
  buildJourneyTracks,
  journeyNodes,
  journeyNode,
  journeyPose,
  journeySegment,
  journeyTrail,
  nodeActivity,
} from "./journeyMotion";
import "./journey.css";
import type { TraceFrame, TraceTensor } from "../simulation/inferenceTrace";
import type { ModelPreset } from "../types";
import { requestColors } from "./InferenceFlow";

function TensorGlyph({ tensor }: { tensor: TraceTensor | undefined }) {
  if (!tensor) return <span className="cj-empty">请求状态 / 控制信息</span>;
  const columns = Math.min(tensor.cols, 4);
  const crop = Array.from({ length: Math.min(tensor.rows, 2) }, (_, row) =>
    tensor.values.slice(row * tensor.cols, row * tensor.cols + columns),
  ).flat();
  return (
    <>
      <span className="cj-tensor-title">
        {tensor.name} <small>{tensor.shape}</small>
      </span>
      <div
        className="cj-values"
        style={{
          gridTemplateColumns: `repeat(${Math.min(tensor.cols, 4)},minmax(0,1fr))`,
        }}
      >
        {crop.map((n, i) => (
          <span
            key={i}
            style={{
              background: `rgba(179,154,240,${0.1 + Math.min(1, Math.abs(n)) * 0.3})`,
            }}
          >
            {n.toFixed(2)}
          </span>
        ))}
      </div>
      <small className="cj-crop-note">
        前 2 行 / 4 列以内；完整行列见数据详情。
      </small>
    </>
  );
}

const moduleMarks: Record<string, string> = {
  receive: "prompt",
  tokenize: "文本 → IDs",
  schedule: "packed batch",
  embedding: "ID → x",
  rmsnorm: "x / rms",
  qkv: "WQ · WK · WV",
  rope: "R(p)",
  attention: "softmax(QKᵀ)V",
  linear: "S ← αS + Δ",
  "attention-output": "context · Wₒ",
  residual: "x + branch",
  ffn: "SiLU(g) ⊙ up",
  moe: "Σ wᵢEᵢ + E共享",
  "final-norm": "x / rms",
  "lm-head": "x · Wᵥ",
  sample: "p → ID",
  emit: "stream",
  feedback: "ID ↺",
};
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
export default function ContinuousJourneyScene({
  trace,
  frame,
  fraction,
  position,
  model,
  requestId,
  onSeek,
  onInspect,
}: {
  trace: TraceFrame[];
  frame: TraceFrame;
  fraction: number;
  position: number;
  model: ModelPreset;
  requestId: string;
  onSeek: (i: number) => void;
  onInspect?: (id: string, element: HTMLElement) => void;
}) {
  const tracks = useMemo(() => buildJourneyTracks(trace), [trace]);
  const next = trace[Math.min(trace.length - 1, frame.index + 1)];
  const t = fraction * fraction * (3 - 2 * fraction);
  const activeLayer = frame.layer ?? next.layer ?? 0;
  const full = model.layerTypes[activeLayer] === "full_attention";
  const nextFull =
    model.layerTypes[next.layer ?? activeLayer] === "full_attention";
  const fullness = mix(full ? 1 : 0, nextFull ? 1 : 0, t);
  const operation = journeyNode(frame),
    following = journeyNode(next);
  const currentValues = frame.tensors.filter(
    (v) => v.requestId === requestId || v.requestId === "batch",
  );
  const layerIndices = useMemo(
    () =>
      trace
        .filter(
          (f) =>
            f.tick === frame.tick &&
            f.layer === frame.layer &&
            f.layer !== null,
        )
        .map((f) => f.index),
    [trace, frame.tick, frame.layer],
  );
  const layerProgress =
    frame.layer === null
      ? 0
      : (position - layerIndices[0]) / Math.max(1, layerIndices.length);
  const afterNetwork = [
    "final-norm",
    "lm-head",
    "sample",
    "emit",
    "feedback",
    "finish",
    "release",
  ].includes(frame.stage);
  const wholeLayerPosition =
    frame.layer !== null
      ? frame.layer + layerProgress
      : afterNetwork
        ? model.layerTypes.length
        : 0;
  const staticPaths = useMemo(() => {
    const track = tracks[0];
    if (!track) return [];
    const paths: string[] = [];
    const embedding = trace.findIndex((f) => f.stage === "embedding");
    if (embedding >= 0) paths.push(journeySegment(track, 0, embedding + 1));
    for (const kind of ["linear", "attention"]) {
      const center = trace.findIndex((f) => f.stage === kind);
      if (center < 0) continue;
      const layer = trace[center].layer;
      const start = trace.findIndex(
        (f) => f.layer === layer && f.tick === trace[center].tick,
      );
      let end = center;
      while (end + 1 < trace.length && trace[end + 1].layer === layer) end++;
      paths.push(
        journeySegment(track, start, Math.min(end + 1, trace.length - 1)),
      );
    }
    const head = trace.findIndex((f) => f.stage === "final-norm"),
      feedback = trace.findIndex((f) => f.stage === "feedback");
    if (head > 0)
      paths.push(
        journeySegment(
          track,
          head - 1,
          feedback >= head ? feedback : Math.min(head + 5, trace.length - 1),
        ),
      );
    if (feedback >= 0) {
      let end = feedback + 1;
      while (end < trace.length - 1 && trace[end].stage !== "embedding") end++;
      paths.push(journeySegment(track, feedback, end));
    }
    return paths;
  }, [tracks, trace]);
  const operationWeight = (stage: string) =>
    journeyNodes
      .filter((n) => n.stage === stage)
      .reduce((total, n) => total + nodeActivity(n, frame, next, fraction), 0);
  const projection = operationWeight("qkv"),
    rotation = operationWeight("rope"),
    attention = operationWeight("attention"),
    delta = operationWeight("linear"),
    ffn = operationWeight("ffn"),
    moe = operationWeight("moe");
  const request = frame.requests.find((r) => r.id === requestId)!;
  const selectedTrack = tracks.find((track) => track.id === requestId)!;
  const preview = selectedTrack.channelValues[frame.index],
    target = selectedTrack.channelValues[next.index];
  const expertFrame = frame.stage === "moe" ? frame : next;
  const expertIds =
    expertFrame.tensors
      .find(
        (v) => v.requestId === requestId && v.name === "选中专家 [ID, 权重]",
      )
      ?.values.filter((_, i) => i % 2 === 0) ?? [];
  const ropePhase = frame.stage === "rope" ? 1 : next.stage === "rope" ? t : 0;
  const projectionPhase =
    frame.stage === "qkv" ? 1 + t : next.stage === "qkv" ? t : 2;
  const deltaPhase =
    frame.stage === "linear" ? 1 + t : next.stage === "linear" ? t : 2;
  const attentionPhase =
    frame.stage === "attention" ? 1 + t : next.stage === "attention" ? t : 2;
  const probabilities = frame.probabilities.length
    ? frame.probabilities
    : next.probabilities;
  return (
    <div
      className="cj-scene cj-unbroken"
      data-testid="continuous-journey-scene"
      data-progress={position / Math.max(1, trace.length - 1)}
    >
      <div className="cj-heading">
        <div>
          <span className="mono">FOLLOW THE DATA</span>
          <h3>同一份数据，一条完整的生成回路。</h3>
        </div>
        <span className="cj-live-tag">
          {frame.pass === "mixed"
            ? "Prefill + Decode"
            : frame.pass === "prefill"
              ? "Prefill · 整段输入"
              : frame.pass === "decode"
                ? "Decode · 新输入 1 Token"
                : "请求生命周期"}
        </span>
      </div>
      <div className="cj-legend">
        {frame.requests.map((r, i) => (
          <span key={r.id} style={{ color: requestColors[i] }}>
            <i style={{ background: requestColors[i] }} />
            {r.id} ·{" "}
            {r.status === "waiting"
              ? "队列等待"
              : r.status === "done"
                ? "保留输出"
                : r.status === "prefill"
                  ? "整段提示"
                  : "单个新 Token"}
          </span>
        ))}
        <small>数据块沿线移动 · 点击模块展开原理</small>
      </div>
      <div
        className="cj-canvas-scroll"
        tabIndex={0}
        role="region"
        aria-label="连续推理通路，可横向滚动"
      >
        <svg
          className="cj-canvas"
          viewBox="0 0 940 520"
          role="group"
          aria-label="输入经过嵌入、每一网络层、预测与采样，新 Token 返回下一轮输入"
        >
          <defs>
            <pattern
              id="cj-grid"
              width="24"
              height="24"
              patternUnits="userSpaceOnUse"
            >
              <circle cx="1" cy="1" r="0.7" fill="#574966" opacity="0.35" />
            </pattern>
          </defs>
          <rect width="940" height="520" fill="url(#cj-grid)" />
          <rect
            className="cj-network-chamber"
            x="257"
            y="77"
            width="423"
            height="302"
            rx="24"
          />
          <text className="cj-section-label" x="275" y="61">
            每一网络层 · 读历史，更新表示
          </text>
          <text className="cj-section-label" x="20" y="40">
            01 / 输入
          </text>
          <text className="cj-section-label" x="710" y="210">
            03 / 预测与生成
          </text>
          {staticPaths.map((d, i) => (
            <path key={i} d={d} className="cj-conduit" />
          ))}
          <path
            d="M 630 173 Q 698 110 753 99 M 542 193 Q 660 97 753 99"
            className="cj-history-line"
          />
          <path
            d="M 310 170 Q 263 295 565 325 M 455 325 Q 380 388 290 265"
            className="cj-residual-line"
          />
          <text className="cj-section-label" x="364" y="385">
            残差流持续保留 · 本层输出进入下一层
          </text>
          <g transform="translate(718 43)">
            <rect className="cj-state-shelf" width="192" height="97" rx="12" />
            <text className="cj-module-title" x="15" y="23">
              缓存位置 · 轮末提交
            </text>
            {frame.requests.map((r, i) => (
              <g key={r.id} transform={`translate(15 ${40 + i * 16})`}>
                <text
                  x="0"
                  y="9"
                  fill={requestColors[i]}
                  className="cj-mini-label"
                >
                  {r.id}
                </text>
                {Array.from({ length: 8 }, (_, j) => (
                  <rect
                    key={j}
                    x={30 + j * 17}
                    y="0"
                    width="12"
                    height="10"
                    rx="2"
                    fill={requestColors[i]}
                    opacity={j < r.cacheTokens ? 0.8 : 0.12}
                  />
                ))}
                <text className="cj-mini-label" x="170" y="9" textAnchor="end">
                  {r.cacheTokens}
                </text>
              </g>
            ))}
            <text className="cj-mini-label" x="15" y="90">
              线性层另持有定长矩阵 S
            </text>
          </g>
          {journeyNodes
            .filter(
              (n) =>
                !["finish", "release"].includes(n.stage) &&
                (n.stage !== "ffn" || model.family !== "moe") &&
                (n.stage !== "moe" || model.family === "moe"),
            )
            .map((node, i) => {
              const activity = nodeActivity(node, frame, next, fraction);
              return (
                <g
                  key={node.stage + node.part}
                  className="cj-module"
                  transform={`translate(${node.x} ${node.y})`}
                  opacity={
                    node.stage === "rope" || node.stage === "attention"
                      ? 0.25 + 0.75 * fullness
                      : node.stage === "linear"
                        ? 1 - 0.75 * fullness
                        : 1
                  }
                  role="button"
                  tabIndex={0}
                  aria-label={`展开${node.title}原理`}
                  onClick={(e) =>
                    onInspect?.(
                      node.module,
                      e.currentTarget as unknown as HTMLElement,
                    )
                  }
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onInspect?.(
                        node.module,
                        e.currentTarget as unknown as HTMLElement,
                      );
                    }
                  }}
                >
                  <rect
                    x="-44"
                    y="-22"
                    width="88"
                    height="44"
                    rx="10"
                    fill={node.stage === "linear" ? "#203d35" : "#272233"}
                    stroke={node.stage === "linear" ? "#5c9e87" : "#726087"}
                    strokeWidth={1 + activity * 1.5}
                  />
                  <rect
                    x="-44"
                    y="-22"
                    width="88"
                    height="44"
                    rx="10"
                    fill="#bba1ed"
                    opacity={activity * 0.23}
                  />
                  <text
                    className="cj-module-title"
                    textAnchor="middle"
                    y={node.y >= 300 ? 40 : -31}
                  >
                    {node.title}
                  </text>
                  <text
                    className="cj-operator-mark"
                    textAnchor="middle"
                    y="4"
                    opacity={1 - activity * 0.85}
                  >
                    {moduleMarks[node.stage]}
                  </text>
                  <rect
                    x="-32"
                    y="14"
                    width={64 * activity}
                    height="2"
                    rx="1"
                    fill="#d5c0f9"
                  />
                  <title>{node.title}：展开独立原理动画</title>
                </g>
              );
            })}
          {/* A persistent workspace: projection splits, rotations, gates and merges
            crossfade geometrically instead of replacing an operator slide. */}
          <g transform="translate(468 235)" className="cj-operation-workspace">
            <g opacity={projection}>
              {["Q", "K", "V"].map((label, i) => (
                <g key={label} transform={`translate(${(i - 1) * 52} -12)`}>
                  <path
                    d={`M${(1 - i) * 52} 32 Q0 24 0 0`}
                    fill="none"
                    stroke={requestColors[i]}
                    strokeWidth="1.5"
                  />
                  {Array.from({ length: 4 }, (_, j) => (
                    <rect
                      key={j}
                      x={-18 + j * 10}
                      y={-10 - mix(0, 10, projectionPhase / 2)}
                      width="7"
                      height={12 + 8 * Math.abs(Math.sin(j + i))}
                      fill={requestColors[i]}
                      rx="2"
                    />
                  ))}
                  <text
                    y="-25"
                    textAnchor="middle"
                    fill={requestColors[i]}
                    className="cj-mini-label"
                  >
                    {label}
                  </text>
                </g>
              ))}
            </g>
            <g opacity={rotation}>
              <circle r="31" className="cj-rotation-ring" />
              <path d="M-38 0H38M0-38V38" className="cj-history-line" />
              <g
                transform={`rotate(${mix(0, (request.processedTokens + request.inputTokens.length - 1) * 57.2958, ropePhase)})`}
              >
                <path d="M0 0H28" stroke="#d5bcfa" strokeWidth="3" />
                <circle cx="28" r="4" fill="#d5bcfa" />
              </g>
            </g>
            <g opacity={attention} transform="translate(-45 -30)">
              {Array.from({ length: 9 }, (_, i) => (
                <rect
                  key={i}
                  x={(i % 3) * 30}
                  y={Math.floor(i / 3) * 22}
                  width="25"
                  height="17"
                  rx="3"
                  fill={i % 3 > Math.floor(i / 3) ? "#433a50" : "#b39af0"}
                  opacity={
                    i % 3 > Math.floor(i / 3)
                      ? 0.2
                      : 0.25 +
                        0.65 * (1 - Math.abs((i % 3) - attentionPhase) / 3)
                  }
                />
              ))}
              <text className="cj-mini-label" x="45" y="80" textAnchor="middle">
                因果权重 × V
              </text>
            </g>
            <g opacity={delta}>
              {Array.from({ length: 4 }, (_, i) => (
                <rect
                  key={i}
                  x={-30 + (i % 2) * 31}
                  y={-27 + Math.floor(i / 2) * 28}
                  width="25"
                  height="23"
                  rx="3"
                  fill="#9bd4bd"
                  opacity={0.3 + 0.5 * Math.abs(Math.sin(i + deltaPhase))}
                />
              ))}
              <path
                d="M-45 0Q-55-42 0-42Q50-42 47 0"
                className="cj-history-line"
              />
              <text className="cj-mini-label" x="0" y="49" textAnchor="middle">
                衰减 → 写入 → 读出 S
              </text>
            </g>
            <g opacity={ffn}>
              {Array.from({ length: 6 }, (_, i) => (
                <rect
                  key={i}
                  x={-50 + i * 18}
                  y={
                    -Math.abs(
                      Math.tanh(
                        mix(
                          preview[i % preview.length] ?? 0,
                          target[i % target.length] ?? 0,
                          t,
                        ),
                      ),
                    ) *
                      15 -
                    5
                  }
                  width="11"
                  height={
                    10 +
                    Math.abs(
                      Math.tanh(
                        mix(
                          preview[i % preview.length] ?? 0,
                          target[i % target.length] ?? 0,
                          t,
                        ),
                      ),
                    ) *
                      30
                  }
                  rx="3"
                  fill={i % 2 ? "#9bd4bd" : "#b39af0"}
                />
              ))}
              <text className="cj-mini-label" textAnchor="middle" y="48">
                扩展 · SiLU 门控 · 投影
              </text>
            </g>
            <g opacity={moe}>
              {[0, 1, 2].map((i) => (
                <g key={i}>
                  <path
                    d={`M0-34Q${(i - 1) * 50}-15 ${(i - 1) * 50}0 Q${(i - 1) * 50}28 0 38`}
                    fill="none"
                    stroke={expertIds.includes(i) ? "#b39af0" : "#5c4f6c"}
                    strokeWidth={expertIds.includes(i) ? 2 : 1}
                  />
                  <rect
                    x={(i - 1) * 50 - 17}
                    y="-10"
                    width="34"
                    height="23"
                    rx="5"
                    fill={expertIds.includes(i) ? "#b39af0" : "#413749"}
                    opacity={
                      expertIds.includes(i)
                        ? 0.5 + 0.2 * Math.sin(position)
                        : 0.35
                    }
                  />
                  <text
                    x={(i - 1) * 50}
                    y="5"
                    textAnchor="middle"
                    className="cj-mini-label"
                  >
                    E{i}
                  </text>
                </g>
              ))}
              <path
                d="M0 -34 Q-89-24 -89 22 Q-89 44 0 38"
                fill="none"
                stroke="#9bd4bd"
                strokeWidth="1.5"
              />
              <rect
                x="-110"
                y="15"
                width="42"
                height="23"
                rx="5"
                fill="#36574b"
                stroke="#9bd4bd"
              />
              <text
                x="-89"
                y="31"
                textAnchor="middle"
                className="cj-mini-label"
              >
                共享
              </text>
              <text className="cj-mini-label" textAnchor="middle" y="57">
                路由分流 → 加权汇合 + 共享
              </text>
            </g>
          </g>
          <g
            opacity={Math.max(
              operationWeight("lm-head"),
              operationWeight("sample"),
            )}
            transform="translate(750 345)"
          >
            {Array.from({ length: 6 }, (_, i) => (
              <rect
                key={i}
                x={i * 12}
                y={-35 * (probabilities[i] ?? 0)}
                width="8"
                height={Math.max(1, 35 * (probabilities[i] ?? 0))}
                fill="#efc78c"
                rx="2"
              />
            ))}
            <text x="30" y="18" textAnchor="middle" className="cj-mini-label">
              logits → 概率 → ID
            </text>
          </g>
          <path
            d="M775 425 Q840 405 870 440 L895 455"
            className="cj-history-line"
          />
          <g transform="translate(876 455)">
            <rect
              x="-36"
              y="-21"
              width="78"
              height="43"
              rx="10"
              fill="#292531"
              stroke="#6a5b75"
            />
            <text className="cj-module-title" textAnchor="middle" y="38">
              结束 · 释放状态
            </text>
          </g>
          <text
            className="cj-feedback-label"
            x="460"
            y="485"
            textAnchor="middle"
          >
            新 Token 沿回路返回 · 复用历史状态 · 继续 Decode
          </text>
          {tracks.map((track, lane) => {
            const pose = journeyPose(track, position),
              a = track.channelValues[frame.index],
              b = track.channelValues[next.index];
            const carried = track.carriedTokens[frame.index],
              upcoming = track.carriedTokens[next.index];
            const token = carried ?? upcoming;
            const tokenWeight = (carried ? 1 - t : 0) + (upcoming ? t : 0);
            return (
              <g key={track.id} data-request-track={track.id}>
                <path
                  d={journeyTrail(track, position)}
                  fill="none"
                  stroke={requestColors[lane]}
                  strokeWidth="5"
                  strokeLinecap="round"
                  opacity="0.45"
                />
                <g
                  className="cj-packet"
                  data-request-id={track.id}
                  data-x={pose.x}
                  data-y={pose.y}
                  transform={`translate(${pose.x} ${pose.y})`}
                >
                  <rect
                    x="-24"
                    y={tracks.length > 1 ? -7 : -17}
                    width="48"
                    height={tracks.length > 1 ? 14 : 34}
                    rx="9"
                    fill="#17151d"
                    stroke={requestColors[lane]}
                    strokeWidth="1.5"
                  />
                  {Array.from({ length: 4 }, (_, i) => {
                    const height =
                      (tracks.length > 1 ? 3 : 5) +
                      (tracks.length > 1 ? 7 : 14) *
                        Math.abs(Math.tanh(mix(a[i] ?? 0, b[i] ?? 0, t)));
                    return (
                      <rect
                        key={i}
                        data-channel={i}
                        x={-17 + i * 9}
                        y={-height / 2}
                        width="6"
                        height={height}
                        rx="2"
                        fill={requestColors[lane]}
                        opacity={1 - 0.85 * Math.min(1, tokenWeight)}
                      />
                    );
                  })}
                  <text
                    textAnchor="middle"
                    y="4"
                    fill={requestColors[lane]}
                    className="cj-packet-token"
                    opacity={token ? Math.min(1, tokenWeight) : 0}
                  >
                    {token?.slice(0, 3)}
                  </text>
                  <text
                    textAnchor={tracks.length > 1 ? "end" : "middle"}
                    x={tracks.length > 1 ? -30 : 0}
                    y={tracks.length > 1 ? 3 : -23}
                    fill={requestColors[lane]}
                    className="cj-packet-id"
                  >
                    {track.id}
                  </text>
                </g>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="cj-operation-caption">
        <span className="mono">NOW /</span>
        <div>
          <span style={{ opacity: 1 - t }}>{operation.title}</span>
          <span style={{ opacity: t }}>{following.title}</span>
        </div>
        <small>沿同一轨迹交接，数据块不会从起点重新出现</small>
      </div>
      <div className="cj-network-heading">
        <div>
          <span className="mono">
            {model.layerTypes.length} LAYERS · 全部真实层次顺序
          </span>
          <p>
            {frame.layer !== null
              ? `第 ${frame.layer + 1} 层 / ${model.layerTypes.length}`
              : "每一轮输入都会经过全部网络层"}
            <small>紫：Full Attention · 绿：Gated DeltaNet</small>
          </p>
        </div>
        <strong>
          {Math.round((wholeLayerPosition / model.layerTypes.length) * 100)}
          <small>%</small>
        </strong>
      </div>
      <div className="cj-layer-wave" aria-label="连续网络层进度">
        {model.layerTypes.map((kind, i) => (
          <button
            key={i}
            className={`cj-layer ${kind === "full_attention" ? "full" : "linear"}${frame.layer === i ? " current" : ""}`}
            aria-label={`连续轨迹第 ${i + 1} 层`}
            onClick={() =>
              onSeek(
                Math.max(
                  0,
                  trace.findIndex(
                    (f) => f.tick === frame.tick && f.layer === i,
                  ),
                ),
              )
            }
            style={
              {
                "--layer-fill": `${Math.max(0, Math.min(1, wholeLayerPosition - i)) * 100}%`,
              } as React.CSSProperties
            }
          >
            <i />
            <span>{i + 1}</span>
          </button>
        ))}
      </div>
      <details className="cj-exact-details">
        <summary>核对当前关键帧的精确数值 · {requestId}</summary>
        <TensorGlyph tensor={currentValues.at(-1)} />
      </details>
      <p className="cj-explanation">
        同色数据块代表同一请求；通路和图形连续插值，数值在计算事件处提交。层内的小图展示投影、旋转、读写与合并。动画时间用于讲解，不表示
        GPU 耗时。窄屏可横向滚动查看通路。
      </p>
    </div>
  );
}
