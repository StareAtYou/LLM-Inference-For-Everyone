import { useRef, useState, useLayoutEffect } from "react";
import type { ModelPreset } from "../types";
import type { TraceFrame, TraceStage } from "../simulation/inferenceTrace";
export const requestColors = ["#b39af0", "#9bd4bd", "#efc78c"];
const groups = [
  {
    id: "input",
    title: "01 / 输入与调度",
    nodes: [
      ["receive", "接收请求", "文本 + 生成配置", "frameworks"],
      ["tokenize", "Tokenization", "文本 → Token IDs", "tokenization"],
      ["schedule", "组织批次", "等待 → 本轮计算", "batching"],
      ["embedding", "Embedding", "IDs → 隐藏向量", "embedding"],
    ],
  },
  {
    id: "attention",
    title: "02 / 注意力与历史状态",
    nodes: [
      ["rmsnorm", "RMSNorm", "归一化当前表示", "rmsnorm"],
      ["qkv", "Q / K / V", "三条线性投影", "qkv"],
      ["rope", "RoPE", "位置 → 通道旋转", "rope"],
      ["attention", "Full Attention", "读取历史 K / V", "attention"],
      ["linear", "Gated DeltaNet", "短卷积 + 递归矩阵", "gated-deltanet"],
      [
        "attention-output",
        "输出投影",
        "多头结果 → 隐藏维度",
        "attention-output",
      ],
      ["residual", "残差相加", "原输入 + 注意力输出", "residual"],
    ],
  },
  {
    id: "ffn",
    title: "03 / 位置内的变换",
    nodes: [
      ["ffn-norm", "RMSNorm", "FFN 前归一化", "rmsnorm"],
      ["ffn", "Dense FFN", "扩展 → 门控 → 投影", "ffn"],
      ["moe", "MoE 路由", "选专家 → 加权合并", "moe"],
      ["ffn-residual", "残差相加", "加回输入 → 下一层", "residual"],
    ],
  },
  {
    id: "output",
    title: "04 / 预测与生成",
    nodes: [
      ["final-norm", "Final Norm", "末层输出归一化", "rmsnorm"],
      ["lm-head", "LM Head", "隐藏向量 → logits", "lm-head"],
      ["sample", "采样", "概率 → 新 Token", "sampling"],
      ["emit", "返回输出", "Token → 文本片段", "decode"],
    ],
  },
];
function activeId(frame: TraceFrame) {
  if (frame.stage === "rmsnorm" && frame.blockPart === "ffn") return "ffn-norm";
  if (frame.stage === "residual" && frame.blockPart === "ffn")
    return "ffn-residual";
  return frame.stage;
}
export default function InferenceFlow({
  frame,
  model,
  playing,
  onInspect,
  presentation = "staged",
  fraction = 0,
}: {
  frame: TraceFrame;
  model: ModelPreset;
  playing: boolean;
  presentation?: "staged" | "continuous";
  fraction?: number;
  onInspect: (id: string, element: HTMLElement) => void;
}) {
  const mapRef = useRef<HTMLDivElement>(null);
  const [edge, setEdge] = useState<{
    path: string;
    width: number;
    height: number;
  } | null>(null);
  const endpoint = (id: string) =>
    id === "rmsnorm" && frame.blockPart === "ffn"
      ? "ffn-norm"
      : id === "residual" && frame.blockPart === "ffn"
        ? "ffn-residual"
        : ["feedback", "finish", "release"].includes(id)
          ? "feedback"
          : id;
  const terminal =
    frame.stage === "release" &&
    frame.requests.every((r) => r.status === "done");
  const from = endpoint(frame.from),
    to = terminal
      ? "feedback"
      : frame.stage === "residual" && frame.blockPart === "attention"
        ? "ffn-norm"
        : frame.stage === "residual" &&
            frame.blockPart === "ffn" &&
            frame.to === "rmsnorm"
          ? "rmsnorm"
          : endpoint(frame.to);
  useLayoutEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const update = () => {
      const a = map.querySelector<HTMLElement>(`[data-node-id="${from}"]`),
        b = map.querySelector<HTMLElement>(`[data-node-id="${to}"]`);
      if (!a || !b || a === b) {
        setEdge(null);
        return;
      }
      const box = map.getBoundingClientRect(),
        ra = a.getBoundingClientRect(),
        rb = b.getBoundingClientRect();
      const sameRow = Math.abs(ra.top - rb.top) < 8;
      const continuous = presentation === "continuous";
      const x1 =
          (continuous
            ? ra.left + ra.width / 2
            : sameRow
              ? ra.right
              : ra.left + ra.width / 2) - box.left,
        y1 =
          (continuous
            ? ra.top + ra.height / 2
            : sameRow
              ? ra.top + ra.height / 2
              : ra.bottom) - box.top;
      const x2 =
          (continuous
            ? rb.left + rb.width / 2
            : sameRow
              ? rb.left
              : rb.left + rb.width / 2) - box.left,
        y2 =
          (continuous
            ? rb.top + rb.height / 2
            : sameRow
              ? rb.top + rb.height / 2
              : rb.top) - box.top;
      const path = sameRow
        ? `M ${x1} ${y1} L ${x2} ${y2}`
        : y2 >= y1
          ? `M ${x1} ${y1} C ${x1} ${(y1 + y2) / 2}, ${x2} ${(y1 + y2) / 2}, ${x2} ${y2}`
          : `M ${x1} ${y1} C ${box.width - 6} ${y1 + 24}, ${box.width - 6} ${y2 - 24}, ${x2} ${y2}`;
      setEdge({ path, width: box.width, height: box.height });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(map);
    return () => observer.disconnect();
  }, [from, to, frame.index, presentation]);
  const nodeName = (id: string) =>
    groups.flatMap((g) => g.nodes).find((n) => n[0] === id)?.[1] ??
    ({ feedback: "回送下一轮", finish: "结束", release: "释放状态" }[id] || id);
  const active = activeId(frame);
  const full =
    frame.layer !== null
      ? model.layerTypes[frame.layer] === "full_attention"
      : true;
  return (
    <div
      ref={mapRef}
      className={
        "wf-map " +
        (playing ? "wf-running " : "") +
        (presentation === "continuous" ? "wf-continuous-map" : "")
      }
      data-testid="inference-map"
    >
      <div className="wf-map-status">
        <span className="mono">LIVE DATA PATH</span>
        <span>
          {frame.pass === "mixed"
            ? "混合批次 · Prefill + Decode"
            : frame.pass === "prefill"
              ? "Prefill · 整段输入"
              : frame.pass === "decode"
                ? "Decode · 新输入 1 Token"
                : frame.pass === "finish"
                  ? "结束与释放"
                  : "请求进入系统"}
        </span>
      </div>
      <div className="wf-transfer-strip" aria-label="当前数据传递">
        <span>{nodeName(frame.from)}</span>
        <div>
          <span aria-hidden="true">→</span>
          {frame.requestIds.map((id) => (
            <b
              key={id}
              style={{
                color:
                  requestColors[frame.requests.findIndex((r) => r.id === id)],
              }}
            >
              {id}
            </b>
          ))}
          <small>
            {frame.tensors[0]?.name ?? "请求与状态"}{" "}
            {frame.tensors[0]?.shape ?? ""}
          </small>
        </div>
        <span>{terminal ? "推理完成" : nodeName(frame.to)}</span>
      </div>
      {edge && (
        <svg
          className="wf-flow-edges"
          width={edge.width}
          height={edge.height}
          viewBox={`0 0 ${edge.width} ${edge.height}`}
          aria-hidden="true"
        >
          <defs>
            <marker
              id="flow-arrow"
              markerWidth="7"
              markerHeight="7"
              refX="6"
              refY="3.5"
              orient="auto"
            >
              <path d="M 0 0 L 7 3.5 L 0 7" fill="#b39af0" />
            </marker>
          </defs>
          <path
            d={edge.path}
            fill="none"
            stroke="#b39af0"
            strokeWidth="2"
            markerEnd="url(#flow-arrow)"
            opacity=".75"
          />
          {frame.requestIds.map((id, i) => (
            <circle
              className={
                "wf-moving-request " +
                (presentation === "continuous" ? "wf-timeline-particle" : "")
              }
              key={frame.index + "-" + id}
              r="4"
              fill={requestColors[frame.requests.findIndex((r) => r.id === id)]}
              style={{
                offsetPath: `path('${edge.path}')`,
                animationDelay: `${i * 0.08}s`,
                ...(presentation === "continuous"
                  ? {
                      animation: "none",
                      offsetDistance: `${fraction * 100}%`,
                      opacity: 1,
                    }
                  : {}),
              }}
            />
          ))}
        </svg>
      )}
      {groups.map((group) => (
        <div className={"wf-map-group wf-group-" + group.id} key={group.id}>
          <div className="wf-group-title">
            <span>{group.title}</span>
            {group.id === "attention" && (
              <span>
                {frame.layer === null
                  ? "层内逻辑顺序"
                  : `第 ${frame.layer + 1} / ${model.layerTypes.length} 层`}
                <i>
                  {" "}
                  ·{" "}
                  {frame.layer === null
                    ? "混合网络"
                    : full
                      ? "全注意力"
                      : "线性状态"}
                </i>
              </span>
            )}
            {group.id === "ffn" && (
              <span>本模型 · {model.family === "moe" ? "MoE" : "Dense"}</span>
            )}
          </div>
          <div className="wf-node-row">
            {group.nodes.map(([nodeId, title, subtitle, mechanism]) => {
              const inactive =
                (nodeId === "linear" && full && frame.layer !== null) ||
                (["rope", "attention"].includes(nodeId) &&
                  !full &&
                  frame.layer !== null) ||
                (nodeId === "moe" && model.family !== "moe") ||
                (nodeId === "ffn" && model.family === "moe");
              return (
                <button
                  type="button"
                  data-node-id={nodeId}
                  key={nodeId}
                  className={
                    "wf-node " +
                    (active === nodeId ? "wf-node-active " : "") +
                    (inactive ? "wf-node-inactive" : "")
                  }
                  onClick={(e) => onInspect(mechanism, e.currentTarget)}
                  aria-label={"展开 " + title + " 原理"}
                >
                  <span className="wf-node-glyph" aria-hidden="true">
                    {nodeId === "embedding"
                      ? "▥"
                      : nodeId === "qkv"
                        ? "⋈"
                        : nodeId === "rope"
                          ? "↻"
                          : nodeId === "attention"
                            ? "▦"
                            : nodeId === "linear"
                              ? "⊞"
                              : nodeId.includes("residual")
                                ? "＋"
                                : nodeId === "moe"
                                  ? "⠿"
                                  : nodeId === "sample"
                                    ? "◒"
                                    : nodeId === "schedule"
                                      ? "▤"
                                      : "◇"}
                  </span>
                  <strong>{title}</strong>
                  <small>{subtitle}</small>
                  <span className="wf-node-hint">
                    {inactive ? "本层未走此分支" : "点击展开 ↗"}
                  </span>
                  {active === nodeId && presentation !== "continuous" && (
                    <span
                      className="wf-data-packet"
                      key={frame.index}
                      style={
                        {
                          "--packet-color":
                            requestColors[
                              Math.max(
                                0,
                                frame.requests.findIndex(
                                  (r) => r.id === frame.requestIds[0],
                                ),
                              )
                            ],
                        } as React.CSSProperties
                      }
                      aria-hidden="true"
                    />
                  )}
                </button>
              );
            })}
          </div>
          {group.id === "attention" && (
            <div className="wf-memory-link">
              <span aria-hidden="true">↔</span>
              <span>Full Attention：按位置保存 K/V</span>
              <span>Gated DeltaNet：更新固定形状状态</span>
            </div>
          )}
          {group.id === "ffn" && (
            <div className="wf-layer-loop">
              <span aria-hidden="true">↳</span> 输出回到下一层的 RMSNorm；共{" "}
              {model.layerTypes.length} 层，轨迹逐层计算。
            </div>
          )}
        </div>
      ))}
      <div
        data-node-id="feedback"
        className={
          "wf-feedback " +
          (["feedback", "finish", "release"].includes(frame.stage)
            ? "wf-feedback-active"
            : "")
        }
      >
        <div>
          <span aria-hidden="true">↶</span>
          <strong>
            {frame.stage === "finish" || frame.stage === "release"
              ? "达到长度上限 → 结束 → 释放状态"
              : "新 Token → 作为下一轮输入 → Embedding"}
          </strong>
        </div>
        <small>
          采样结果要在下一轮被读入后才加入 K/V；最后输出的 Token
          不会自动进入缓存。
        </small>
      </div>
    </div>
  );
}
export const overviewStages: TraceStage[] = [
  "receive",
  "tokenize",
  "schedule",
  "embedding",
  "final-norm",
  "lm-head",
  "sample",
  "emit",
  "feedback",
  "finish",
  "release",
];
