import { useMemo, useRef, useState } from "react";
import { useTimeline } from "../hooks/useTimeline";
import ContinuousMechanismScene, {
  continuousProgress,
} from "./ContinuousMechanismScene";
import {
  ArrowRight,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
  SkipForward,
} from "lucide-react";
import {
  buildMechanismFrames,
  mechanisms,
  type MechanismDefinition,
  type MechanismPanel,
} from "./catalog";
import "./mechanisms.css";
import ShapeTeachingPanel, {
  MechanismModelComparison,
} from "../education/ShapeTeachingPanel";
import { deriveMechanismShapes } from "../education/shapeDerivations";

const format = (value: number) => {
  if (!Number.isFinite(value)) return "—";
  if (Number.isInteger(value)) return String(value);
  if (Math.abs(value) < 0.0001 && value !== 0) return value.toExponential(2);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
};
const readoutLabels: Record<string, string> = {
  rms: "均方根 r",
  meanSquare: "均方值",
  angle: "旋转角 / rad",
  length: "向量长度",
  contribution: "选中项贡献",
  weightSum: "权重之和",
  probabilitySum: "概率之和",
  temperature: "温度",
  coefficient: "分支系数",
  kvHeads: "KV 头",
  queryHeads: "Q 头",
  cacheRatio: "相对缓存量",
  alpha: "保留门 α",
  beta: "写入门 β",
  convolution: "局部卷积值",
  selectedExperts: "参与专家",
  cachedTokens: "已缓存 Token",
  sampledTokens: "已采样 Token",
  queryPositions: "本轮查询位置",
  random: "固定随机数 u",
  sampledId: "采样 ID",
  releasedTokens: "释放的位置数",
  blocks: "分配块数",
  capacity: "总 Token 容量",
  slack: "尾块空位",
  reusedTokens: "复用位置",
  newlyComputedTokens: "新计算位置",
  matchedTokens: "匹配前缀",
  computedTokens: "计算位置",
  waiting: "等待请求",
  allocated: "占用槽位",
  completed: "完成请求",
  mse: "均方误差",
  payloadBytes: "整数 payload / B",
  scale: "量化 scale",
  accepted: "接受草稿",
  discarded: "丢弃草稿",
  committed: "确认 Token",
  max: "在线最大值 m",
  denominator: "分母 l",
  numerator: "加权和 o",
  devices: "设备数",
  processedInput: "已处理输入",
  decodeSteps: "decode 轮次",
  tokenBudget: "本轮 Token 预算",
  hostLaunchCalls: "主机启动调用",
  arrival: "到达步骤",
  start: "执行开始步骤",
  ttft: "TTFT / 教学步",
  waitingTime: "等待 / 教学步",
  itl: "ITL / 教学步",
  throughput: "输出 Token / 教学步",
  outputTokens: "输出 Token",
  tokens: "Token 数",
};

function DataPanel({ panel, index }: { panel: MechanismPanel; index: number }) {
  const [picked, setPicked] = useState<number | null>(null);
  const rows = panel.matrix ?? [];
  const max = Math.max(1, ...(panel.values ?? []).map(Math.abs));
  const active = (i: number) => picked === i || panel.selected?.includes(i);
  return (
    <div
      className={`mp-panel mp-panel-${panel.kind}`}
      data-panel-kind={panel.kind}
    >
      <div className="mp-panel-head">
        <span className="mp-panel-index">
          {String(index + 1).padStart(2, "0")}
        </span>
        <h4>{panel.label}</h4>
      </div>
      {panel.kind === "matrix" && (
        <>
          <div
            className="mp-matrix"
            style={{
              gridTemplateColumns: `repeat(${rows[0]?.length ?? 1}, minmax(44px, 1fr))`,
            }}
            role="group"
            aria-label={panel.label}
          >
            {rows.flatMap((row, r) =>
              row.map((value, c) => {
                const i = r * row.length + c,
                  masked = panel.mask && c > r;
                return (
                  <button
                    key={i}
                    type="button"
                    className={`mp-cell ${active(i) ? "mp-selected" : ""} ${masked ? "mp-masked" : ""}`}
                    style={
                      {
                        "--mp-intensity": Math.min(1, Math.abs(value)),
                      } as React.CSSProperties
                    }
                    aria-label={`${panel.label}，行 ${r} 列 ${c}，${masked ? "未来位置被遮罩" : format(value)}`}
                    aria-pressed={picked === i}
                    onClick={() => setPicked(picked === i ? null : i)}
                  >
                    {masked ? "×" : format(value)}
                  </button>
                );
              }),
            )}
          </div>
          {picked !== null && (
            <p className="mp-cell-detail">
              行 {Math.floor(picked / rows[0].length)} · 列{" "}
              {picked % rows[0].length} · 值 {format(rows.flat()[picked])}
            </p>
          )}
        </>
      )}
      {panel.kind === "vector" && (
        <div className="mp-vector" role="group" aria-label={panel.label}>
          {panel.values?.map((value, i) => (
            <button
              key={i}
              type="button"
              className={`mp-vector-cell ${active(i) ? "mp-selected" : ""}`}
              aria-label={`通道 ${i}，值 ${format(value)}`}
              aria-pressed={picked === i}
              onClick={() => setPicked(picked === i ? null : i)}
            >
              <small>{i}</small>
              <span>{format(value)}</span>
            </button>
          ))}
        </div>
      )}
      {panel.kind === "bars" && (
        <div className="mp-bars">
          {panel.values?.map((value, i) => (
            <button
              key={i}
              type="button"
              className={`mp-bar-row ${active(i) ? "mp-selected" : ""}`}
              aria-label={`${panel.columns?.[i] ?? i}，${format(value)}`}
              aria-pressed={picked === i}
              onClick={() => setPicked(picked === i ? null : i)}
            >
              <span className="mp-bar-label">{panel.columns?.[i] ?? i}</span>
              <span
                className={`mp-bar-track ${value < 0 ? "mp-negative" : ""}`}
              >
                <span
                  style={{
                    width: `${Math.max(0, Math.abs(value) / max) * 100}%`,
                  }}
                />
              </span>
              <span className="mp-bar-value">{format(value)}</span>
            </button>
          ))}
        </div>
      )}
      {["tokens", "cache", "queue", "graph"].includes(panel.kind) && (
        <div className={`mp-items mp-items-${panel.kind}`}>
          {panel.items?.map((item, i) => (
            <div
              key={i}
              className={`mp-item ${active(i) ? "mp-selected" : ""}`}
            >
              <span>{item}</span>
              {panel.kind === "graph" && i < panel.items!.length - 1 && (
                <svg
                  className="mp-graph-edge"
                  viewBox="0 0 40 26"
                  width="40"
                  height="26"
                  aria-hidden="true"
                >
                  <path d="M20 0V22M16 18L20 22L24 18" />
                  <circle cx="20" cy="3" r="2" />
                </svg>
              )}
            </div>
          ))}
        </div>
      )}
      {panel.kind === "experts" && (
        <div className="mp-experts">
          {panel.items?.map((item, i) => (
            <button
              key={i}
              type="button"
              className={`mp-expert ${active(i) ? "mp-selected" : ""}`}
              aria-label={`${item}，路由概率 ${format(panel.values?.[i] ?? 0)}${panel.selected?.includes(i) ? "，被选中" : "，未选中"}`}
              aria-pressed={picked === i}
              onClick={() => setPicked(picked === i ? null : i)}
            >
              <span>{item}</span>
              <strong>{format(panel.values?.[i] ?? 0)}</strong>
              <small>
                {panel.selected?.includes(i) ? "参与计算" : "本轮休眠"}
              </small>
            </button>
          ))}
        </div>
      )}
      {panel.kind === "rotation" && (
        <svg
          className="mp-rotation"
          viewBox="0 0 260 220"
          role="img"
          aria-label={`通道旋转坐标 (${format(panel.values?.[0] ?? 0)}, ${format(panel.values?.[1] ?? 0)})`}
        >
          <line x1="20" y1="110" x2="245" y2="110" className="mp-axis" />
          <line x1="130" y1="15" x2="130" y2="205" className="mp-axis" />
          <circle cx="130" cy="110" r="85" className="mp-circle" />
          <path
            d={`M 210 110 A 80 80 0 ${(panel.values?.[2] ?? 0) > Math.PI ? 1 : 0} 0 ${130 + Math.cos(panel.values?.[2] ?? 0) * 80} ${110 - Math.sin(panel.values?.[2] ?? 0) * 80}`}
            className="mp-angle"
          />
          <line
            x1="130"
            y1="110"
            x2="215"
            y2="110"
            className="mp-original-vector"
          />
          <line
            x1="130"
            y1="110"
            x2={130 + (panel.values?.[0] ?? 0) * 85}
            y2={110 - (panel.values?.[1] ?? 0) * 85}
            className="mp-rotated-vector"
          />
          <circle
            cx={130 + (panel.values?.[0] ?? 0) * 85}
            cy={110 - (panel.values?.[1] ?? 0) * 85}
            r="5"
            className="mp-point"
          />
          <text x="234" y="128">
            x₀
          </text>
          <text x="137" y="25">
            x₁
          </text>
          <text x="22" y="207">
            虚线：原始 · 绿色：当前
          </text>
        </svg>
      )}
      {panel.kind === "plot" && (
        <>
          <svg
            className="mp-quant-plot"
            viewBox="0 0 300 190"
            role="img"
            aria-label="原始权重与量化重建权重对照"
          >
            <line x1="25" y1="95" x2="280" y2="95" className="mp-axis" />
            {(rows[0] ?? []).map((value, i) => {
              const bound = Math.max(1, ...rows.flat().map(Math.abs)),
                x = 30 + i * 48,
                originalY = 95 - (value / bound) * 70,
                reconstructedY = 95 - ((rows[1]?.[i] ?? 0) / bound) * 70;
              return (
                <g key={i}>
                  <line
                    x1={x}
                    x2={x}
                    y1={originalY}
                    y2={reconstructedY}
                    className="mp-quant-error"
                  />
                  <circle
                    cx={x}
                    cy={originalY}
                    r="5"
                    className="mp-original-point"
                  />
                  <rect
                    x={x - 4}
                    y={reconstructedY - 4}
                    width="8"
                    height="8"
                    className="mp-reconstructed-point"
                  />
                  <text x={x - 8} y="184">
                    w{i}
                  </text>
                </g>
              );
            })}
          </svg>
          <p className="mp-plot-key">
            <span>● 原始</span>
            <span>■ 重建</span>
            <span>连接线 = 误差</span>
          </p>
        </>
      )}
      {panel.note && <p className="mp-panel-note">{panel.note}</p>}
    </div>
  );
}

function LocalPlayer({
  definition,
  compact,
  onPlay,
}: {
  definition: MechanismDefinition;
  compact: boolean;
  onPlay?: () => void;
}) {
  const [params, setParams] = useState<Record<string, number>>(() =>
    Object.fromEntries(definition.parameters.map((p) => [p.key, p.initial])),
  );
  const ref = useRef<HTMLElement>(null);
  const frames = useMemo(
    () => buildMechanismFrames(definition.id, params),
    [definition.id, params],
  );
  const timeline = useTimeline({
    frameCount: frames.length,
    intervalMs: 1600,
    durationsMs: frames.slice(1).map(() => 1800),
    resetKey: `${definition.id}-${JSON.stringify(params)}`,
    surfaceRef: ref,
  });
  const { index, position, playing, mode, speed } = timeline;
  const current = frames[index];
  const shapeLesson = useMemo(
    () => deriveMechanismShapes(definition.id, index, params),
    [definition.id, index, params],
  );
  const changeParam = (key: string, value: number) => {
    timeline.reset();
    setParams((previous) => ({ ...previous, [key]: value }));
  };
  const start = () => {
    if (playing) timeline.pause();
    else {
      onPlay?.();
      timeline.play();
    }
  };
  const step = () => {
    onPlay?.();
    timeline.step();
  };

  const controls = (
    <>
      <div className="mp-controls">
        <div className="mp-control-buttons">
          <button
            type="button"
            onClick={start}
            aria-label={playing ? "暂停原理动图" : "播放原理动图"}
            className="mp-play-button"
          >
            {playing ? <Pause size={18} /> : <Play size={18} />}
            {playing ? "暂停" : "播放"}
          </button>
          <button
            type="button"
            onClick={step}
            aria-label="单步原理动图"
            disabled={index === frames.length - 1}
          >
            <SkipForward size={18} />
            <span>单步</span>
          </button>
          <button
            type="button"
            onClick={() => {
              timeline.reset();
            }}
            aria-label="重置原理动图"
          >
            <RotateCcw size={17} />
            <span>重置</span>
          </button>
        </div>
        <label className="mp-speed">
          速度
          <select
            value={speed}
            aria-label="原理动画速度"
            onChange={(e) => timeline.setSpeed(Number(e.target.value))}
          >
            <option value={0.25}>0.25×</option>
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </label>
      </div>
      <label className="mp-seek">
        <span>
          {mode === "continuous"
            ? `进度 ${Math.round(continuousProgress(position, frames.length) * 100)}%`
            : `步骤 ${index + 1} / ${frames.length}`}
        </span>
        <input
          type="range"
          aria-label={
            mode === "continuous" ? "连续原理动画进度" : "原理动画进度"
          }
          min="0"
          max={frames.length - 1}
          step={mode === "continuous" ? 0.001 : 1}
          value={mode === "continuous" ? position : index}
          onChange={(e) => {
            onPlay?.();
            timeline.seek(Number(e.target.value));
          }}
        />
      </label>
    </>
  );

  const exactSnapshot = (
    <>
      <div className="mp-formula" aria-label="当前运算">
        {current.formula}
      </div>
      <div
        className={`mp-scene mp-scene-${definition.view}`}
        key={`${definition.id}-${index}-${JSON.stringify(params)}`}
      >
        {current.panels.map((panel, i) => (
          <DataPanel key={`${panel.label}-${i}`} panel={panel} index={i} />
        ))}
      </div>
      {Object.keys(current.readouts).length > 0 && (
        <div className="mp-readouts">
          {Object.entries(current.readouts).map(([key, value]) => (
            <div key={key}>
              <span>{readoutLabels[key] ?? key}</span>
              <strong>{format(value)}</strong>
            </div>
          ))}
        </div>
      )}
      <div className="mp-caption" aria-live="polite" aria-atomic="true">
        <span className="mp-caption-icon">
          <ArrowRight size={18} aria-hidden="true" />
        </span>
        <div>
          <h4>{current.stage}</h4>
          <p>{current.caption}</p>
        </div>
      </div>
    </>
  );

  return (
    <section
      ref={ref}
      className={`mp-player ${compact ? "mp-compact" : ""} ${playing ? "mp-playing" : ""}`}
      data-testid="mechanism-player"
      data-mechanism-id={definition.id}
      data-frame={index}
      data-mode={mode}
      data-position={position}
      data-view={definition.view}
      aria-label={`${definition.title}原理动图`}
    >
      <header className="mp-header">
        <div className="mp-eyebrow">
          <span className="mp-live-dot" />
          {definition.category} / PRINCIPLE LAB
          <span className="mp-fixture-tag">小维度教学数据</span>
        </div>
        <h3>{definition.title}</h3>
        <p>{definition.subtitle}</p>
      </header>
      {definition.parameters.length > 0 && (
        <div className="mp-parameters">
          {definition.parameters.map((parameter) => (
            <label key={parameter.key} className="mp-parameter">
              <span>
                {parameter.label}
                <output>{format(params[parameter.key])}</output>
              </span>
              {parameter.choices ? (
                <select
                  aria-label={parameter.label}
                  value={params[parameter.key]}
                  onChange={(event) =>
                    changeParam(parameter.key, Number(event.target.value))
                  }
                >
                  {parameter.choices.map((value) => (
                    <option value={value} key={value}>
                      {value} KV 头
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="range"
                  min={parameter.min}
                  max={parameter.max}
                  step={parameter.step}
                  value={params[parameter.key]}
                  aria-label={parameter.label}
                  onChange={(event) =>
                    changeParam(parameter.key, Number(event.target.value))
                  }
                />
              )}
              <small>{parameter.meaning}</small>
            </label>
          ))}
        </div>
      )}
      <div className="mp-mode-switch" role="group" aria-label="原理演示模式">
        <button
          type="button"
          aria-label="分段讲解原理"
          aria-pressed={mode === "staged"}
          onClick={() => timeline.setMode("staged")}
        >
          分段讲解
        </button>
        <button
          type="button"
          aria-label="连续演示原理"
          aria-pressed={mode === "continuous"}
          onClick={() => timeline.setMode("continuous")}
        >
          连续演示
        </button>
        <span>
          {mode === "continuous"
            ? "观察整条数据通路 · 可连续拖动"
            : "逐步查看运算与精确数值"}
        </span>
      </div>
      <div className="mp-stage-rail" aria-label="原理步骤">
        {frames.map((f, i) => (
          <button
            key={f.stage}
            type="button"
            className={i === index ? "mp-active-stage" : ""}
            aria-current={i === index ? "step" : undefined}
            onClick={() => {
              onPlay?.();
              timeline.seek(i);
            }}
          >
            <span>{String(i + 1).padStart(2, "0")}</span>
            {f.stage}
            <ChevronRight size={13} aria-hidden="true" />
          </button>
        ))}
      </div>
      {mode === "continuous" && controls}
      <div className="mp-board" data-testid="mechanism-board">
        <div className="mp-board-status">
          <span>
            STEP {String(index + 1).padStart(2, "0")} /{" "}
            {String(frames.length).padStart(2, "0")}
          </span>
          <span>{current.stage}</span>
        </div>
        {mode === "continuous" && (
          <ContinuousMechanismScene
            definition={definition}
            frames={frames}
            position={position}
          />
        )}
        {mode === "continuous" ? (
          <details className="mp-continuous-details">
            <summary>查看当前关键帧的公式、精确读数与解释</summary>
            {exactSnapshot}
          </details>
        ) : (
          exactSnapshot
        )}
      </div>
      <ShapeTeachingPanel lesson={shapeLesson} />
      <MechanismModelComparison id={definition.id} params={params} />
      {mode === "staged" && controls}
      {definition.source && (
        <p className="mp-source">
          算法依据：
          <a href={definition.source.url} target="_blank" rel="noreferrer">
            {definition.source.title} ↗
          </a>
        </p>
      )}
      <p className="mp-assumption">
        每一步都可暂停、回看。数值来自小数组计算与固定机制 fixture；不使用真实
        Qwen 权重，不代表 GPU 实测。
      </p>
    </section>
  );
}

export default function MechanismPlayer({
  id,
  compact = false,
  onPlay,
}: {
  id: string;
  compact?: boolean;
  onPlay?: () => void;
}) {
  const definition = mechanisms.find((m) => m.id === id);
  if (!definition)
    return (
      <section
        className="mp-player mp-unknown"
        data-testid="mechanism-player"
        data-mechanism-id={id}
      >
        <h3>尚未收录这个原理</h3>
        <p>请选择原理目录中已有的模块。</p>
      </section>
    );
  return (
    <LocalPlayer
      key={id}
      definition={definition}
      compact={compact}
      onPlay={onPlay}
    />
  );
}
