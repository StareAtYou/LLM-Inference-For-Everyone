import { useMemo, useState, useEffect, useRef } from "react";
import { Link, useSearchParams } from "react-router";
import {
  ArrowUpRight,
  Search,
  ChevronLeft,
  ChevronRight,
  X,
  Play,
  Pause,
  SkipForward,
  RotateCcw,
} from "lucide-react";
import { models, getModel } from "../content/models";
import { scenarios } from "../content/scenarios";
import {
  buildInferenceTrace,
  type TraceFrame,
  type TraceConfig,
} from "../simulation/inferenceTrace";
import { parseRouteState } from "../lib/urlState";
import { useTimeline } from "../hooks/useTimeline";
import { createTimeline, positionAtElapsed } from "../simulation/timeline";
import ContinuousJourneyScene from "../visualizations/ContinuousJourneyScene";
import PlaybackControls from "../components/PlaybackControls";
import ParameterControl from "../components/ParameterControl";
import DepthSwitch, { useLearningDepth } from "../components/DepthSwitch";
import SourceNote from "../components/SourceNote";
import InferenceFlow, {
  requestColors,
  overviewStages,
} from "../visualizations/InferenceFlow";
import TensorInspector from "../visualizations/TensorInspector";
import { mechanisms } from "../mechanisms/catalog";
import MechanismPlayer from "../mechanisms/MechanismPlayer";
import "../styles/workbench.css";
type Detail = "overview" | "layer" | "operator";
const statusLabels = {
  waiting: "等待中",
  prefill: "阅读提示",
  decode: "逐步生成",
  done: "已完成",
};
const chapters = [
  { label: "接收与编码", stage: "receive" },
  { label: "调度与入槽", stage: "schedule" },
  { label: "进入网络", stage: "embedding" },
  { label: "预测与采样", stage: "lm-head" },
  { label: "输出与反馈", stage: "emit" },
  { label: "结束与释放", stage: "release" },
];
function filterFrames(frames: TraceFrame[], detail: Detail, layers: number) {
  return detail === "operator"
    ? frames
    : frames.filter(
        (f) =>
          overviewStages.includes(f.stage) ||
          (detail === "layer" &&
            ["linear", "attention", "ffn", "moe"].includes(f.stage)) ||
          (detail === "overview" &&
            (f.layer === 0 || f.layer === layers - 1) &&
            ["linear", "attention", "ffn", "moe"].includes(f.stage)),
      );
}
export default function Pipeline() {
  const [params, setParams] = useSearchParams(),
    route = parseRouteState(params),
    model = getModel(route.modelId),
    depth = useLearningDepth();
  const view = params.get("view") === "mechanisms" ? "mechanisms" : "journey";
  const mode = params.get("mode") === "batch" ? "batch" : "single";
  const [scenarioId, setScenario] = useState("sky"),
    [capacity, setCapacity] = useState(2),
    [outputLimit, setLimit] = useState(3);
  const [temperature, setTemperature] = useState(0),
    [topK, setTopK] = useState(6),
    [topP, setTopP] = useState(1);
  const [detail, setDetail] = useState<Detail>(
    depth === "expert"
      ? "operator"
      : depth === "advanced"
        ? "layer"
        : "overview",
  );
  const [requestId, setRequest] = useState("R1"),
    [moduleId, setModule] = useState<string | null>(null),
    [query, setQuery] = useState("");
  const returnFocus = useRef<HTMLElement | null>(null),
    flowSurface = useRef<HTMLElement | null>(null),
    modulePanel = useRef<HTMLDivElement>(null);
  const config: TraceConfig = {
    modelId: model.id,
    scenarioId,
    mode,
    capacity,
    temperature,
    topK,
    topP,
    outputLimit,
  };
  const resetKey = JSON.stringify(config);
  const trace = useMemo(
    () => buildInferenceTrace(config),
    [
      model.id,
      scenarioId,
      mode,
      capacity,
      temperature,
      topK,
      topP,
      outputLimit,
    ],
  );
  const frames = useMemo(
    () => filterFrames(trace, detail, model.layerTypes.length),
    [trace, detail, model.layerTypes.length],
  );
  const keyframes = useMemo(() => frames.map((f) => f.index), [frames]);
  const durations = useMemo(
    () =>
      trace
        .slice(0, -1)
        .map((f) =>
          f.layer !== null
            ? 220
            : [
                  "receive",
                  "tokenize",
                  "schedule",
                  "embedding",
                  "emit",
                  "feedback",
                  "finish",
                  "release",
                ].includes(f.stage)
              ? 900
              : 650,
        ),
    [trace],
  );
  const timing = useMemo(
    () =>
      createTimeline({
        frameCount: trace.length,
        intervalMs: 800,
        durationsMs: durations,
        keyframes,
      }),
    [trace.length, durations, keyframes],
  );
  const player = useTimeline({
    frameCount: trace.length,
    intervalMs: 800,
    resetKey,
    surfaceRef: flowSurface,
    surfaceKey: view,
    durationsMs: durations,
    keyframes,
  });
  const frame = trace[player.index];
  const visibleIndex = Math.max(
    0,
    frames.length -
      1 -
      [...frames].reverse().findIndex((f) => f.index <= player.index),
  );
  const snapshotPlayer = {
    ...player,
    index: visibleIndex,
    seek: (i: number) =>
      player.seek(frames[Math.max(0, Math.min(i, frames.length - 1))].index),
  };
  const continuous = player.mode === "continuous";
  useEffect(() => {
    setRequest("R1");
    setModule(null);
  }, [model.id, mode, scenarioId]);
  useEffect(() => {
    if (moduleId) {
      modulePanel.current?.focus();
      modulePanel.current?.scrollIntoView({
        block: "nearest",
        behavior: "instant",
      });
    }
  }, [moduleId, view]);
  function openModule(id: string, element: HTMLElement) {
    player.pause();
    returnFocus.current = element;
    setModule(id);
  }
  function closeModule() {
    setModule(null);
    returnFocus.current?.focus();
  }
  function switchView(next: "journey" | "mechanisms") {
    player.pause();
    setModule(null);
    const q = new URLSearchParams(params);
    q.set("view", next);
    setParams(q, { replace: true });
  }
  function switchMode(next: "single" | "batch") {
    const q = new URLSearchParams(params);
    q.set("mode", next);
    setParams(q, { replace: true });
  }
  const found = mechanisms.filter((m) =>
    [m.id, m.title, m.subtitle, m.category]
      .join(" ")
      .toLowerCase()
      .includes(query.toLowerCase().trim()),
  );
  const module = mechanisms.find((m) => m.id === moduleId);
  const currentRequest =
    frame.requests.find((r) => r.id === requestId) ?? frame.requests[0];
  const probabilityFrame = useMemo(
    () =>
      frame.probabilities.length && frame.requestIds.includes(currentRequest.id)
        ? frame
        : trace
            .slice(0, frame.index + 1)
            .reverse()
            .find(
              (f) =>
                f.requestIds.includes(currentRequest.id) &&
                f.probabilities.length,
            ),
    [frame, currentRequest.id, trace],
  );
  function nextRound() {
    const i = trace.findIndex(
      (f) => f.index > player.index && f.tick > frame.tick,
    );
    player.seek(i < 0 ? trace.length - 1 : i);
  }
  function selectLayer(n: number) {
    const i = trace.findIndex((f) => f.layer === n && f.tick === frame.tick);
    setDetail("operator");
    player.seek(Math.max(0, i));
  }
  const timeLabel = (ms: number) =>
    `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}`;
  return (
    <div className="page-width content-bottom wf-page">
      <header className="wf-page-heading">
        <div>
          <div className="eyebrow">THE INFERENCE WORKBENCH / 02</div>
          <h1>
            看清每一步，
            <br />
            跟随数据走完推理。
          </h1>
          <p>从请求入队到缓存释放。展开计算，看见数字与 shape 怎样变化。</p>
        </div>
        <DepthSwitch />
      </header>
      {route.warning && <p className="notice">{route.warning}</p>}
      <nav className="wf-view-tabs" aria-label="工作台视图">
        <button
          aria-label="完整推理"
          aria-pressed={view === "journey"}
          className={view === "journey" ? "active" : ""}
          onClick={() => switchView("journey")}
        >
          完整推理<span>单请求 / 批请求</span>
        </button>
        <button
          aria-label="原理动图"
          aria-pressed={view === "mechanisms"}
          className={view === "mechanisms" ? "active" : ""}
          onClick={() => switchView("mechanisms")}
        >
          原理动图<span>{mechanisms.length} 个可展开的机制</span>
        </button>
        <Link className="wf-source-badge" to={"/distributed?depth=" + depth}>
          展开多 GPU 与通信 →
        </Link>
      </nav>
      {view === "journey" ? (
        <>
          <p className="wf-depth-guide">
            {depth === "beginner"
              ? "入门建议：从旅程概览开始，跟随有颜色的请求。点击任意模块，单独播放它的原理。"
              : depth === "advanced"
                ? "深入建议：逐层观察 Full Attention 与线性状态的交替，选中一个请求核对张量和缓存。"
                : "精通建议：逐算子追踪 RMSNorm、投影与残差，用张量数值核对计算，比较并发请求的独立状态。"}
          </p>
          <div className="wf-config">
            <label>
              模型
              <select
                aria-label="选择模型"
                value={model.id}
                onChange={(e) => {
                  const q = new URLSearchParams(params);
                  q.set("model", e.target.value);
                  setParams(q, { replace: true });
                }}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </label>
            <div className="wf-mode">
              <span>请求组织</span>
              <div className="segment">
                <button
                  aria-pressed={mode === "single"}
                  className={mode === "single" ? "selected" : ""}
                  onClick={() => switchMode("single")}
                >
                  单请求
                </button>
                <button
                  aria-pressed={mode === "batch"}
                  className={mode === "batch" ? "selected" : ""}
                  onClick={() => switchMode("batch")}
                >
                  批请求
                </button>
              </div>
            </div>
            <label>
              教学场景
              <select
                aria-label="选择教学场景"
                value={scenarioId}
                onChange={(e) => setScenario(e.target.value)}
              >
                {scenarios.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </label>
            {mode === "batch" && (
              <label>
                并发槽位
                <select
                  aria-label="并发槽位"
                  value={capacity}
                  onChange={(e) => setCapacity(Number(e.target.value))}
                >
                  {[1, 2, 3].map((n) => (
                    <option value={n} key={n}>
                      {n} 个
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label>
              输出上限
              <select
                aria-label="教学输出上限"
                value={outputLimit}
                onChange={(e) => setLimit(Number(e.target.value))}
              >
                {[1, 2, 3, 4].map((n) => (
                  <option key={n} value={n}>
                    {n} Token
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="wf-context-note">
            <span className="mono">READ THE FLOW</span>
            <p>
              {mode === "single"
                ? "追踪 R1：整段提示先做 Prefill，首个输出被反馈后进入 Decode，直到结束释放。"
                : "三条独立请求，R1 / R2 第 0 轮到达、R3 第 1 轮到达。空槽补位，Prefill 和 Decode 可在同轮混合；R2 最多输出 2 个 Token。"}
            </p>
          </div>
          <div className="wf-request-lanes" aria-label="请求与状态">
            {frame.requests.map((r, i) => (
              <button
                type="button"
                key={r.id}
                data-testid={"request-" + r.id}
                aria-pressed={currentRequest.id === r.id}
                className={
                  "wf-request " +
                  (currentRequest.id === r.id ? "selected " : "") +
                  (frame.requestIds.includes(r.id) ? "computing" : "")
                }
                style={
                  { "--request-color": requestColors[i] } as React.CSSProperties
                }
                onClick={() => setRequest(r.id)}
              >
                <div>
                  <strong>{r.id}</strong>
                  <span className="wf-request-status">
                    {statusLabels[r.status]}
                  </span>
                  <i className="wf-request-dot" />
                </div>
                <p>{r.prompt}</p>
                <span className="wf-request-phase">{r.phase}</span>
                <div className="wf-request-numbers">
                  <span>输入 {r.inputTokens.length}</span>
                  <span>输出 {r.outputTokens.length}</span>
                  <span>缓存 {r.cacheTokens}</span>
                </div>
                <small>
                  {r.blocks.length
                    ? `物理块 ${r.blocks.map((n) => "P" + n).join(" · ")}`
                    : "未持有物理块"}
                </small>
              </button>
            ))}
          </div>
          <div className="wf-layout">
            <section
              ref={flowSurface}
              className="wf-flow-board"
              data-mode={player.mode}
              data-position={player.position}
              data-playback-surface
              aria-label="全过程数据流"
            >
              <div className="wf-board-toolbar">
                <span className="mono">
                  ROUND {String(frame.tick).padStart(2, "0")} /{" "}
                  {mode === "single" ? "SINGLE REQUEST" : "CONTINUOUS BATCH"}
                </span>
                <div>
                  <label>
                    计算粒度
                    <select
                      aria-label="计算粒度"
                      value={detail}
                      onChange={(e) => setDetail(e.target.value as Detail)}
                    >
                      <option value="overview">旅程概览</option>
                      <option value="layer">逐层观察</option>
                      <option value="operator">逐算子计算</option>
                    </select>
                  </label>
                  <label>
                    速度
                    <select
                      aria-label="播放速度"
                      value={player.speed}
                      onChange={(e) => player.setSpeed(Number(e.target.value))}
                    >
                      {[0.25, 0.5, 1, 2, 4, 8, 16].map((n) => (
                        <option value={n} key={n}>
                          {n}×
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>
              <div
                className="wf-presentation-selector"
                role="group"
                aria-label="全过程演示方式"
              >
                <div className="segment">
                  <button
                    className={!continuous ? "selected" : ""}
                    aria-label="分段讲解全过程"
                    aria-pressed={!continuous}
                    onClick={() => player.setMode("staged")}
                  >
                    分段讲解<span>停在关键帧</span>
                  </button>
                  <button
                    className={continuous ? "selected" : ""}
                    aria-label="连续演示全过程"
                    aria-pressed={continuous}
                    onClick={() => player.setMode("continuous")}
                  >
                    连续演示<span>跟随整条时间轴</span>
                  </button>
                </div>
                <p>
                  {continuous
                    ? "连续经过所有层与算子；暂停后可展开任意模块。"
                    : "关键帧采样自同一条完整轨迹，可切换为连续演示。"}
                </p>
              </div>
              {continuous && (
                <div className="wf-continuous-controls">
                  <div>
                    <button
                      className="button primary"
                      onClick={player.playing ? player.pause : player.play}
                    >
                      {player.playing ? (
                        <Pause size={16} />
                      ) : (
                        <Play size={16} />
                      )}{" "}
                      {player.playing ? "暂停" : "播放"}
                    </button>
                    <button
                      className="control-button"
                      onClick={player.step}
                      disabled={player.index === trace.length - 1}
                    >
                      <SkipForward size={16} />
                      单步
                    </button>
                    <button className="control-button" onClick={player.reset}>
                      <RotateCcw size={16} />
                      重置
                    </button>
                    <span className="mono">
                      {timeLabel(player.elapsedMs)} /{" "}
                      {timeLabel(player.durationMs)} · {player.speed}×
                    </span>
                  </div>
                  <input
                    type="range"
                    aria-label="连续演示进度"
                    min={0}
                    max={player.durationMs}
                    step={1}
                    value={player.elapsedMs}
                    onChange={(e) =>
                      player.seek(
                        positionAtElapsed(timing, Number(e.target.value)),
                      )
                    }
                  />
                  <small>
                    动画时间轴 · 与 GPU
                    耗时无关。切换方式保留当前位置，单步定位下一个计算事件。
                  </small>
                </div>
              )}
              <div className="wf-live-stage">
                <span className="wf-live-dot" />
                <div>
                  <h2 data-testid="trace-stage">{frame.label}</h2>
                  <p>{frame.description}</p>
                </div>
                <span className="wf-active-requests">
                  {frame.requestIds.join(" + ") || "—"}
                </span>
              </div>
              <div className="stage-rail wf-chapters">
                {chapters.map((c, i) => (
                  <button
                    key={c.stage}
                    aria-label={"跳到" + c.label}
                    onClick={() =>
                      player.seek(
                        c.stage === "release"
                          ? trace.length - 1
                          : Math.max(
                              0,
                              trace.findIndex((f) => f.stage === c.stage),
                            ),
                      )
                    }
                  >
                    <small className="mono">0{i + 1}</small>
                    {c.label}
                  </button>
                ))}
              </div>
              {continuous && (
                <ContinuousJourneyScene
                  trace={trace}
                  frame={frame}
                  fraction={player.fraction}
                  position={player.position}
                  model={model}
                  requestId={currentRequest.id}
                  onSeek={player.seek}
                  onInspect={openModule}
                />
              )}
              {continuous ? (
                <details className="wf-static-reference">
                  <summary>展开算子地图与模块说明</summary>
                  <InferenceFlow
                    frame={frame}
                    model={model}
                    playing={false}
                    onInspect={openModule}
                  />
                </details>
              ) : (
                <InferenceFlow
                  frame={frame}
                  model={model}
                  playing={player.playing}
                  onInspect={openModule}
                  presentation={player.mode}
                  fraction={player.fraction}
                />
              )}
              <div className="wf-layer-navigator">
                <label>
                  定位网络层
                  <select
                    aria-label="定位网络层"
                    value={frame.layer ?? 0}
                    onChange={(e) => selectLayer(Number(e.target.value))}
                  >
                    {model.layerTypes.map((type, i) => (
                      <option key={i} value={i}>
                        第 {i + 1} 层 ·{" "}
                        {type === "full_attention"
                          ? "Full Attention"
                          : "Gated DeltaNet"}
                      </option>
                    ))}
                  </select>
                </label>
                <button className="control-button" onClick={nextRound}>
                  下一轮 <ChevronRight size={15} />
                </button>
                <span>
                  {detail === "overview"
                    ? "分段模式取首末层关键帧；连续模式经过全部层。"
                    : detail === "layer"
                      ? "逐层显示注意力与 FFN；算子视图提供完整中间值。"
                      : "每个算子都有独立事件；可拖动进度或跳层。"}
                </span>
              </div>
              <div className="wf-player">
                {!continuous && (
                  <PlaybackControls
                    player={snapshotPlayer}
                    count={frames.length}
                  />
                )}
                <div className="wf-progress-note">
                  <span>
                    {continuous
                      ? `连续轨迹 ${player.index + 1} / ${trace.length} · 当前传递 ${Math.round(player.fraction * 100)}%`
                      : `关键帧 ${visibleIndex + 1} / ${frames.length}`}
                  </span>
                  <span>完整轨迹 {trace.length} 个计算事件 · 非耗时指标</span>
                </div>
              </div>
            </section>
            {continuous ? (
              <details className="wf-continuous-details">
                <summary>展开当前计算事件的完整数据详情</summary>
                <TensorInspector
                  frame={frame}
                  requestId={currentRequest.id}
                  model={model}
                />
              </details>
            ) : (
              <TensorInspector
                frame={frame}
                requestId={currentRequest.id}
                model={model}
              />
            )}
          </div>
          {moduleId && (
            <div
              className="wf-inline-module"
              ref={modulePanel}
              tabIndex={-1}
              role="region"
              aria-label="展开的模块原理"
            >
              <div className="wf-expanded-heading">
                <div>
                  <div className="eyebrow">INSIDE THE OPERATION</div>
                  <h2>{module?.title}</h2>
                  <p>
                    全过程已暂停。这里可以独立播放、逐步理解，再回到同一帧。
                  </p>
                </div>
                <button className="button secondary" onClick={closeModule}>
                  <ChevronLeft size={16} />
                  返回全过程
                </button>
              </div>
              <MechanismPlayer id={moduleId} onPlay={player.pause} />
            </div>
          )}
          <section className="wf-result-layout">
            <div className="wf-output-panel">
              <div className="eyebrow">THE TOKEN JOURNEY</div>
              <h2>输入与输出，始终对应。</h2>
              <div className="wf-input-pieces">
                <span>{currentRequest.id} / 教学 Token IDs</span>
                <div>
                  {currentRequest.inputTokens.map((t, i) => (
                    <span key={i}>
                      <small>{currentRequest.inputIds[i]}</small>
                      {t}
                    </span>
                  ))}
                </div>
              </div>
              <div data-testid="trace-output" className="wf-output-lines">
                {frame.requests.map((r, i) => (
                  <div key={r.id}>
                    <strong style={{ color: requestColors[i] }}>{r.id}</strong>
                    <div>
                      {r.outputTokens.length ? (
                        r.outputTokens.map((t, j) => (
                          <span className="wf-generated-token" key={j}>
                            <small>{j + 1}</small>
                            {t}
                          </span>
                        ))
                      ) : (
                        <span className="wf-output-empty">尚未产生输出</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <p>
                输出来自小维度网络的 logits 和采样。它用于理解生成机制，不是
                Qwen 的语言回答；停止依据教学输出上限。
              </p>
            </div>
            <aside className="wf-sampling-panel">
              <div className="eyebrow">PROBABILITIES / {currentRequest.id}</div>
              <h2>数字如何成为下一个 Token？</h2>
              <ParameterControl
                label="温度 Temperature"
                value={temperature}
                min={0}
                max={2}
                step={0.1}
                onChange={setTemperature}
              />
              <div className="wf-sampling-pair">
                <ParameterControl
                  label="Top-k"
                  value={topK}
                  min={1}
                  max={6}
                  onChange={setTopK}
                />
                <ParameterControl
                  label="Top-p"
                  value={topP}
                  min={0.1}
                  max={1}
                  step={0.05}
                  onChange={setTopP}
                />
              </div>
              <div className="probability-bars">
                {frame.candidateLabels.map((label, i) => (
                  <div key={label}>
                    <span>{label}</span>
                    <div>
                      <i
                        style={{
                          width:
                            (probabilityFrame?.probabilities[i] ?? 0) * 100 +
                            "%",
                        }}
                      />
                    </div>
                    <span className="mono">
                      {(
                        (probabilityFrame?.probabilities[i] ?? 0) * 100
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                ))}
              </div>
              <p className="small-note">
                {probabilityFrame
                  ? "当前或最近一次 " +
                    currentRequest.id +
                    " 的采样分布；温度 0 使用 argmax，非零温度用固定分位 0.61。"
                  : "经过网络和 LM Head 后，才会出现该请求的真实教学概率。"}
                改变参数重置整条轨迹。
              </p>
            </aside>
          </section>
          <div className="wf-assumptions">
            <div className="eyebrow">WHAT THIS SIMULATION MEANS</div>
            <p>
              真实 Qwen 层数、混合层顺序与逻辑维度来自官方配置。数值计算缩小为
              hidden=4、2 个 Q 头 / 1 个 KV
              头、head_dim=2，使用确定性教学权重；MoE 缩小为 3 个路由专家、Top-2
              和 1 个共享专家。批次按请求 packed，无 padding；轮数和事件数不代表
              GPU 延迟。每个请求、每层的历史状态独立。
            </p>
            <Link className="text-link" to={"/models?model=" + model.id}>
              核对完整模型结构 <ArrowUpRight size={15} />
            </Link>
          </div>
          <SourceNote sources={model.sources} />
        </>
      ) : (
        <section className="wf-mechanism-library">
          <div className="wf-library-heading">
            <div>
              <h2>一个原理，一幅会动的解释。</h2>
              <p>
                先看输入，逐步执行，再核对输出。每幅图都有自己的运算与状态。
              </p>
            </div>
            <label className="search-field">
              <Search size={18} />
              <input
                type="search"
                aria-label="搜索原理动图"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="搜索 Attention、RMSNorm、缓存…"
              />
            </label>
          </div>
          {moduleId && (
            <div
              className="wf-library-player"
              ref={modulePanel}
              tabIndex={-1}
              role="region"
              aria-label="展开的原理动图"
            >
              <div className="wf-expanded-heading">
                <h2>{module?.title}</h2>
                <button
                  className="control-button"
                  onClick={closeModule}
                  aria-label="关闭原理动图"
                >
                  <X size={16} />
                  收起
                </button>
              </div>
              <MechanismPlayer key={moduleId} id={moduleId} />
            </div>
          )}
          <div className="wf-mechanism-grid">
            {found.map((m, i) => (
              <button
                key={m.id}
                className={moduleId === m.id ? "selected" : ""}
                aria-label={"打开 " + m.title + " 原理动图"}
                onClick={(e) => openModule(m.id, e.currentTarget)}
              >
                <span className="mono">
                  {String(i + 1).padStart(2, "0")} / {m.category}
                </span>
                <strong>{m.title}</strong>
                <p>{m.subtitle}</p>
                <span>
                  逐步展开 <ArrowUpRight size={14} />
                </span>
              </button>
            ))}
          </div>
          {!found.length && (
            <p className="notice">
              没有匹配的原理。可以搜索英文术语或中文名称。
            </p>
          )}
        </section>
      )}
    </div>
  );
}
