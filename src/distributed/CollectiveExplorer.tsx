import { useId, useMemo, useRef, useState } from "react";
import { useInRouterContext, useLocation, useSearchParams } from "react-router";
import { Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { useTimeline } from "../hooks/useTimeline";
import {
  buildCollectiveDemo,
  collectiveOperations,
  collectiveSnapshot,
  collectiveSources,
  runCollective,
  type CollectiveDemo,
  type CollectiveOperation,
  type CollectiveTransfer,
} from "./collectives";
import "./collectives.css";

type Point = { x: number; y: number };
const vector = (values: number[]) => `[${values.join(", ")}]`;
const COLORS = ["#6e48d7", "#167c6f", "#b36b2d", "#5362a6"];
const rootOperations: CollectiveOperation[] = [
  "broadcast",
  "reduce",
  "gather",
  "scatter",
];
function nodePoint(rank: number | "sum", ranks: number): Point {
  if (rank === "sum") return { x: 450, y: 310 };
  return {
    x: rank % 2 === 0 ? 170 : 730,
    y: ranks === 2 ? 310 : rank < 2 ? 125 : 495,
  };
}
function pathFor(transfer: CollectiveTransfer, ranks: number) {
  const a = nodePoint(transfer.from, ranks),
    b = nodePoint(transfer.to, ranks);
  const sourceX =
    transfer.from === "sum"
      ? a.x + (b.x < a.x ? -58 : 58)
      : a.x + (a.x < 450 ? 124 : -124);
  const destinationX =
    transfer.to === "sum"
      ? b.x + (a.x < b.x ? -58 : 58)
      : b.x + (b.x < 450 ? 124 : -124);
  const offset = ((Number(transfer.id.split(":").at(-1)) % 4) - 1.5) * 18;
  const start = {
    x: sourceX,
    y: a.y + (transfer.from === transfer.to ? -24 : offset),
  };
  const end = {
    x: destinationX,
    y: b.y + (transfer.from === transfer.to ? 24 : offset),
  };
  const loopX = a.x < 450 ? sourceX + 105 : sourceX - 105;
  const control1 = {
    x: transfer.from === transfer.to ? loopX : 450 + offset,
    y: start.y,
  };
  const control2 = {
    x: transfer.from === transfer.to ? loopX : 450 + offset,
    y: end.y,
  };
  return {
    start,
    end,
    control1,
    control2,
    d: `M ${start.x} ${start.y} C ${control1.x} ${control1.y}, ${control2.x} ${control2.y}, ${end.x} ${end.y}`,
  };
}
function along(path: ReturnType<typeof pathFor>, t: number): Point {
  const u = 1 - t;
  return {
    x:
      u ** 3 * path.start.x +
      3 * u ** 2 * t * path.control1.x +
      3 * u * t ** 2 * path.control2.x +
      t ** 3 * path.end.x,
    y:
      u ** 3 * path.start.y +
      3 * u ** 2 * t * path.control1.y +
      3 * u * t ** 2 * path.control2.y +
      t ** 3 * path.end.y,
  };
}
function CollectiveScene({
  demo,
  position,
  selectedRank,
  focus,
}: {
  demo: CollectiveDemo;
  position: number;
  selectedRank: number;
  focus: boolean;
}) {
  const marker = `ce-arrow-${useId().replaceAll(":", "")}`;
  const snapshot = collectiveSnapshot(demo, position);
  const reducing = ["all-reduce", "reduce-scatter", "reduce"].includes(
    demo.operation,
  );
  const related = (transfer: CollectiveTransfer) =>
    !focus || transfer.from === selectedRank || transfer.to === selectedRank;
  return (
    <div
      className="ce-scene-scroll"
      tabIndex={0}
      role="region"
      aria-label="通信数据路径，可横向滚动查看"
    >
      <svg
        className="ce-scene"
        viewBox={demo.ranks === 2 ? "0 200 900 220" : "0 0 900 620"}
        role="img"
        aria-label={`${demo.ranks} 个 rank 的 ${demo.operation} 缓冲区传输，当前进度 ${position.toFixed(2)}`}
      >
        <defs>
          <marker
            id={marker}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
          </marker>
        </defs>
        {demo.transfers.map((transfer) => (
          <path
            key={transfer.id}
            className={
              related(transfer) ? "ce-route ce-route-focus" : "ce-route"
            }
            d={pathFor(transfer, demo.ranks).d}
            markerEnd={`url(#${marker})`}
          />
        ))}
        {reducing && (
          <g className="ce-sum" transform="translate(450 310)">
            <rect x="-58" y="-58" width="116" height="116" rx="24" />
            <text textAnchor="middle" y="-20">
              逐位置 SUM
            </text>
            <text textAnchor="middle" y="7">
              {position >= 2 ? "贡献已汇合" : "等待贡献"}
            </text>
            <text className="ce-svg-small" textAnchor="middle" y="31">
              教学运算节点
            </text>
          </g>
        )}
        {demo.inputs.map((input, rank) => {
          const point = nodePoint(rank, demo.ranks),
            output = snapshot.outputs[rank];
          return (
            <g
              key={rank}
              className={`ce-rank ${rank === selectedRank ? "ce-rank-selected" : ""}`}
              transform={`translate(${point.x} ${point.y})`}
            >
              <rect x="-124" y="-91" width="248" height="182" rx="20" />
              <text
                className="ce-rank-title"
                x="-106"
                y="-65"
                fill={COLORS[rank]}
              >
                rank {rank} · GPU {rank * 2}
                {rootOperations.includes(demo.operation) &&
                rank === demo.options.root
                  ? " · root"
                  : ""}
              </text>
              <text className="ce-svg-small" x="-106" y="-41">
                发送：{input ? `[${input.length}]` : "无发送缓冲区"}
              </text>
              <text className="ce-svg-buffer" x="-106" y="-20">
                {input ? vector(input.slice(0, 4)) : "—"}
              </text>
              {input && input.length > 4 && (
                <text className="ce-svg-buffer" x="-106" y="0">
                  {vector(input.slice(4))}
                </text>
              )}
              <line x1="-106" x2="106" y1="13" y2="13" />
              <text className="ce-svg-small" x="-106" y="34">
                接收：
                {demo.outputs[rank]
                  ? `[${demo.outputs[rank]!.length}]`
                  : "无结果"}
              </text>
              <text className="ce-svg-buffer" x="-106" y="56">
                {output
                  ? vector(output.slice(0, 4))
                  : demo.outputs[rank]
                    ? "等待数据到达"
                    : "此 rank 不接收结果"}
              </text>
              {output && output.length > 4 && (
                <text className="ce-svg-buffer" x="-106" y="77">
                  {vector(output.slice(4))}
                </text>
              )}
            </g>
          );
        })}
        {snapshot.transfers.map((transfer) => {
          const path = pathFor(transfer, demo.ranks),
            point = along(path, transfer.progress);
          const known = transfer.from !== "sum" || position >= 2;
          const height = transfer.values.length > 4 ? 64 : 48;
          return (
            <g
              key={transfer.id}
              className="ce-packet"
              data-testid="collective-packet"
              data-object-id={transfer.id}
              data-progress={transfer.progress.toFixed(4)}
              data-from={transfer.from}
              data-to={transfer.to}
              transform={`translate(${point.x.toFixed(3)} ${point.y.toFixed(3)})`}
              opacity={transfer.active && related(transfer) ? 1 : 0}
              aria-hidden="true"
            >
              <rect
                x="-81"
                y={-height / 2}
                width="162"
                height={height}
                rx="9"
                fill={
                  typeof transfer.from === "number"
                    ? COLORS[transfer.from]
                    : "#3b3a43"
                }
              />
              <text
                textAnchor="middle"
                className="ce-packet-label"
                y={-height / 2 + 15}
              >
                {transfer.from === transfer.to
                  ? "本地切片 / 复制"
                  : transfer.label}
              </text>
              <text
                textAnchor="middle"
                className="ce-packet-values"
                y={-height / 2 + 33}
              >
                {known ? vector(transfer.values.slice(0, 4)) : "待归约"}
              </text>
              {transfer.values.length > 4 && (
                <text
                  textAnchor="middle"
                  className="ce-packet-values"
                  y={-height / 2 + 50}
                >
                  {known ? vector(transfer.values.slice(4)) : "—"}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function LocalCollectiveExplorer({
  initialOperation = "all-reduce",
  onOperationChange,
  controlledOperation,
}: {
  initialOperation?: CollectiveOperation;
  onOperationChange?: (operation: CollectiveOperation) => void;
  controlledOperation?: CollectiveOperation;
}) {
  const [localOperation, setOperation] =
    useState<CollectiveOperation>(initialOperation);
  const operation = controlledOperation ?? localOperation;
  const [ranks, setRanks] = useState<2 | 4>(2);
  const [root, setRoot] = useState(0);
  const [sender, setSender] = useState(0);
  const [receiver, setReceiver] = useState(1);
  const [selectedRank, setSelectedRank] = useState(0);
  const [focus, setFocus] = useState(true);
  const surfaceRef = useRef<HTMLElement>(null);
  const demo = useMemo(
    () => buildCollectiveDemo(operation, ranks, { root, sender, receiver }),
    [operation, ranks, root, sender, receiver],
  );
  const timeline = useTimeline({
    frameCount: demo.steps.length,
    intervalMs: 1800,
    keyframes: [0, 1, 2, 3, 4],
    resetKey: `${operation}/${ranks}/${root}/${sender}/${receiver}`,
    surfaceRef,
  });
  const definition = collectiveOperations.find((op) => op.id === operation)!;
  const snapshot = collectiveSnapshot(demo, timeline.position);
  const change = (update: () => void) => {
    timeline.reset();
    update();
  };
  const selectedOutput = snapshot.outputs[selectedRank];
  const selectedInput = demo.inputs[selectedRank];
  const receive = demo.outputs[selectedRank];
  const gatheredReduction =
    operation === "all-reduce" && timeline.position >= 3
      ? runCollective(
          "all-gather",
          runCollective("reduce-scatter", demo.inputs),
        )
      : undefined;
  return (
    <section
      className="ce-explorer"
      ref={surfaceRef}
      data-testid="collective-explorer"
      data-operation={operation}
      data-position={timeline.position.toFixed(4)}
    >
      <div className="ce-heading">
        <div>
          <p className="ce-eyebrow">COMMUNICATION · 数据去哪里</p>
          <h2>通信原语</h2>
        </div>
        <span className="ce-tag">整数缓冲区 · 教学执行</span>
      </div>
      <div className="ce-operation-list" role="group" aria-label="选择通信原语">
        {collectiveOperations.map((op) => (
          <button
            key={op.id}
            type="button"
            aria-label={`选择 ${op.id} 通信`}
            aria-pressed={operation === op.id}
            onClick={() =>
              change(() => {
                setOperation(op.id);
                onOperationChange?.(op.id);
              })
            }
          >
            <strong>{op.title}</strong>
            <span>{op.action}</span>
          </button>
        ))}
      </div>
      <div className="ce-parameters">
        <label>
          组内 rank 数
          <select
            aria-label="通信 rank 数"
            value={ranks}
            onChange={(event) =>
              change(() => {
                const n = Number(event.target.value) as 2 | 4;
                setRanks(n);
                setRoot(root < n ? root : 0);
                setSender(sender < n ? sender : 0);
                setReceiver(
                  receiver < n && receiver !== (sender < n ? sender : 0)
                    ? receiver
                    : (sender < n ? sender : 0) === 0
                      ? 1
                      : 0,
                );
                setSelectedRank(selectedRank < n ? selectedRank : 0);
              })
            }
          >
            <option value={2}>2 ranks</option>
            <option value={4}>4 ranks</option>
          </select>
        </label>
        {rootOperations.includes(operation) && (
          <label>
            接收/发送 root
            <select
              aria-label="通信 root rank"
              value={root}
              onChange={(event) =>
                change(() => setRoot(Number(event.target.value)))
              }
            >
              {Array.from({ length: ranks }, (_, rank) => (
                <option key={rank} value={rank}>
                  rank {rank} · GPU {rank * 2}
                </option>
              ))}
            </select>
          </label>
        )}
        {operation === "send-recv" && (
          <>
            <label>
              发送者
              <select
                aria-label="通信发送 rank"
                value={sender}
                onChange={(event) =>
                  change(() => {
                    const next = Number(event.target.value);
                    setSender(next);
                    if (receiver === next) setReceiver(next === 0 ? 1 : 0);
                  })
                }
              >
                {Array.from({ length: ranks }, (_, rank) => (
                  <option key={rank} value={rank}>
                    rank {rank}
                  </option>
                ))}
              </select>
            </label>
            <label>
              接收者
              <select
                aria-label="通信接收 rank"
                value={receiver}
                onChange={(event) =>
                  change(() => setReceiver(Number(event.target.value)))
                }
              >
                {Array.from(
                  { length: ranks },
                  (_, rank) =>
                    rank !== sender && (
                      <option key={rank} value={rank}>
                        rank {rank}
                      </option>
                    ),
                )}
              </select>
            </label>
          </>
        )}
        <label>
          观察 rank
          <select
            aria-label="通信观察 rank"
            value={selectedRank}
            onChange={(event) => setSelectedRank(Number(event.target.value))}
          >
            {Array.from({ length: ranks }, (_, rank) => (
              <option key={rank} value={rank}>
                rank {rank}
              </option>
            ))}
          </select>
        </label>
        <label className="ce-focus">
          <input
            type="checkbox"
            checked={focus}
            onChange={(event) => setFocus(event.target.checked)}
          />
          只看观察 rank 的路径
        </label>
      </div>
      <div className="ce-explanation">
        <div>
          <h3>
            {definition.title} <span>{definition.action}</span>
          </h3>
          <p>{definition.purpose}</p>
          <code>{definition.formula}</code>
        </div>
        <p className="ce-shape-summary">
          {definition.shape}
          <br />
          <small>p = {ranks} 个组内 rank；每个括号轴是元素位置。</small>
        </p>
      </div>
      <div className="ce-mode" role="group" aria-label="通信动画模式">
        <button
          type="button"
          aria-label="分段讲解通信"
          aria-pressed={timeline.mode === "staged"}
          onClick={() => timeline.setMode("staged")}
        >
          分段讲解
        </button>
        <button
          type="button"
          aria-label="连续演示通信"
          aria-pressed={timeline.mode === "continuous"}
          onClick={() => timeline.setMode("continuous")}
        >
          连续演示
        </button>
        <span>同一条路径 · 同一个播放时钟</span>
      </div>
      <div className="ce-controls">
        <button
          type="button"
          className="ce-play"
          aria-label={timeline.playing ? "暂停通信动画" : "播放通信动画"}
          onClick={() =>
            timeline.playing ? timeline.pause() : timeline.play()
          }
        >
          {timeline.playing ? (
            <Pause size={17} aria-hidden="true" />
          ) : (
            <Play size={17} aria-hidden="true" />
          )}
          {timeline.playing ? "暂停" : "播放"}
        </button>
        <button
          type="button"
          aria-label="单步通信动画"
          onClick={timeline.step}
          disabled={timeline.index >= 4}
        >
          <SkipForward size={17} aria-hidden="true" />
          单步
        </button>
        <button
          type="button"
          aria-label="重置通信动画"
          onClick={timeline.reset}
        >
          <RotateCcw size={17} aria-hidden="true" />
          重置
        </button>
        <label>
          速度
          <select
            aria-label="通信动画速度"
            value={timeline.speed}
            onChange={(event) => timeline.setSpeed(Number(event.target.value))}
          >
            {[0.25, 0.5, 1, 2, 4].map((speed) => (
              <option key={speed} value={speed}>
                {speed}×
              </option>
            ))}
          </select>
        </label>
      </div>
      <label className="ce-seek">
        <span>
          步骤 {timeline.index + 1}/5 ·{" "}
          {Math.round((timeline.position / 4) * 100)}%
        </span>
        <input
          type="range"
          aria-label="通信动画进度"
          min={0}
          max={4}
          step={timeline.mode === "continuous" ? 0.001 : 1}
          value={timeline.position}
          onChange={(event) => timeline.seek(Number(event.target.value))}
        />
      </label>
      <div className="ce-step-rail" aria-label="通信关键帧">
        {demo.steps.map((step, index) => (
          <button
            key={step}
            type="button"
            onClick={() => timeline.seek(index)}
            aria-current={timeline.index === index ? "step" : undefined}
          >
            <span>{index + 1}</span>
            {step}
          </button>
        ))}
      </div>
      <p className="ce-status" aria-live="polite">
        {demo.steps[timeline.index]} ·{" "}
        {timeline.position < 3
          ? "接收缓冲区等待到达；在途对象保留来源标签。"
          : "接收完成；核对各 rank 的结果。"}
      </p>
      <CollectiveScene
        demo={demo}
        position={timeline.position}
        selectedRank={selectedRank}
        focus={focus}
      />
      <div className="ce-buffer-grid">
        {demo.inputs.map((input, rank) => (
          <article
            key={rank}
            className={rank === selectedRank ? "ce-buffer-selected" : ""}
          >
            <h4 style={{ color: COLORS[rank] }}>
              rank {rank} · GPU {rank * 2}
            </h4>
            <div>
              <span>发送 shape {input ? `[${input.length}]` : "—"}</span>
              <code>{input ? vector(input) : "此 rank 无发送缓冲区"}</code>
            </div>
            <div>
              <span>
                接收 shape{" "}
                {demo.outputs[rank] ? `[${demo.outputs[rank]!.length}]` : "—"}
              </span>
              {snapshot.outputs[rank] ? (
                <code className="ce-output-values">
                  {vector(snapshot.outputs[rank]!)}
                </code>
              ) : (
                <p>
                  {demo.outputs[rank]
                    ? "等待数据到达"
                    : "无接收结果（undefined），不是全零数组"}
                </p>
              )}
            </div>
          </article>
        ))}
      </div>
      <div className="ce-derivation">
        <h3>观察 rank {selectedRank} 的 shape 与归属</h3>
        <p>
          <strong>输入</strong>{" "}
          {selectedInput
            ? `${selectedInput.length} 个元素 [${selectedInput.length}]`
            : "无发送缓冲区"}{" "}
          → <strong>{definition.action}</strong> → <strong>输出</strong>{" "}
          {receive
            ? `${receive.length} 个元素 [${receive.length}]`
            : "未定义接收结果"}
          。shape 是缓冲区布局；并不等于字节数或传输耗时。
        </p>
        {operation === "all-to-all" && selectedInput && (
          <>
            <p>
              发送缓冲区切成 {ranks} 片，每片 2 个元素；目的 rank j 从偏移 j×2
              取片。
            </p>
            <div className="ce-chunks">
              {Array.from({ length: ranks }, (_, destination) => (
                <span
                  key={destination}
                  style={{ borderColor: COLORS[destination] }}
                >
                  → rank {destination}:{" "}
                  <code>
                    {vector(
                      selectedInput.slice(destination * 2, destination * 2 + 2),
                    )}
                  </code>
                </span>
              ))}
            </div>
            <p>
              接收按来源 rank 0 → {ranks - 1} 排列。
              {selectedOutput
                ? `此 rank 接收 ${vector(selectedOutput)}；元素只重排，不求和。`
                : "数值在切片到达后显示。"}
            </p>
          </>
        )}
        {(operation === "all-gather" || operation === "gather") && (
          <p>
            每 rank 2 个元素，接收端按 rank 顺序拼接 {ranks}×2={ranks * 2}{" "}
            个元素；不相加。
          </p>
        )}
        {(operation === "reduce-scatter" || operation === "scatter") && (
          <p>
            总长度 {ranks * 2} = {ranks}×2；rank {selectedRank} 持有区间 [
            {selectedRank * 2}, {(selectedRank + 1) * 2})。
            {operation === "reduce-scatter"
              ? "切片前先将所有 rank 的同位置值相加。"
              : "切片直接来自 root，没有归约。"}
          </p>
        )}
        {operation === "all-reduce" && (
          <div className="ce-equivalence">
            <strong>AllReduce ≡ ReduceScatter + AllGather</strong>
            <p>
              先逐位置求和并分成 {ranks} 个两元素片段，再按 rank
              拼接回完整结果。
              {gatheredReduction
                ? `重建结果：${vector(gatheredReduction[selectedRank]!)}，与本次 AllReduce 接收结果相同。`
                : "通信完成后可核对重建数值。"}
            </p>
          </div>
        )}
        {(operation === "reduce" || operation === "gather") && (
          <p>
            只有 root rank {root} 有接收结果，其余 rank 的输出为
            undefined；发送缓冲区仍保留。
          </p>
        )}
        {operation === "send-recv" && (
          <p>
            仅 rank {sender} 与 rank {receiver} 匹配 Send/Recv，其他 {ranks - 2}{" "}
            个 rank 不参与。本例为单向传输；双向交换需要两对匹配调用。
          </p>
        )}
      </div>
      <details className="ce-contract">
        <summary>深入：通信组、调用约束与数值顺序</summary>
        <p>
          本例把 8 张全局 GPU 中的 [
          {Array.from({ length: ranks }, (_, rank) => rank * 2).join(", ")}]
          组成一个 communicator。组内 rank 0…{ranks - 1} 是这个组的编号，root=1
          指组内 rank 1（示例 GPU 2），不要求整个 world 参与。
        </p>
        <p>
          参与同一 collective 的 rank
          必须匹配操作、count、数据类型及调用顺序。这里使用有限整数；Send 与对应
          Recv 也必须匹配 count、类型和 peer，并将互相依赖的调用正确分组。
        </p>
        <p>
          整数例子可精确核对。浮点求和不满足结合律；不同归约顺序可能产生小的舍入差异，不能把数学等价理解为任意实现逐位相同。
        </p>
        <p>
          参照{" "}
          <a
            href={collectiveSources.communicators}
            target="_blank"
            rel="noreferrer"
          >
            NCCL communicator 创建
          </a>{" "}
          与{" "}
          <a href={collectiveSources.groups} target="_blank" rel="noreferrer">
            Group 调用顺序
          </a>
          。
        </p>
      </details>
      <p className="ce-boundary">
        路径、汇合节点和先后顺序是语义教学调度，不是 NCCL 的 Ring/Tree 算法、GPU
        实测延迟或带宽。相同 rank 的箭头表示本地切片/复制。
      </p>
      <p className="ce-sources">
        官方依据：
        <a
          href={
            operation === "send-recv"
              ? collectiveSources.pointToPoint
              : collectiveSources.collectives
          }
          target="_blank"
          rel="noreferrer"
        >
          NVIDIA NCCL{" "}
          {operation === "send-recv"
            ? "Point-to-point communication"
            : "Collective Operations"}{" "}
          ↗
        </a>{" "}
        · 核查 {collectiveSources.checkedAt}
      </p>
    </section>
  );
}

function RoutedCollectiveExplorer(props: {
  initialOperation?: CollectiveOperation;
}) {
  const location = useLocation();
  const [params, setParams] = useSearchParams();
  return (
    <LocalCollectiveExplorer
      {...props}
      controlledOperation={
        location.pathname === "/distributed"
          ? (props.initialOperation ?? "all-reduce")
          : undefined
      }
      onOperationChange={
        location.pathname === "/distributed"
          ? (operation) => {
              const next = new URLSearchParams(params);
              next.set("view", "collectives");
              next.set("collective", operation);
              setParams(next, { replace: true });
            }
          : undefined
      }
    />
  );
}

export default function CollectiveExplorer(props: {
  initialOperation?: CollectiveOperation;
}) {
  const routed = useInRouterContext();
  return routed ? (
    <RoutedCollectiveExplorer {...props} />
  ) : (
    <LocalCollectiveExplorer {...props} />
  );
}
