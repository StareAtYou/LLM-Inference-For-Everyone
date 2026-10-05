import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { Link, useLocation, useSearchParams } from "react-router";
import { useLearningDepth } from "../components/DepthSwitch";
import { useTimeline } from "../hooks/useTimeline";
import {
  buildParallelExample,
  sampleParallelObject,
  type ParallelBuffer,
  type ParallelOptions,
  type ParallelStrategy,
} from "./parallel";
import "./parallel.css";
export type { ParallelStrategy } from "./parallel";

const strategies: { id: ParallelStrategy; name: string; axis: string }[] = [
  { id: "tp", name: "张量", axis: "权重通道" },
  { id: "dp", name: "数据", axis: "独立请求" },
  { id: "ep", name: "专家", axis: "专家 owner" },
  { id: "pp", name: "流水线", axis: "连续层块" },
  { id: "cp", name: "上下文", axis: "全局序列" },
  { id: "sp", name: "序列", axis: "Norm / TP 布局" },
];
const initialOptions: Record<ParallelStrategy, ParallelOptions> = {
  tp: { size: 4 },
  dp: { size: 4 },
  ep: { size: 4, topK: 2 },
  pp: { microbatches: 2 },
  cp: { size: 4, causal: true },
  sp: { size: 4 },
};
const format = (value: number) =>
  Number.isFinite(value)
    ? Number.isInteger(value)
      ? String(value)
      : value.toFixed(3)
    : "−∞";
const related: Record<ParallelStrategy, { id: string; label: string }[]> = {
  tp: [{ id: "all-reduce", label: "AllReduce 求和" }],
  dp: [{ id: "scatter", label: "Scatter 的分片语义" }],
  ep: [{ id: "all-to-all", label: "AllToAll 重排" }],
  pp: [{ id: "send-recv", label: "Send / Recv" }],
  cp: [
    { id: "send-recv", label: "环形 Send / Recv" },
    { id: "all-gather", label: "AllGather 语义" },
  ],
  sp: [
    { id: "all-gather", label: "AllGather" },
    { id: "reduce-scatter", label: "ReduceScatter" },
  ],
};

function MatrixBuffer({ data }: { data: ParallelBuffer }) {
  const [cell, setCell] = useState<{ row: number; column: number } | null>(
    null,
  );
  const picked = cell && data.matrix[cell.row]?.[cell.column];
  return (
    <section className="px-buffer" aria-label={`${data.name} ${data.shape}`}>
      <div className="px-buffer-title">
        <strong>{data.name}</strong>
        <code>{data.shape}</code>
      </div>
      <div
        className="px-matrix-scroll"
        tabIndex={0}
        aria-label={`${data.name} 数值，可横向滚动`}
      >
        {data.matrix.length ? (
          <table className="px-matrix">
            <tbody>
              {data.matrix.map((row, i) => (
                <tr key={i}>
                  <th scope="row">{i}</th>
                  {row.map((value, j) => (
                    <td key={j}>
                      <button
                        type="button"
                        aria-label={`${data.name} 行 ${i} 列 ${j} 值 ${format(value)}`}
                        aria-pressed={cell?.row === i && cell.column === j}
                        onClick={() => setCell({ row: i, column: j })}
                      >
                        {format(value)}
                      </button>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="px-empty">没有 token / 缓冲区为空</p>
        )}
      </div>
      {picked !== null && picked !== undefined && cell && (
        <p className="px-cell-readout">
          本地索引 [{cell.row},{cell.column}] = {format(picked)}
        </p>
      )}
    </section>
  );
}
function NumberChoice({
  label,
  value,
  values,
  onChange,
}: {
  label: string;
  value: number;
  values: number[];
  onChange: (value: number) => void;
}) {
  return (
    <label className="px-choice">
      <span>{label}</span>
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {values.map((n) => (
          <option value={n} key={n}>
            {n}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function ParallelExplorer({
  initialStrategy = "tp",
}: {
  initialStrategy?: ParallelStrategy;
}) {
  const [strategy, setStrategy] = useState<ParallelStrategy>(initialStrategy);
  const [configs, setConfigs] = useState(initialOptions);
  const [rankCounts, setRankCounts] = useState<Record<ParallelStrategy, 2 | 4>>(
    { tp: 2, dp: 2, ep: 2, pp: 2, cp: 2, sp: 2 },
  );
  const [selectedQuery, setSelectedQuery] = useState(0);
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();
  const depth = useLearningDepth();
  const ref = useRef<HTMLElement>(null);
  useEffect(() => setStrategy(initialStrategy), [initialStrategy]);
  const ranks = rankCounts[strategy],
    config = configs[strategy];
  const example = useMemo(
    () => buildParallelExample(strategy, ranks, config),
    [strategy, ranks, config],
  );
  const timeline = useTimeline({
    frameCount: example.stages.length,
    intervalMs: 1600,
    resetKey: `parallel-${strategy}-${ranks}-${JSON.stringify(config)}`,
    surfaceRef: ref,
    durationsMs: example.stages.slice(1).map(() => 1800),
  });
  const stage = example.stages[timeline.index];
  const update = (patch: ParallelOptions) =>
    setConfigs((previous) => ({
      ...previous,
      [strategy]: { ...previous[strategy], ...patch },
    }));
  const activeQuery = Math.min(selectedQuery, (config.size ?? 4) - 1);
  const sourceLinks = [
    {
      title: "Megatron Bridge · 并行策略",
      url: "https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html",
    },
    {
      title: "Megatron Core · CP 前向与全局 KV",
      url: "https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/features/context_parallel.html",
    },
    {
      title: "vLLM · 推理并行部署",
      url: "https://docs.vllm.ai/en/latest/serving/parallelism_scaling/",
    },
    {
      title: "vLLM · DP / EP 组映射",
      url: "https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/",
    },
  ];
  return (
    <section
      className="parallel-explorer"
      ref={ref}
      data-testid="parallel-explorer"
      data-strategy={strategy}
      data-position={timeline.position.toFixed(4)}
      aria-label="六种 GPU 并行策略教学演示"
    >
      <div className="px-top">
        <p className="px-eyebrow">PARTITION LAB · 浏览器教学计算</p>
      </div>
      <div className="px-strategies" role="group" aria-label="并行策略">
        {strategies.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-label={`选择 ${item.id.toUpperCase()} 并行`}
            aria-pressed={item.id === strategy}
            className={item.id === strategy ? "selected" : ""}
            onClick={() => {
              setStrategy(item.id);
              if (location.pathname === "/distributed") {
                const next = new URLSearchParams(searchParams);
                next.set("view", "parallel");
                next.set("parallel", item.id);
                setSearchParams(next, { replace: true });
              }
            }}
          >
            <strong>{item.id.toUpperCase()}</strong>
            <span>{item.name}并行</span>
            <small>{item.axis}</small>
          </button>
        ))}
      </div>
      <header className="px-heading">
        <div>
          <h2>{example.title}</h2>
          <p>{example.subtitle}</p>
        </div>
        <span className="px-teaching">教学值 · 非真实 GPU / 非测速</span>
      </header>
      <div className="px-options">
        <NumberChoice
          label="教学 GPU 数"
          value={ranks}
          values={[2, 4]}
          onChange={(n) =>
            setRankCounts((previous) => ({
              ...previous,
              [strategy]: n as 2 | 4,
            }))
          }
        />
        {strategy !== "pp" && (
          <NumberChoice
            label={
              strategy === "tp"
                ? "中间通道 F"
                : strategy === "dp"
                  ? "请求批次 B"
                  : strategy === "ep"
                    ? "Token 数 T"
                    : "序列长度 S"
            }
            value={config.size ?? 4}
            values={[4, 8]}
            onChange={(n) => update({ size: n })}
          />
        )}
        {strategy === "ep" && (
          <NumberChoice
            label="路由 top-k"
            value={config.topK ?? 2}
            values={[1, 2]}
            onChange={(n) => update({ topK: n })}
          />
        )}
        {strategy === "pp" && (
          <NumberChoice
            label="微批数 m"
            value={config.microbatches ?? 2}
            values={[1, 2, 4]}
            onChange={(n) => update({ microbatches: n })}
          />
        )}
        {strategy === "cp" && (
          <>
            <label className="px-choice">
              <span>注意力遮罩</span>
              <select
                aria-label="注意力遮罩"
                value={config.causal ? "causal" : "full"}
                onChange={(event) =>
                  update({ causal: event.target.value === "causal" })
                }
              >
                <option value="causal">因果 · j≤i</option>
                <option value="full">全序列</option>
              </select>
            </label>
            <NumberChoice
              label="查看 Q 位置"
              value={activeQuery}
              values={Array.from({ length: config.size ?? 4 }, (_, i) => i)}
              onChange={setSelectedQuery}
            />
          </>
        )}
      </div>
      <p className="px-boundary">
        支持的切分轴为 4 / 8，GPU 数为 2 /
        4，均可整除；不合法的切分不会被自动截断。每种策略保留自己的参数。
      </p>
      <div className="px-shape-summary">
        <div>
          <span>切什么</span>
          <p>{example.split}</p>
        </div>
        <div>
          <span>全局逻辑 shape</span>
          <code>{example.global}</code>
          <small>
            B 请求 / 批次 · S 序列位置 · H 隐藏通道 · F 中间通道 · E 专家 · d
            头维度。教学矩阵的 H=2 与真实模型配置分开。
          </small>
        </div>
      </div>
      <div className="px-player">
        <div className="px-playback">
          <div className="px-mode" role="group" aria-label="并行播放模式">
            <button
              type="button"
              aria-pressed={timeline.mode === "staged"}
              onClick={() => timeline.setMode("staged")}
            >
              分段演示
            </button>
            <button
              type="button"
              aria-pressed={timeline.mode === "continuous"}
              onClick={() => timeline.setMode("continuous")}
            >
              连续演示
            </button>
          </div>
          <button
            type="button"
            className="px-play"
            aria-label={timeline.playing ? "暂停并行演示" : "播放并行演示"}
            onClick={timeline.playing ? timeline.pause : timeline.play}
          >
            {timeline.playing ? <Pause size={16} /> : <Play size={16} />}{" "}
            {timeline.playing ? "暂停" : "播放"}
          </button>
          <button
            type="button"
            aria-label="并行单步"
            onClick={timeline.step}
            disabled={timeline.index === example.stages.length - 1}
          >
            <SkipForward size={16} />
            单步
          </button>
          <button
            type="button"
            aria-label="重置并行演示"
            onClick={timeline.reset}
          >
            <RotateCcw size={16} />
            重置
          </button>
          <label className="px-speed">
            速度
            <select
              aria-label="并行播放速度"
              value={timeline.speed}
              onChange={(event) =>
                timeline.setSpeed(Number(event.target.value))
              }
            >
              {[0.5, 1, 2, 4].map((n) => (
                <option key={n} value={n}>
                  {n}×
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="px-seek">
          <input
            type="range"
            aria-label="并行演示进度"
            min={0}
            max={example.stages.length - 1}
            step={0.01}
            value={timeline.position}
            onChange={(event) => timeline.seek(Number(event.target.value))}
          />
          <span data-testid="parallel-frame-position">
            {timeline.index + 1} / {example.stages.length}
          </span>
        </div>
        <div className="px-stage-list" role="group" aria-label="并行演示阶段">
          {example.stages.map((item, i) => (
            <button
              type="button"
              key={`${strategy}-${i}`}
              aria-pressed={i === timeline.index}
              onClick={() => timeline.seek(i)}
            >
              <span>{String(i + 1).padStart(2, "0")}</span>
              {item.title}
            </button>
          ))}
        </div>
        <div
          className="px-scene-scroll"
          tabIndex={0}
          aria-label="GPU 数据流图，可横向滚动"
        >
          <svg
            className="px-scene"
            viewBox="0 0 840 370"
            role="img"
            aria-label={`${example.title}：${stage.title}，带标签的激活与通信轨迹`}
          >
            <rect
              className="px-scene-paper"
              x={0}
              y={0}
              width={840}
              height={370}
              rx={16}
            />
            {example.objects.map((object) => (
              <polyline
                key={`path-${object.id}`}
                points={object.points.map((p) => `${p.x},${p.y}`).join(" ")}
                className={`px-path px-path-${object.kind}`}
              />
            ))}
            {Array.from({ length: ranks }, (_, r) => {
              const x = 105 + (r * 630) / (ranks - 1);
              return (
                <g key={`rank-${r}`} data-rank={r}>
                  <rect
                    className="px-gpu"
                    x={x - 83}
                    y={20}
                    width={166}
                    height={87}
                    rx={12}
                  />
                  <text
                    className="px-gpu-name"
                    x={x}
                    y={45}
                    textAnchor="middle"
                  >
                    GPU {r} · rank {r}
                  </text>
                  <text
                    className="px-gpu-owner"
                    x={x}
                    y={66}
                    textAnchor="middle"
                  >
                    {stage.ranks[r].ownership}
                  </text>
                  <text
                    className="px-gpu-buffer"
                    x={x}
                    y={87}
                    textAnchor="middle"
                  >
                    {stage.ranks[r].buffers.at(-1)?.shape ?? "空闲 · 气泡"}
                  </text>
                </g>
              );
            })}
            {example.objects.map((object, i) => {
              const p = sampleParallelObject(object, timeline.position);
              return (
                <g
                  key={object.id}
                  data-testid="parallel-object"
                  data-object-id={object.id}
                  data-x={p.x.toFixed(3)}
                  data-y={p.y.toFixed(3)}
                  transform={`translate(${p.x},${p.y})`}
                  className={`px-packet px-packet-${object.kind}`}
                  opacity={p.visible === false ? 0 : 1}
                >
                  <circle r={7} />
                  <text
                    x={
                      p.x > 650 ? -11 : p.x < 140 ? 11 : i % 2 === 0 ? 11 : -11
                    }
                    y={-8}
                    textAnchor={
                      p.x > 650
                        ? "end"
                        : p.x < 140
                          ? "start"
                          : i % 2 === 0
                            ? "start"
                            : "end"
                    }
                  >
                    {p.label}
                  </text>
                  <title>{p.label}</title>
                </g>
              );
            })}
            <text className="px-scene-note" x={24} y={346}>
              {strategy === "dp"
                ? "左：请求分派 · 右：返回服务端 · rank 间无 collective"
                : "路径表示逻辑数据依赖；布局与距离不表示真实网络拓扑、耗时或带宽"}
            </text>
          </svg>
        </div>
        <p className="px-motion-note">
          带标签的对象沿路径连续移动；暂停后位置冻结。数字缓冲区在阶段边界提交。分段和连续模式共用一条时间轴。
        </p>
        <div className="px-current">
          <div>
            <span>当前阶段 {timeline.index + 1}</span>
            <h3>{stage.title}</h3>
            <p>{stage.explanation}</p>
          </div>
          <div className="px-communication">
            <span>何时 / 为什么通信</span>
            <strong>{stage.communication}</strong>
            {timeline.fraction > 0 && (
              <small>
                正在向“{example.stages[timeline.index + 1]?.title}”移动 ·{" "}
                {(timeline.fraction * 100).toFixed(0)}%
              </small>
            )}
          </div>
        </div>
      </div>
      <div className="px-ranks">
        {stage.ranks.map((rank) => (
          <article className="px-rank" key={rank.rank}>
            <header>
              <h3>GPU {rank.rank}</h3>
              <span>rank {rank.rank}</span>
            </header>
            <p className="px-owner">{rank.ownership}</p>
            {rank.buffers.length ? (
              rank.buffers.map((data, i) => (
                <MatrixBuffer key={`buffer-${i}`} data={data} />
              ))
            ) : (
              <p className="px-empty">等待前一 stage · 当前无活动微批</p>
            )}
          </article>
        ))}
      </div>
      {example.schedule && (
        <section className="px-extra">
          <h3>微批调度与气泡</h3>
          <p>
            每格是一段等时教学计算；μ 编号沿对角线前进。气泡占比{" "}
            {(example.bubbleFraction! * 100).toFixed(1)}%。
          </p>
          <div
            className="px-matrix-scroll"
            tabIndex={0}
            aria-label="流水线调度表"
          >
            <table className="px-schedule">
              <thead>
                <tr>
                  <th>rank / 时隙</th>
                  {example.schedule.map((_, i) => (
                    <th key={i}>{i + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: ranks }, (_, r) => (
                  <tr key={r}>
                    <th>GPU {r}</th>
                    {example.schedule!.map((slot, i) => (
                      <td className={slot[r] === null ? "bubble" : ""} key={i}>
                        {slot[r] === null ? "气泡" : `μ${slot[r]}`}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      {example.routes && (
        <section className="px-extra">
          <h3>Token → 专家 → 来源 rank</h3>
          <p>
            所有发送记录可核对：同卡路由是本地访问，其余经过分发 / 返回通信。
          </p>
          <div
            className="px-matrix-scroll"
            tabIndex={0}
            aria-label="专家路由记录"
          >
            <table className="px-route-table">
              <thead>
                <tr>
                  <th>Token</th>
                  <th>来源</th>
                  <th>专家 / owner</th>
                  <th>权重</th>
                  <th>专家输出</th>
                </tr>
              </thead>
              <tbody>
                {example.routes.map((route) => (
                  <tr key={`${route.token}-${route.expert}`}>
                    <td>t{route.token}</td>
                    <td>rank {route.from}</td>
                    <td>
                      e{route.expert} / rank {route.to}
                    </td>
                    <td>{route.weight}</td>
                    <td>[{route.output.map(format).join(", ")}]</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      {example.attention && (
        <section className="px-extra">
          <h3>Q{activeQuery} 的全局 softmax</h3>
          <p>
            最终参考分布，展示完整轨迹的目标。Q 由 rank{" "}
            {Math.floor((activeQuery * ranks) / (config.size ?? 4))}{" "}
            持有；未来位置的权重在因果模式下为 0。
          </p>
          <MatrixBuffer
            data={{
              name: `Q${activeQuery} → 每个全局 K 位置的最终权重`,
              matrix: [example.attention.weights[activeQuery]],
              shape: `[1,${config.size ?? 4}]`,
            }}
          />
          {depth !== "beginner" && (
            <div
              className="px-matrix-scroll"
              tabIndex={0}
              aria-label="在线 softmax 累积"
            >
              <table className="px-route-table">
                <thead>
                  <tr>
                    <th>KV 轮</th>
                    <th>已访问块</th>
                    <th>m</th>
                    <th>l</th>
                    <th>o/l（中间或最终）</th>
                  </tr>
                </thead>
                <tbody>
                  {example.attention.rounds.map((round, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>{round[activeQuery].visited.join(" → ")}</td>
                      <td>{format(round[activeQuery].m)}</td>
                      <td>{format(round[activeQuery].l)}</td>
                      <td>
                        [{round[activeQuery].value.map(format).join(", ")}]
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
      <section className="px-derivation">
        <h3>从全局 shape 推到本地 shape</h3>
        {example.derivation.map((line, i) => (
          <p key={i}>
            <span>{i + 1}</span>
            <code>{line}</code>
          </p>
        ))}
        {depth !== "beginner" && (
          <p className="px-depth-detail">
            {strategy === "cp"
              ? "局部 softmax 的分母不同，不能直接平均；在线更新通过重新缩放旧的分母 l 和加权和 o，得到等价全局归一化。"
              : strategy === "sp"
                ? "AllReduce 可以按数学语义分解成 ReduceScatter + AllGather；SP 在输出边界只保留 ReduceScatter 的本地序列块，下一段需要完整输入时再 AllGather。"
                : "全局逻辑结果与本地物理缓冲区分开理解：复制、拼接、求和、路由返回分别保留不同的数据归属。"}
          </p>
        )}
      </section>
      <section className="px-result">
        <div>
          <h3>完整轨迹的参考输出</h3>
          <p>此表用于核对最终结果；阶段中的 GPU 缓冲区反映当前已提交状态。</p>
        </div>
        <MatrixBuffer
          data={{
            name: "逻辑 Y / O（按原始行顺序）",
            matrix: example.output,
            shape: `[${example.output.length},${example.output[0]?.length ?? 0}]`,
          }}
        />
      </section>
      <div className="px-tradeoffs">
        <article>
          <h3>收益</h3>
          <p>{example.benefit}</p>
        </article>
        <article>
          <h3>代价</h3>
          <p>{example.cost}</p>
        </article>
      </div>
      <p className="px-limitation">
        <strong>推理 / 训练边界</strong>
        {example.limitation}
      </p>
      {depth === "expert" && (
        <p className="px-expert-note">
          组合并行先确定 rank 分组和层类型，再计算设备覆盖。SP 是 TP
          的激活布局选项；EP 的设备组可与 attention TP/DP
          重叠。网络库、拓扑、内核与框架版本决定具体通信调度，本图不给出硬件性能结论。
        </p>
      )}
      <div className="px-related">
        {related[strategy].map((item) => (
          <Link key={item.id} to={`/learn/communication-${item.id}`}>
            {item.label}
            <ArrowRight size={15} />
          </Link>
        ))}
      </div>
      <footer className="px-sources">
        <strong>官方依据 · 核查 2026-10-05</strong>
        {sourceLinks.map((source) => (
          <a
            key={source.url}
            href={source.url}
            target="_blank"
            rel="noreferrer"
          >
            {source.title} ↗
          </a>
        ))}
      </footer>
    </section>
  );
}
