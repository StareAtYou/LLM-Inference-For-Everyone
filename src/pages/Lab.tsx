import { labGuides } from "../content/depthGuides";
import { lazy, Suspense } from "react";
import { Link, useParams } from "react-router";
import DepthSwitch, { useLearningDepth } from "../components/DepthSwitch";
import NotFound from "./NotFound";
const ContinuousBatching = lazy(
  () => import("../experiments/ContinuousBatching"),
);
const Quantization = lazy(() => import("../experiments/Quantization"));
const SpeculativeDecoding = lazy(
  () => import("../experiments/SpeculativeDecoding"),
);
const KvCache = lazy(() => import("../experiments/KvCache"));
const PagedAttention = lazy(() => import("../experiments/PagedAttention"));
const experiments = [
  {
    id: "kv-cache",
    name: "KV Cache",
    title: "缓存复用，显存增长。",
    subtitle: "把缓存公式变成可操作的参数，理解混合网络的显存分项。",
    Component: KvCache,
  },
  {
    id: "paged-attention",
    name: "分页与前缀",
    title: "让有限显存，流动起来。",
    subtitle: "亲手分配和释放物理块，观察尾块浪费与共同前缀。",
    Component: PagedAttention,
  },
  {
    id: "continuous-batching",
    name: "连续批处理",
    title: "每一轮，都重新组织计算。",
    subtitle: "把相同的请求放进两种调度器，观察空闲槽位与完成时间。",
    Component: ContinuousBatching,
  },
  {
    id: "quantization",
    name: "量化",
    title: "压缩数字，看见误差。",
    subtitle: "比较原始权重、整数码与反量化值，探索位宽和离群值的影响。",
    Component: Quantization,
  },
  {
    id: "speculative-decoding",
    name: "投机解码",
    title: "先猜测，再验证。",
    subtitle: "逐轮观察草稿的接受、拒绝与修正，连接正确性和成本。",
    Component: SpeculativeDecoding,
  },
];
export default function Lab() {
  const { experiment } = useParams();
  const item = experiments.find((x) => x.id === experiment);
  const depth = useLearningDepth();
  if (!item) return <NotFound />;
  const Component = item.Component;
  return (
    <div className="page-width content-bottom">
      <header className="page-header lab-header">
        <div className="eyebrow">THE MECHANISM LAB / 04</div>
        <div className="page-heading-line">
          <div>
            <h1>
              {item.id === "continuous-batching" ? (
                <>
                  每一轮，
                  <br />
                  都重新组织计算。
                </>
              ) : (
                item.title
              )}
            </h1>
            <p>{item.subtitle}</p>
          </div>
          <DepthSwitch />
        </div>
      </header>
      <nav className="lab-tabs" aria-label="选择实验">
        {experiments.map((x, i) => (
          <Link
            key={x.id}
            aria-current={x.id === experiment ? "page" : undefined}
            to={"/lab/" + x.id + "?depth=" + depth}
          >
            <span className="mono">0{i + 1}</span>
            {x.name}
          </Link>
        ))}
      </nav>
      <div className="lab-depth-note depth-guide" data-testid="depth-guide">
        <span className="eyebrow">
          {depth === "beginner"
            ? "观察机制"
            : depth === "advanced"
              ? "连接公式与状态"
              : "实现边界与验证"}
        </span>
        {labGuides[item.id][depth].map((text) => (
          <p key={text}>{text}</p>
        ))}
      </div>
      <Suspense fallback={<p role="status">正在打开实验…</p>}>
        <Component key={experiment} />
      </Suspense>
    </div>
  );
}
