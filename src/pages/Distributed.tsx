import { lazy, Suspense } from "react";
import { Link, useSearchParams } from "react-router";
import DepthSwitch, { useLearningDepth } from "../components/DepthSwitch";
import ShareConfig from "../components/ShareConfig";
import {
  parallelTopicIds,
  collectiveTopicIds,
} from "../content/distributedTopics";
import type { ParallelStrategy } from "../distributed/parallel";
import type { CollectiveOperation } from "../distributed/collectives";
import "../styles/distributed-page.css";
const ParallelExplorer = lazy(() => import("../distributed/ParallelExplorer"));
const CollectiveExplorer = lazy(
  () => import("../distributed/CollectiveExplorer"),
);
const guides = {
  beginner: [
    "先选一种切分，观察每张 GPU 的输入、负责对象和输出。再打开通信，辨认数据是在复制、拼接、求和，还是改变归属。",
    "N 是本轮参与计算的 Token 数；H 是隐藏宽度；P 是当前通信组的设备数。只有切分的轴才会变成 /P。",
  ],
  advanced: [
    "先推导全局 shape，再看本地 shape。TP 切通道，PP 切层，DP 切请求，EP 切专家归属；CP 与 Megatron SP 虽都涉及序列轴，计算边界和通信目的不同。",
    "用整数示例核对每一次通信的输入输出：AllGather 拼接，AllReduce 求和，ReduceScatter 求和后切分，AllToAll 按目的地交换。",
  ],
  expert: [
    "检查逻辑切分与实际布局的差异：通信 group、rank 顺序、padding、变长 Token、KV 头复制、专家负载和流水依赖都会改变实现。",
    "六种策略并非六个可以随意相乘的独立设备维度。SP 常沿用 TP 组，MoE EP 可以和 attention TP/DP 共享 rank；示例不运行 NCCL，也不预测 GPU 性能。",
  ],
};
export default function Distributed() {
  const [params, setParams] = useSearchParams();
  const depth = useLearningDepth();
  const view =
    params.get("view") === "collectives" ? "collectives" : "parallel";
  const parallel = params.get("parallel") || "tp";
  const collective = params.get("collective") || "all-reduce";
  const strategy = (
    parallelTopicIds.includes(parallel) ? parallel : "tp"
  ) as ParallelStrategy;
  const operation = (
    collectiveTopicIds.includes(collective) ? collective : "all-reduce"
  ) as CollectiveOperation;
  const invalid =
    (params.has("parallel") && !parallelTopicIds.includes(parallel)) ||
    (params.has("collective") && !collectiveTopicIds.includes(collective)) ||
    (params.has("view") &&
      !["parallel", "collectives"].includes(params.get("view")!));
  function changeView(next: "parallel" | "collectives") {
    const query = new URLSearchParams(params);
    query.set("view", next);
    setParams(query, { replace: true });
  }
  return (
    <div className="page-width content-bottom distributed-page">
      <header className="page-header">
        <div className="eyebrow">THE DISTRIBUTED WORKBENCH / 06</div>
        <div className="page-heading-line">
          <div>
            <h1>
              多 GPU，
              <br />
              怎样协作完成计算？
            </h1>
            <p>
              切分对象，追踪归属，再看数据如何通信。每一次交接都有可核对的 shape
              与结果。
            </p>
          </div>
          <DepthSwitch />
        </div>
      </header>
      <div className="distributed-intro">
        <div className="mono">GLOBAL → LOCAL → COMMUNICATE</div>
        <div>
          <span>01 全局对象</span>
          <span>02 每卡分工</span>
          <span>03 传输与归并</span>
          <span>04 输出归属</span>
        </div>
      </div>
      <div className="distributed-tabs-line">
        <nav className="distributed-tabs" aria-label="分布式工作台视图">
          <button
            aria-label="并行切分工作台"
            aria-pressed={view === "parallel"}
            className={view === "parallel" ? "selected" : ""}
            onClick={() => changeView("parallel")}
          >
            GPU 切分<span>TP · DP · EP · PP · CP · SP</span>
          </button>
          <button
            aria-label="通信原语工作台"
            aria-pressed={view === "collectives"}
            className={view === "collectives" ? "selected" : ""}
            onClick={() => changeView("collectives")}
          >
            通信原语<span>9 种输入 / 输出演示</span>
          </button>
        </nav>
        <ShareConfig label="分享主题链接" />
      </div>
      <p className="small-note">
        链接保存主题与解释深度；GPU 数等教学参数仅留在当前页。
      </p>
      {invalid && (
        <p className="notice" role="status">
          链接中的工作台选项无法识别，已展示对应默认示例。
        </p>
      )}
      <div
        className="distributed-depth-note"
        data-testid="distributed-depth-guide"
      >
        <span className="eyebrow">
          {depth === "beginner"
            ? "先看对象怎样移动"
            : depth === "advanced"
              ? "推导形状与通信"
              : "核对实现约束"}
        </span>
        {guides[depth].map((t) => (
          <p key={t}>{t}</p>
        ))}
      </div>
      <Suspense fallback={<p role="status">正在展开多 GPU 教学示例…</p>}>
        <section data-testid="parallel-workbench" hidden={view !== "parallel"}>
          <ParallelExplorer initialStrategy={strategy} />
        </section>
        <section
          data-testid="collective-workbench"
          hidden={view !== "collectives"}
        >
          <CollectiveExplorer initialOperation={operation} />
        </section>
      </Suspense>
      <section className="distributed-connections">
        <div>
          <span className="eyebrow">CONNECT THE COMPUTATION</span>
          <h2>把切分放回模型里。</h2>
          <p>先看单设备算子中的 shape 推导，再理解多设备上的布局和交接。</p>
        </div>
        <div>
          <Link to={`/pipeline?depth=${depth}`}>回到完整推理与张量详情 →</Link>
          <Link to={`/learn?depth=${depth}`}>查找独立主题与通信解释 →</Link>
          <Link to={`/frameworks/vllm?depth=${depth}`}>
            连接 vLLM 的工程模块 →
          </Link>
        </div>
      </section>
    </div>
  );
}
