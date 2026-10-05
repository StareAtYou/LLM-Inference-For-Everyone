import { useState } from "react";
import { Link } from "react-router";
import { useExperimentParams } from "../hooks/useExperimentParams";
import { createPagedState, stepPaged } from "../simulation/paging";
import { buildPrefixTree } from "../simulation/prefix";
import CacheBlocks from "../visualizations/CacheBlocks";
import PrefixTree from "../visualizations/PrefixTree";
import ParameterControl from "../components/ParameterControl";
import ShareConfig from "../components/ShareConfig";
function Pool({
  capacity,
  blockSize,
  tokens,
}: {
  capacity: number;
  blockSize: number;
  tokens: number;
}) {
  const [state, setState] = useState(() =>
    createPagedState(capacity, blockSize),
  );
  const [next, setNext] = useState(1);
  const [error, setError] = useState("");
  function allocate() {
    try {
      setState(stepPaged(state, { type: "allocate", id: "R" + next, tokens }));
      setNext(next + 1);
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const waste = Object.values(state.requests).reduce(
    (s, r) => s + r.blocks.length * blockSize - r.tokens,
    0,
  );
  return (
    <div className="dark-board">
      <div className="board-heading">
        <span className="mono">PAGED ALLOCATION / TEACHING POOL</span>
        <span className="board-label">机制模拟</span>
      </div>
      <div className="pool-stats">
        <div>
          <span>空闲物理块</span>
          <strong>
            {state.freeBlocks.length} / {capacity}
          </strong>
        </div>
        <div>
          <span>尾块未使用位置</span>
          <strong>
            {waste} <small>tokens</small>
          </strong>
        </div>
      </div>
      <CacheBlocks state={state} />
      <div className="pool-actions">
        <button className="button primary" onClick={allocate}>
          分配请求 · {tokens} tokens
        </button>
        <button
          className="control-button"
          onClick={() => {
            setState(createPagedState(capacity, blockSize));
            setNext(1);
            setError("");
          }}
        >
          清空块池
        </button>
      </div>
      {error && (
        <p className="allocation-error" role="alert">
          {error}。当前分配保持不变，可释放请求后重试。
        </p>
      )}
      <div className="request-mappings">
        {Object.entries(state.requests).map(([id, r]) => (
          <div key={id}>
            <span className="mono">{id}</span>
            <p>{r.blocks.map((p, i) => `L${i} → P${p}`).join(" / ")}</p>
            <button
              className="control-button"
              aria-label={"释放 " + id}
              onClick={() => {
                setState(stepPaged(state, { type: "release", id }));
                setError("");
              }}
            >
              释放
            </button>
          </div>
        ))}
      </div>
      <p className="dark-description">
        独立教学块池，只展示逻辑/物理映射与尾块浪费。它不对应上一个实验的 GPU
        字节容量。
      </p>
    </div>
  );
}
export default function PagedAttention() {
  const { values, warning, update } = useExperimentParams("paged-attention");
  const capacity = Number(values.capacity),
    blockSize = Number(values.blockSize),
    tokens = Number(values.tokens);
  const [prefix, setPrefix] = useState("shared");
  const sequences =
    prefix === "shared"
      ? [
          ["你是", "老师", "。", "解释", "KV"],
          ["你是", "老师", "。", "解释", "MoE"],
          ["你是", "老师", "。", "讲解", "采样"],
        ]
      : [
          ["解释", "KV"],
          ["讲解", "MoE"],
          ["比较", "采样"],
        ];
  return (
    <>
      <div className="experiment-layout">
        <Pool
          key={capacity + "-" + blockSize}
          capacity={capacity}
          blockSize={blockSize}
          tokens={tokens}
        />
        <aside className="parameter-panel">
          <div className="eyebrow">BLOCK POOL CONTROLS</div>
          <h2>
            逻辑连续，
            <br />
            物理可以分散。
          </h2>
          <ParameterControl
            label="每块 Token 数"
            value={blockSize}
            min={1}
            max={64}
            onChange={(n) => update("blockSize", n)}
          />
          <ParameterControl
            label="物理块容量"
            value={capacity}
            min={1}
            max={128}
            onChange={(n) => update("capacity", n)}
          />
          <ParameterControl
            label="新请求 Token 数"
            value={tokens}
            min={1}
            max={262144}
            onChange={(n) => update("tokens", n)}
          />
          <div className="segment request-presets" aria-label="请求长度预设">
            {[5, 8, 13, 32].map((n) => (
              <button
                key={n}
                className={tokens === n ? "selected" : ""}
                onClick={() => update("tokens", n)}
              >
                {n} tokens
              </button>
            ))}
          </div>
          <p className="small-note">
            块尺寸或容量改变会清空块池。尝试先分配，再释放一个请求，观察后来的请求复用空块。
          </p>
          <ShareConfig />
        </aside>
      </div>
      {warning && <p className="notice">{warning}</p>}
      <section className="explain-section">
        <div>
          <div className="eyebrow">PREFIX REUSE</div>
          <h2>
            相同的开头，
            <br />
            可以只计算一次。
          </h2>
          <p>
            前缀复用匹配 Token 序列，而不只是字符串。vLLM
            以块哈希组织完整块；SGLang RadixCache
            使用压缩前缀树。这里展示概念树，不执行实际缓存命中。
          </p>
          <div className="segment">
            <button
              className={prefix === "shared" ? "selected" : ""}
              onClick={() => setPrefix("shared")}
            >
              共享前缀
            </button>
            <button
              className={prefix === "different" ? "selected" : ""}
              onClick={() => setPrefix("different")}
            >
              不同前缀
            </button>
          </div>
          <Link className="text-link" to="/frameworks/sglang?module=cache">
            追踪 RadixCache 源码 →
          </Link>
        </div>
        <div className="dark-board">
          <PrefixTree tree={buildPrefixTree(sequences)} />
        </div>
      </section>
    </>
  );
}
