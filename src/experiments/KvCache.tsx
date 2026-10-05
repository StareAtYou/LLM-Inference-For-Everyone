import { useSearchParams, Link } from "react-router";
import { getModel, models } from "../content/models";
import { parseRouteState } from "../lib/urlState";
import { useExperimentParams } from "../hooks/useExperimentParams";
import {
  estimateFullAttentionKvBytes,
  formatBytes,
} from "../simulation/memory";
import ParameterControl from "../components/ParameterControl";
import SourceNote from "../components/SourceNote";
import ShareConfig from "../components/ShareConfig";
export default function KvCache() {
  const { values, warning, update } = useExperimentParams("kv-cache");
  const [params, setParams] = useSearchParams();
  const route = parseRouteState(params);
  const model = getModel(route.modelId);
  const tokens = Number(values.tokens),
    batch = Number(values.batch),
    bytes = Number(values.bytes);
  const size = estimateFullAttentionKvBytes(model, tokens, batch, bytes);
  const full = model.layerTypes.filter((t) => t === "full_attention").length;
  return (
    <>
      <div className="experiment-layout">
        <div className="dark-board">
          <div className="board-heading">
            <span className="mono">MEMORY / FULL ATTENTION KV</span>
            <span className="board-label">理论估算</span>
          </div>
          <div className="big-observation">
            <span>全注意力 KV 分项</span>
            <strong data-testid="kv-bytes">{formatBytes(size)}</strong>
            <small>{size.toLocaleString()} bytes · 有效 K/V 数据</small>
          </div>
          <div className="kv-formula">
            <span>
              2 <small>K + V</small>
            </span>
            <b>×</b>
            <span>
              {full}
              <small>Full 层数</small>
            </span>
            <b>×</b>
            <span>
              {tokens.toLocaleString()}
              <small>Token 数</small>
            </span>
            <b>×</b>
            <span>
              {batch}
              <small>Batch</small>
            </span>
            <b>×</b>
            <span>
              {model.kvHeads}
              <small>KV 头数</small>
            </span>
            <b>×</b>
            <span>
              {model.headDim}
              <small>Head dim</small>
            </span>
            <b>×</b>
            <span>
              {bytes}
              <small>Bytes</small>
            </span>
          </div>
          <div className="context-bars">
            {[1024, 4096, 16384, 65536].map((t) => (
              <div key={t}>
                <span className="mono">{t.toLocaleString()} tokens</span>
                <div>
                  <i style={{ width: Math.max(2, (t / 65536) * 100) + "%" }} />
                </div>
                <span>
                  {formatBytes(
                    estimateFullAttentionKvBytes(model, t, batch, bytes),
                  )}
                </span>
              </div>
            ))}
          </div>
          <p className="dark-description">
            线性层维护递归矩阵与卷积状态，未计入上面的 KV
            数字。该数字也不包含权重、激活、对齐和运行时元数据。
          </p>
        </div>
        <aside className="parameter-panel">
          <span className="eyebrow">MEMORY CONTROLS</span>
          <h2>
            历史越长，
            <br />
            需要留下的越多。
          </h2>
          <label className="select-field">
            模型
            <select
              aria-label="选择模型"
              value={model.id}
              onChange={(e) => {
                const q = new URLSearchParams(params);
                q.set("model", e.target.value);
                setParams(q);
              }}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <ParameterControl
            label="上下文 Token 数"
            value={tokens}
            min={1}
            max={262144}
            onChange={(n) => update("tokens", n)}
          />
          <ParameterControl
            label="Batch 大小"
            value={batch}
            min={1}
            max={64}
            onChange={(n) => update("batch", n)}
          />
          <label className="select-field">
            缓存元素
            <select
              aria-label="缓存元素字节数"
              value={bytes}
              onChange={(e) => update("bytes", Number(e.target.value))}
            >
              {[4, 2, 1, 0.5].map((b) => (
                <option key={b} value={b}>
                  {b} bytes{" "}
                  {b === 2
                    ? " / FP16、BF16"
                    : b === 0.5
                      ? " / 4bit 有效负载"
                      : ""}
                </option>
              ))}
            </select>
          </label>
          <p className="small-note">
            0.5 byte 仅表示 4bit 有效负载。实际量化缓存还需 scale
            等元数据与支持的内核。
          </p>
          <ShareConfig />
        </aside>
      </div>
      {(warning || route.warning) && (
        <p className="notice" role="status">
          {warning ?? route.warning}
        </p>
      )}
      <div className="experiment-reading">
        <h2>复用计算，也占用资源。</h2>
        <p>
          增加 Token、并发和精度都会线性增加这一分项。GQA 通过减少 KV
          头数降低缓存大小；分页改变分配方式，不能把同一份有效 K/V
          数据凭空消掉。
        </p>
        <Link to="/lab/paged-attention" className="text-link">
          继续看分页如何组织缓存 →
        </Link>
      </div>
      <SourceNote sources={model.sources} />
    </>
  );
}
