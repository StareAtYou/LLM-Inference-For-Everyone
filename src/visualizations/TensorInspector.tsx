import { useState, useEffect } from "react";
import type { TraceFrame } from "../simulation/inferenceTrace";
import type { ModelPreset } from "../types";
export default function TensorInspector({
  frame,
  requestId,
  model,
}: {
  frame: TraceFrame;
  requestId: string;
  model: ModelPreset;
}) {
  const tensors = frame.tensors.filter(
    (t) => t.requestId === requestId || t.requestId === "batch",
  );
  const [name, setName] = useState("");
  const selected = tensors.find((t) => t.name === name) ?? tensors[0];
  useEffect(() => setName(""), [requestId]);
  const request = frame.requests.find((r) => r.id === requestId);
  const processed = request?.processedTokens ?? 0;
  return (
    <aside className="wf-inspector" aria-label="当前数据详情">
      <div className="eyebrow">THE DATA, EXPLAINED</div>
      <h2>数据现在是什么？</h2>
      <p className="wf-inspector-subtitle">
        {requestId} · {frame.label}
      </p>
      <div className="wf-shape-note">
        <span className="mono">REAL MODEL / LOGICAL SHAPE</span>
        <strong>hidden_size = {model.hiddenSize.toLocaleString()}</strong>
        <p>
          Full Q：{model.qHeads} 头 × {model.headDim}
          <br />K / V：各 {model.kvHeads} 头 × {model.headDim}
        </p>
        <small>
          配置维度只读。下面的数值来自 H=4
          的教学网络，使用真实层类型顺序；不代表 Qwen 激活。
        </small>
      </div>
      <div className="wf-tensor-view" data-testid="trace-tensors">
        <div className="wf-tensor-head">
          <span className="mono">COMPUTED TOY VALUES</span>
          {selected && <span>{selected.shape}</span>}
        </div>
        {tensors.length > 1 && (
          <label className="wf-tensor-select">
            当前张量
            <select
              aria-label="选择当前张量"
              value={selected?.name ?? ""}
              onChange={(e) => setName(e.target.value)}
            >
              {tensors.map((t, i) => (
                <option key={t.name + i} value={t.name}>
                  {t.name}
                </option>
              ))}
            </select>
          </label>
        )}
        {selected ? (
          <>
            <h3>{selected.name}</h3>
            <div className="wf-tensor-scroller">
              <div
                className="wf-tensor-grid"
                style={{
                  gridTemplateColumns: `repeat(${selected.cols},minmax(48px,1fr))`,
                }}
                role="img"
                aria-label={
                  selected.name +
                  "，" +
                  selected.shape +
                  "，数值 " +
                  selected.values.map((v) => v.toFixed(3)).join(", ")
                }
              >
                {selected.values.map((v, i) => (
                  <span
                    key={frame.index + "-" + i}
                    style={{
                      background: `rgba(${v < 0 ? "146,186,213" : "167,139,233"},${0.12 + Math.min(Math.abs(v) / 2, 1) * 0.5})`,
                    }}
                    title={`元素 ${i}: ${v}`}
                  >
                    {v.toFixed(3)}
                  </span>
                ))}
              </div>
            </div>
            <p>{selected.note}</p>
          </>
        ) : (
          <p className="wf-data-empty">
            这一帧没有 {requestId}{" "}
            的运算张量。可以单步前进，或选择当前正在计算的请求。
          </p>
        )}
      </div>
      <div className="wf-cache-detail">
        <div>
          <span>已处理位置</span>
          <strong>{processed}</strong>
        </div>
        <div>
          <span>已提交缓存位置</span>
          <strong>{request?.cacheTokens ?? 0}</strong>
        </div>
        <div>
          <span>持有物理块</span>
          <strong>{request?.blocks.length ?? 0}</strong>
        </div>
      </div>
      <p className="wf-tiny-note">
        教学块大小为 4
        Token；显示的是已完成网络计算的状态提交，层内矩阵更新可在当前张量查看。释放时缓存清零，输出保留。
      </p>
    </aside>
  );
}
