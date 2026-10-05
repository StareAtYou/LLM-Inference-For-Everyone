import { useState } from "react";
const stages = ["Tokenization", "Prefill", "Decode"];
const explanations = [
  "文本被拆分为模型可处理的 Token 序列。",
  "一次处理提示词，建立缓存并预测第一个输出 Token。",
  "复用历史 K/V，逐步生成新的 Token。",
];
export default function RequestFlow({
  stage,
  compact = false,
}: {
  stage?: number;
  compact?: boolean;
}) {
  const [selection, setSelection] = useState(1);
  const active = stage ?? selection;
  return (
    <div className={"flow-board " + (compact ? "compact" : "")}>
      <div className="board-heading">
        <span className="mono">A REQUEST, UNDER THE HOOD</span>
        <span className="board-label">机制示意</span>
      </div>
      <div className="flow-input">
        <span className="mono dim">01 / INPUT TOKENS</span>
        <div className="token-row">
          {["解释", "大模型", "如何", "推理"].map((t, i) => (
            <span className="token" key={t}>
              <small>{i.toString().padStart(2, "0")}</small>
              {t}
            </span>
          ))}
        </div>
      </div>
      <div className="flow-wire" />
      <div className="flow-engine">
        <div className="flow-engine-title">
          <span className="engine-symbol">✳</span>
          <div>
            Qwen3.8-27B<small>DENSE · HYBRID ATTENTION</small>
          </div>
          <span className="mono dim">64 LAYERS</span>
        </div>
        <div className="engine-grid">
          <div className="layer-stack">
            {[0, 1, 2].map((i) => (
              <div key={i} className={"stack-layer layer-" + i}>
                <span className="mono">0{i + 1}</span>
                <span>Gated DeltaNet</span>
                <span className="layer-dot" />
              </div>
            ))}
            <div className="stack-layer attention">
              <span className="mono">04</span>
              <span>Gated Attention</span>
              <span className="layer-dot" />
            </div>
            <div className="stack-caption mono">3 LINEAR + 1 FULL × 16</div>
          </div>
          <div className="flow-cache">
            <span className="mono">MEMORY</span>
            <strong>KV Cache</strong>
            <div className="mini-blocks">
              {Array.from({ length: 12 }, (_, i) => (
                <i key={i} className={i < 8 ? "filled" : ""} />
              ))}
            </div>
            <span>历史计算，持续复用</span>
          </div>
        </div>
        <div className="engine-bottom">
          <span>Dense FFN</span>
          <span className="mono">5,120 → 17,408 → 5,120</span>
        </div>
      </div>
      <div className="flow-wire" />
      <div className="flow-output">
        <span className="mono dim">02 / NEXT TOKEN</span>
        <span className="output-token">模型</span>
        <span className="output-next">逐个生成，直到停止</span>
      </div>
      <div
        className="flow-stage-selector"
        role="group"
        aria-label="选择推理阶段"
      >
        {stages.map((s, i) => (
          <button
            key={s}
            onClick={() => setSelection(i)}
            aria-pressed={active === i}
            className={active === i ? "selected" : ""}
          >
            <span className="mono">0{i + 1}</span>
            {s}
          </button>
        ))}
      </div>
      <p className="flow-caption">{explanations[Math.min(active, 2)]}</p>
    </div>
  );
}
