import { Link } from "react-router";
import type { ModelPreset } from "../types";
export default function ModelGraph({
  model,
  selectedLayer,
  onSelect,
}: {
  model: ModelPreset;
  selectedLayer: number;
  onSelect: (n: number) => void;
}) {
  return (
    <div className="model-graph">
      <div className="model-endpoint">
        <span className="mono">TEXT INPUT</span>
        <strong>Token → Embedding</strong>
        <small>hidden_size = {model.hiddenSize.toLocaleString()}</small>
      </div>
      <div className="layer-connector" />
      <div className="layer-group-header">
        <span className="mono">HYBRID BACKBONE</span>
        <strong>{model.layerTypes.length} 层</strong>
      </div>
      <div className="layer-grid">
        {model.layerTypes.map((type, i) => (
          <button
            key={i}
            className={
              "layer-cell " +
              (type === "full_attention" ? "full" : "linear") +
              (selectedLayer === i ? " selected" : "")
            }
            onClick={() => onSelect(i)}
            aria-label={`第 ${i + 1} 层 · ${type === "full_attention" ? "Full Attention" : "Gated DeltaNet"}`}
            aria-pressed={selectedLayer === i}
          >
            <span className="mono">{String(i + 1).padStart(2, "0")}</span>
            <i />
          </button>
        ))}
      </div>
      <div className="chart-legend">
        <span>
          <i className="legend-dot green" />
          Gated DeltaNet ·{" "}
          {model.layerTypes.filter((t) => t === "linear_attention").length}
        </span>
        <span>
          <i className="legend-dot purple" />
          Full Attention ·{" "}
          {model.layerTypes.filter((t) => t === "full_attention").length}
        </span>
      </div>
      <div className="layer-connector" />
      <div className="model-endpoint">
        <strong>Final Norm → LM Head</strong>
        <small>生成词表 logits，交给采样器</small>
      </div>
      <div className="inactive-branches">
        <span>Vision 分支 · 文本路径未启用</span>
        <span>
          MTP · <Link to="/learn/speculation">仅解释概念</Link>
        </span>
      </div>
    </div>
  );
}
