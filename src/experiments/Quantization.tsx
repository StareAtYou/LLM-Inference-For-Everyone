import { useState } from "react";
import { Link } from "react-router";
import { useExperimentParams } from "../hooks/useExperimentParams";
import { quantizeSymmetric } from "../simulation/quantization";
import QuantizationPlot from "../visualizations/QuantizationPlot";
import ShareConfig from "../components/ShareConfig";
const weightsByScenario: Record<string, number[]> = {
  balanced: [-0.92, -0.63, -0.31, -0.08, 0.12, 0.38, 0.71, 1],
  outlier: [-0.09, -0.04, 0.03, 0.06, 0.08, 0.11, 0.14, 1],
  zero: [0, 0, 0, 0, 0, 0, 0, 0],
};
export default function Quantization() {
  const { values, update, warning } = useExperimentParams("quantization");
  const [selected, setSelected] = useState(4);
  const bits = Number(values.bits) as 4 | 8;
  const weights = weightsByScenario[String(values.scenario)];
  const result = quantizeSymmetric(weights, bits);
  return (
    <>
      <div className="experiment-layout">
        <section className="dark-board">
          <div className="board-heading">
            <span className="mono">WEIGHT → CODE → RECONSTRUCTION</span>
            <span className="board-label">教学数值</span>
          </div>
          <div className="pool-stats">
            <div>
              <span>均方误差 MSE</span>
              <strong data-testid="quant-error">
                {result.meanSquaredError.toFixed(6)}
              </strong>
            </div>
            <div>
              <span>有效负载 / FP16 基准</span>
              <strong>
                {result.payloadBytes} / {weights.length * 2}{" "}
                <small>bytes</small>
              </strong>
            </div>
          </div>
          <QuantizationPlot
            weights={weights}
            result={result}
            selected={selected}
            onSelect={setSelected}
          />
          <p className="dark-description">
            对称 INT{bits}：scale = {result.scale.toFixed(6)}，范围 ±
            {2 ** (bits - 1) - 1}。保留一个码点的教学简化，未模拟完整 FP16
            舍入；有效负载不计 scale、分组和打包对齐。
          </p>
        </section>
        <aside className="parameter-panel">
          <div className="eyebrow">PRECISION CONTROLS</div>
          <h2>
            更少的位，
            <br />
            怎样表达原值？
          </h2>
          <label className="select-field">
            权重场景
            <select
              aria-label="权重场景"
              value={String(values.scenario)}
              onChange={(e) => update("scenario", e.target.value)}
            >
              <option value="balanced">均匀幅度</option>
              <option value="outlier">存在离群值</option>
              <option value="zero">全部为零</option>
            </select>
          </label>
          <div className="small-note">量化位宽</div>
          <div className="segment precision-switch">
            {[4, 8].map((b) => (
              <button
                key={b}
                aria-pressed={bits === b}
                className={bits === b ? "selected" : ""}
                onClick={() => update("bits", b)}
              >
                INT{b}
              </button>
            ))}
          </div>
          <div className="quant-summary">
            <span>权重有效负载相对 FP16</span>
            <strong>
              {((weights.length * 2) / result.payloadBytes).toFixed(1)}×
            </strong>
            <p className="small-note">
              这是存储比值。位宽变化不直接说明端到端推理速度，也不对应所有实际量化格式。
            </p>
          </div>
          <ShareConfig />
        </aside>
      </div>
      {warning && <p className="notice">{warning}</p>}
      <div className="experiment-reading">
        <h2>误差来自离散化，也取决于尺度。</h2>
        <p>
          一个大离群值会拉大整个向量的
          scale，让小权重的格点更粗。真实分组量化、校准与误差补偿在更复杂的约束下平衡精度与内核效率。
        </p>
        <Link className="text-link" to="/learn/quantization">
          继续阅读量化原理与边界 →
        </Link>
      </div>
    </>
  );
}
