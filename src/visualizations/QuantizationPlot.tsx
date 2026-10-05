import type { QuantizedVector } from "../simulation/quantization";
export default function QuantizationPlot({
  weights,
  result,
  selected,
  onSelect,
}: {
  weights: number[];
  result: QuantizedVector;
  selected: number;
  onSelect: (n: number) => void;
}) {
  const max = Math.max(1, ...weights.map(Math.abs)) * 1.15;
  const y = (v: number) => 150 - (v / max) * 108;
  const x = (i: number) => 60 + (i * 500) / (weights.length - 1);
  return (
    <>
      <svg
        className="quantization-plot"
        viewBox="0 0 620 310"
        role="img"
        aria-label="原值与反量化值比较；竖线表示误差，负值位于零轴下方"
      >
        {[-1, 0, 1].map((v) => (
          <g key={v}>
            <line
              className="plot-grid"
              x1="40"
              x2="590"
              y1={y((v * max) / 1.15)}
              y2={y((v * max) / 1.15)}
            />
            <text x="12" y={y((v * max) / 1.15) + 4}>
              {((v * max) / 1.15).toFixed(1)}
            </text>
          </g>
        ))}
        {weights.map((w, i) => (
          <g key={i} onClick={() => onSelect(i)}>
            <line
              className="error-stem"
              x1={x(i)}
              x2={x(i)}
              y1={y(w)}
              y2={y(result.reconstructed[i])}
            />
            <circle
              className="original-point"
              cx={x(i)}
              cy={y(w)}
              r={i === selected ? 8 : 5}
            />
            <rect
              className="reconstructed-point"
              x={x(i) - 4}
              y={y(result.reconstructed[i]) - 4}
              width="8"
              height="8"
            />
            <text x={x(i) - 6} y="288">
              {i}
            </text>
          </g>
        ))}
      </svg>
      <div className="chart-legend">
        <span>
          <i className="legend-dot green" />
          圆点 · 原始权重
        </span>
        <span>
          <i className="legend-dot purple" />
          方点 · 反量化值
        </span>
        <span>连线 · 数值误差</span>
      </div>
      <div className="weight-selector" role="group" aria-label="选择权重位置">
        {weights.map((_, i) => (
          <button
            key={i}
            className={i === selected ? "selected" : ""}
            onClick={() => onSelect(i)}
          >
            w{i}
          </button>
        ))}
      </div>
      <div className="quantization-detail">
        <div>
          <span>原值</span>
          <strong>{weights[selected].toFixed(4)}</strong>
        </div>
        <b>→</b>
        <div>
          <span>整数码</span>
          <strong>{result.codes[selected]}</strong>
        </div>
        <b>→</b>
        <div>
          <span>反量化值</span>
          <strong>{result.reconstructed[selected].toFixed(4)}</strong>
        </div>
      </div>
    </>
  );
}
