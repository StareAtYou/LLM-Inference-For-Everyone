import { useState } from "react";
import { causalAttention } from "../simulation/attention";
export default function AttentionMatrix({
  tokens = ["为什么", "天空", "是", "蓝色", "的", "？"],
}: {
  tokens?: string[];
}) {
  const [selected, setSelected] = useState(2);
  const n = tokens.length;
  const matrix = causalAttention(
    tokens.map((_, i) => tokens.map((_, j) => Math.sin(i * 2 + j) * 1.6)),
  );
  const row = Math.min(selected, n - 1);
  return (
    <div className="attention-matrix">
      <div
        className="matrix-grid"
        style={{ gridTemplateColumns: `64px repeat(${n},minmax(20px,1fr))` }}
      >
        <span className="matrix-axis mono">Q ↓ K →</span>
        {tokens.map((t, i) => (
          <span className="matrix-axis" key={i}>
            {t}
          </span>
        ))}
        {matrix.map((r, i) => (
          <div className="matrix-row" key={i}>
            <button
              className="matrix-axis"
              aria-label={"观察位置 " + i}
              onClick={() => setSelected(i)}
            >
              {tokens[i]}
            </button>
            {r.map((p, j) => (
              <button
                key={j}
                aria-label={`位置 ${i} 读取 ${j}：${(p * 100).toFixed(1)}%${j > i ? "，未来不可见" : ""}`}
                onClick={() => setSelected(i)}
                className={
                  "matrix-cell " +
                  (i === row ? "matrix-selected " : "") +
                  (j > i ? "masked" : "")
                }
                style={{
                  background:
                    j > i ? "#25242d" : `rgba(162,132,239,${0.12 + p * 0.88})`,
                }}
              >
                {j > i ? "—" : p.toFixed(2)}
              </button>
            ))}
          </div>
        ))}
      </div>
      <p className="dark-description">
        位置 {row}「{tokens[row]}」只能读取 0–{row}；未来位置权重为
        0。可见行权重和为 1。
      </p>
      <div className="chart-legend">
        <span>
          <i className="legend-dot purple" />
          可见位置 · 权重越大越亮
        </span>
        <span>
          <i className="legend-dot gray" />
          未来位置 · 因果遮罩
        </span>
      </div>
    </div>
  );
}
