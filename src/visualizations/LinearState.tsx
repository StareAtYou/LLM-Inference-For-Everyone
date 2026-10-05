import { useState } from "react";
import { buildDeltaTrace } from "../simulation/gatedDelta";
import ParameterControl from "../components/ParameterControl";
import type { Depth, ModelPreset } from "../types";
export default function LinearState({
  model,
  depth,
}: {
  model: ModelPreset;
  depth: Depth;
}) {
  const [index, setIndex] = useState(0),
    [alpha, setAlpha] = useState(0.8),
    [beta, setBeta] = useState(0.6);
  const frames = buildDeltaTrace(alpha, beta),
    frame = frames[index];
  const description = {
    beginner:
      "两种记忆同时工作：短卷积记住最近输入，递归矩阵压缩更长的历史。点击更新，观察固定大小的格子怎样改变。",
    advanced:
      "本图使用 value×key 记法。先衰减矩阵，再用 k 读取预测值，以 v−预测值作定向写入；q 从更新后的矩阵读出。",
    expert:
      "缩小矩阵遵循 Gated Delta Rule，但省略真实投影、SiLU、归一化与输出门控。卷积为单通道三抽头；q/k/v 是独立固定夹具，不由这个单通道卷积生成。两种图只解释各自的状态更新，不执行 Qwen 层。",
  }[depth];
  return (
    <section className="explain-section linear-state-section">
      <div>
        <div className="eyebrow">LINEAR MEMORY / GATED DELTA RULE</div>
        <h2>
          历史被更新，
          <br />
          状态尺寸保持。
        </h2>
        <p>{description}</p>
        <ParameterControl
          label="遗忘门 α"
          value={alpha}
          min={0}
          max={1}
          step={0.1}
          onChange={(n) => {
            setAlpha(n);
            setIndex(0);
          }}
        />
        <ParameterControl
          label="写入门 β"
          value={beta}
          min={0}
          max={1}
          step={0.1}
          onChange={(n) => {
            setBeta(n);
            setIndex(0);
          }}
        />
        <p className="small-note">
          真实预设每个 Value 头的递归矩阵含 {model.linearHeadDim} ×{" "}
          {model.linearHeadDim} 个元素，共 {model.linearValueHeads} 个 Value
          头；实际缓存布局与卷积宽度须核对具体后端。这里缩小为一个 2 × 2 头。
        </p>
        <a
          className="text-link"
          href="https://arxiv.org/html/2412.06464v3#S3.SS1"
          target="_blank"
          rel="noreferrer"
        >
          阅读 Gated Delta Rule 原始公式 ↗
        </a>
      </div>
      <div className="dark-board linear-board">
        <div className="board-heading">
          <span className="mono">RECURRENT STATE / TOKEN {index} OF 4</span>
          <span className="board-label">教学数值</span>
        </div>
        <h3>固定矩阵 S · 2 × 2</h3>
        <div
          className="delta-matrix"
          data-testid="delta-state"
          role="img"
          aria-label={
            "递归矩阵 " +
            frame.state
              .flat()
              .map((x) => x.toFixed(3))
              .join(", ")
          }
        >
          {frame.state.flat().map((x, i) => (
            <span
              key={i}
              style={{
                background: `rgba(167,139,233,${0.12 + Math.min(Math.abs(x), 1) * 0.5})`,
              }}
            >
              {x.toFixed(3)}
            </span>
          ))}
        </div>
        <div className="delta-vectors mono">
          <span>k = [{frame.key.join(", ")}]</span>
          <span>v = [{frame.value.join(", ")}]</span>
          <span>
            读出 = [{frame.output.map((x) => x.toFixed(3)).join(", ")}]
          </span>
        </div>
        {depth !== "beginner" && (
          <div className="delta-formula mono">
            S̄ = αS
            <br />e = v − S̄k
            <br />
            S′ = S̄ + βekᵀ
          </div>
        )}
        <h3>短卷积缓冲 · 最近 3 个输入</h3>
        <div className="conv-buffer" data-testid="conv-buffer">
          {frame.buffer.map((x, i) => (
            <span key={i}>
              {x.toFixed(2)}
              <small>{i === 2 ? "当前输入" : "历史输入"}</small>
            </span>
          ))}
        </div>
        <p className="dark-description">
          固定抽头 [0.2, 0.3, 0.5] → 卷积值 {frame.conv.toFixed(3)}
          。缓冲左移，最早输入被移出；矩阵不沿 Token 轴扩张。
        </p>
        <div className="pool-actions">
          <button
            className="button primary"
            disabled={index === 4}
            onClick={() => setIndex((i) => i + 1)}
          >
            更新一个 Token
          </button>
          <button className="control-button" onClick={() => setIndex(0)}>
            重置状态
          </button>
        </div>
      </div>
    </section>
  );
}
