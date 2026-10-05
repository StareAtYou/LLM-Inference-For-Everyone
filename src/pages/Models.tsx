import { useState } from "react";
import { Link, useSearchParams } from "react-router";
import { ArrowDown, ArrowUpRight } from "lucide-react";
import { models, getModel } from "../content/models";
import { parseRouteState } from "../lib/urlState";
import { routeExperts } from "../simulation/experts";
import DepthSwitch, { useLearningDepth } from "../components/DepthSwitch";
import SourceNote from "../components/SourceNote";
import LinearState from "../visualizations/LinearState";
import ModelGraph from "../visualizations/ModelGraph";
import ExpertRouter from "../visualizations/ExpertRouter";
import ParameterControl from "../components/ParameterControl";
export default function Models() {
  const [params, setParams] = useSearchParams();
  const route = parseRouteState(params);
  const preset = getModel(route.modelId);
  const depth = useLearningDepth();
  const [customHidden, setHidden] = useState(128);
  const [customQ, setQ] = useState(4),
    [customKV, setKV] = useState(2),
    [sequence, setSequence] = useState(8);
  const model =
    preset.id === "teaching"
      ? {
          ...preset,
          hiddenSize: customHidden,
          qHeads: customQ,
          kvHeads: customKV,
          headDim: customHidden / customQ,
          name:
            customHidden === 128 && customQ === 4 && customKV === 2
              ? preset.name
              : "自定义教学配置",
        }
      : preset;
  const rawLayer = Number(params.get("layer") ?? 0);
  const layer =
    Number.isInteger(rawLayer) &&
    rawLayer >= 0 &&
    rawLayer < model.layerTypes.length
      ? rawLayer
      : 0;
  const full = model.layerTypes[layer] === "full_attention";
  const [token, setToken] = useState(0);
  const [position, setPosition] = useState(2);
  const radians = position * 0.35;
  const routed = routeExperts(
    Array.from(
      { length: model.expertCount ?? 16 },
      (_, i) => Math.sin(i * 1.73 + token * 2) * 4,
    ),
    model.activeExperts ?? 2,
  );
  function selectLayer(n: number) {
    const q = new URLSearchParams(params);
    q.set("layer", String(n));
    setParams(q, { replace: true });
  }
  return (
    <div className="page-width content-bottom">
      <header className="page-header">
        <div className="eyebrow">THE MODEL, UNFOLDED / 03</div>
        <div className="page-heading-line">
          <div>
            <h1>
              展开网络，
              <br />
              看见计算的结构。
            </h1>
            <p>
              两个真实 Qwen 结构快照。Dense 与 MoE 指
              FFN，两者都采用混合注意力。
            </p>
          </div>
          <DepthSwitch />
        </div>
      </header>
      {route.warning && <p className="notice">{route.warning}</p>}
      <div className="workspace-toolbar">
        <label>
          模型
          <select
            aria-label="选择模型"
            value={preset.id}
            onChange={(e) => {
              const q = new URLSearchParams(params);
              q.set("model", e.target.value);
              q.delete("layer");
              setParams(q);
              setToken(0);
            }}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
        <span className="simulation-tag">
          {model.id === "teaching" ? model.name : "真实结构 · 教学激活"}
        </span>
      </div>
      <div className="model-stat-row">
        <div>
          <span>总层数</span>
          <strong>{model.layerTypes.length}</strong>
        </div>
        <div>
          <span>隐藏维度</span>
          <strong>{model.hiddenSize.toLocaleString()}</strong>
        </div>
        <div>
          <span>Full / Linear</span>
          <strong>
            {model.layerTypes.filter((t) => t === "full_attention").length} /{" "}
            {model.layerTypes.filter((t) => t === "linear_attention").length}
          </strong>
        </div>
        <div>
          <span>FFN 类型</span>
          <strong>{model.family === "moe" ? "MoE" : "Dense"}</strong>
        </div>
      </div>
      <div className="models-layout">
        <section className="dark-board">
          <div className="board-heading">
            <span className="mono">ARCHITECTURE / {model.name}</span>
            <span className="board-label">点击层展开</span>
          </div>
          <ModelGraph
            model={model}
            selectedLayer={layer}
            onSelect={selectLayer}
          />
        </section>
        <section className="block-inspector">
          <div className="inspector-heading">
            <span className="eyebrow">
              INSIDE BLOCK {String(layer + 1).padStart(2, "0")}
            </span>
            <span className={"type-badge " + (full ? "purple" : "green")}>
              {full ? "Full Attention" : "Gated DeltaNet"}
            </span>
          </div>
          <h2>一层，两次信息变换。</h2>
          <div className="block-flow">
            <div className="block-node">
              <span>01 / RMSNorm</span>
              <small>归一化输入 · 保持残差支路</small>
            </div>
            <ArrowDown size={16} />
            <div className="block-node accent">
              <strong>
                {full ? "Gated Full Attention" : "Gated DeltaNet"}
              </strong>
              <small>
                {full
                  ? `Q: ${model.qHeads} × ${model.headDim} / K,V: ${model.kvHeads} × ${model.headDim}`
                  : `Key: ${model.linearKeyHeads} heads / Value: ${model.linearValueHeads} heads / dim ${model.linearHeadDim}`}
              </small>
              <p>
                {full
                  ? "RoPE → 因果注意力 → 门控 → 输出投影；按历史位置增长的 K/V。"
                  : "短卷积 → 门控 Delta 更新 → 输出门控与投影；维护递归矩阵和卷积状态。"}
              </p>
            </div>
            <div className="residual-line mono">＋ RESIDUAL</div>
            <div className="block-node">
              <span>02 / RMSNorm</span>
              <small>归一化后送入 FFN</small>
            </div>
            <ArrowDown size={16} />
            <div className="block-node">
              <strong>
                {model.family === "moe"
                  ? "Router → Expert FFNs → Weighted Sum"
                  : "Dense Gated FFN"}
              </strong>
              <small>
                {model.hiddenSize} → {model.ffnSize} → {model.hiddenSize}
                {model.family === "moe" ? " / 每个专家" : ""}
              </small>
            </div>
            <div className="residual-line mono">＋ RESIDUAL → NEXT BLOCK</div>
          </div>
          <p className="small-note">
            {depth === "beginner"
              ? "残差把原始输入加回来，让信息沿主干持续传递。点击左侧任意层查看类型。"
              : depth === "advanced"
                ? "注意力分支混合位置信息，FFN 分支逐位置变换。3 个线性层接 1 个全注意力层，重复堆叠。"
                : "图展示文本骨干的逻辑顺序。完整实现还涉及张量布局、投影融合、缓存索引和并行划分；不以示意块推断完整显存。"}
          </p>
          <label className="inline-field">
            隐藏维度
            <input
              type="number"
              aria-label="隐藏维度"
              min={32}
              max={1024}
              step={32}
              disabled={preset.id !== "teaching"}
              value={model.hiddenSize}
              onChange={(e) => {
                const n = Math.max(
                  32,
                  Math.min(
                    1024,
                    Math.round((Number(e.target.value) || 32) / 32) * 32,
                  ),
                );
                e.currentTarget.value = String(n);
                setHidden(n);
              }}
            />
          </label>
          <p className="small-note">
            真实预设参数只读。教学隐藏维度取 32 的整数倍；改变 Q/KV
            头数可观察张量与共享分组。
          </p>
        </section>
      </div>
      <section className="shape-explorer mechanism-card">
        <div className="eyebrow">TENSOR SHAPES / BATCH = 1</div>
        <h2>维度如何流经这一层？</h2>
        <div className="shape-controls">
          <label>
            Q 头数
            <select
              aria-label="教学 Q 头数"
              value={model.qHeads}
              disabled={preset.id !== "teaching"}
              onChange={(e) => {
                const n = Number(e.target.value);
                setQ(n);
                setKV((k) => Math.min(k, n));
              }}
            >
              {(preset.id === "teaching" ? [2, 4, 8, 16] : [model.qHeads]).map(
                (n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ),
              )}
            </select>
          </label>
          <label>
            KV 头数
            <select
              aria-label="教学 KV 头数"
              value={model.kvHeads}
              disabled={preset.id !== "teaching"}
              onChange={(e) => setKV(Number(e.target.value))}
            >
              {(preset.id === "teaching"
                ? [1, 2, 4, 8].filter((n) => n <= model.qHeads)
                : [model.kvHeads]
              ).map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label>
            教学序列长度
            <input
              type="number"
              aria-label="教学序列长度"
              min={1}
              max={64}
              value={sequence}
              onChange={(e) => {
                const n = Math.max(
                  1,
                  Math.min(64, Math.round(Number(e.target.value) || 1)),
                );
                e.currentTarget.value = String(n);
                setSequence(n);
              }}
            />
          </label>
        </div>
        <div className="tensor-shapes mono" data-testid="tensor-shapes">
          <p>
            Embedding · 1 × {sequence} × {model.hiddenSize}
          </p>
          <p>
            Full Q · 1 × {sequence} × {model.qHeads} × {model.headDim}
          </p>
          <p>
            Full K / V 各 · 1 × {sequence} × {model.kvHeads} × {model.headDim}
          </p>
          <p>
            Decode 新 Q · 1 × 1 × {model.qHeads} × {model.headDim}
          </p>
        </div>
        <p className="small-note">
          轴为 batch × sequence × heads × head_dim。真实 Qwen 的 head_dim
          独立配置；仅缩小教学结构使用 hidden_size /
          Qheads。这是全注意力投影的逻辑形状，未包含额外输出门控分支，实际内核布局可能不同。
        </p>
      </section>
      {model.id !== "teaching" && !full && (
        <LinearState key={model.id} model={model} depth={depth} />
      )}
      {model.family === "moe" ? (
        <section className="explain-section">
          <div>
            <div className="eyebrow">SPARSE COMPUTE / EXPERT ROUTING</div>
            <h2>
              不是每个 Token，
              <br />
              都访问每个专家。
            </h2>
            <p>
              总参数描述模型容量，激活参数描述当前计算。不同 Token
              可选择不同专家，全部权重仍需合适的驻留与分布策略。
            </p>
            <ParameterControl
              label="教学 Token 位置"
              value={token}
              min={0}
              max={12}
              onChange={setToken}
            />
            <Link className="text-link" to="/learn/moe">
              阅读 MoE 原理 <ArrowUpRight size={15} />
            </Link>
          </div>
          <div className="dark-board">
            <ExpertRouter
              routed={routed}
              expertCount={model.expertCount!}
              sharedExperts={model.sharedExperts!}
            />
          </div>
        </section>
      ) : (
        <section className="dense-explanation">
          <span className="eyebrow">DENSE FFN</span>
          <h2>同一组权重，处理每个 Token。</h2>
          <p>
            门控分支与扩展分支逐元素相乘，再投影回隐藏维度。Dense
            并不意味着每层都是标准全注意力。
          </p>
          <div className="ffn-formula mono">
            SiLU(x W<sub>gate</sub>) ⊙ (x W<sub>up</sub>) → W<sub>down</sub>
          </div>
        </section>
      )}
      <div className="mechanism-pair">
        <section className="mechanism-card">
          <div className="eyebrow">POSITION / ROPE</div>
          <h2>用旋转，连接位置。</h2>
          <div className="rope-visual">
            <svg
              viewBox="0 0 280 210"
              role="img"
              aria-label={`二维向量旋转 ${Math.round((radians * 180) / Math.PI)} 度`}
            >
              <line x1="35" y1="150" x2="250" y2="150" />
              <line x1="80" y1="195" x2="80" y2="25" />
              <circle cx="80" cy="150" r="85" />
              <line
                className="original-vector"
                x1="80"
                y1="150"
                x2="165"
                y2="150"
              />
              <line
                className="rotated-vector"
                x1="80"
                y1="150"
                x2={80 + 85 * Math.cos(radians)}
                y2={150 - 85 * Math.sin(radians)}
              />
              <text x="175" y="172">
                x
              </text>
              <text x="30" y="25">
                x′ = R(mθ)x
              </text>
            </svg>
            <div className="mono">
              ({Math.cos(radians).toFixed(3)}, {Math.sin(radians).toFixed(3)})
            </div>
          </div>
          <ParameterControl
            label="位置 m"
            value={position}
            min={0}
            max={16}
            onChange={setPosition}
          />
          <p className="small-note">
            一个通道对的旋转示意，θ=0.35。真实 RoPE 对不同通道使用不同频率。
          </p>
        </section>
        <section className="mechanism-card">
          <div className="eyebrow">SHARED MEMORY / GQA</div>
          <h2>多个 Query，共享 K/V。</h2>
          <div className="gqa-visual">
            {Array.from({ length: model.kvHeads }, (_, i) => (
              <div key={i}>
                <span className="kv-head mono">KV {i}</span>
                <div>
                  {Array.from(
                    { length: model.qHeads / model.kvHeads },
                    (_, j) => (
                      <span key={j} className="q-head mono">
                        Q{i * (model.qHeads / model.kvHeads) + j}
                      </span>
                    ),
                  )}
                </div>
              </div>
            ))}
          </div>
          <p className="small-note">
            {model.qHeads} 个 Q 头共享 {model.kvHeads} 组 K/V，每头维度{" "}
            {model.headDim}。全注意力 KV 显存按 KV 头数计算。
          </p>
          <Link className="text-link" to={"/lab/kv-cache?model=" + model.id}>
            查看缓存估算 <ArrowUpRight size={15} />
          </Link>
        </section>
      </div>
      <SourceNote sources={model.sources} />
    </div>
  );
}
