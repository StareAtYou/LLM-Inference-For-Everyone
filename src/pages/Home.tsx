import { Link } from "react-router";
import {
  Layers3,
  MemoryStick,
  Network,
  Code2,
  ScanLine,
  SlidersHorizontal,
} from "lucide-react";
import RequestFlow from "../visualizations/RequestFlow";
const paths = [
  {
    n: "01",
    level: "入门",
    en: "THE FOUNDATIONS",
    to: "/learn?depth=beginner",
    text: "从一个 Token 出发",
    desc: "建立直觉，看懂从输入到输出的每一步。",
    tags: "Token · Attention · Prefill / Decode",
  },
  {
    n: "02",
    level: "深入",
    en: "THE MECHANISMS",
    to: "/learn?depth=advanced",
    text: "让原理成为你的直觉",
    desc: "调整参数，观察计算、显存与调度的变化。",
    tags: "KV Cache · Batching · Quantization",
  },
  {
    n: "03",
    level: "精通",
    en: "THE SYSTEMS",
    to: "/learn?depth=expert",
    text: "走进推理引擎的内部",
    desc: "连接系统设计与源码，理解优化的条件与代价。",
    tags: "SGLang · vLLM · Parallelism",
  },
];
export default function Home() {
  return (
    <div className="home">
      <section className="hero page-width">
        <div className="hero-copy">
          <div className="eyebrow">
            <span className="tiny-cross">+</span> 大模型推理 · 交互式学习图谱
          </div>
          <h1>
            让推理，
            <br />
            从此<span className="accent-word">可见</span>。
          </h1>
          <p className="hero-description">
            从一个 Token 到一整个推理系统。
            <br />
            看见计算的过程，理解优化的原理，
            <br />
            走进开源引擎的内部。
          </p>
          <div className="hero-actions">
            <Link className="button primary" to="/learn">
              开始探索
            </Link>
            <Link className="button quiet" to="/pipeline">
              <ScanLine size={18} /> 跟随一次推理
            </Link>
          </div>
          <div className="hero-bottom">
            <div className="small-orbit" aria-hidden="true">
              ◎
            </div>
            <p>
              为好奇心而建。
              <br />
              <span>从第一次接触，到深入工程。</span>
            </p>
          </div>
        </div>
        <div className="hero-visual">
          <div className="visual-topline mono">
            <span>THE ANATOMY OF INFERENCE</span>
            <span>FIG. 001</span>
          </div>
          <RequestFlow />
          <div className="visual-bottomline mono">
            <span>EXPLORE THE PROCESS, ONE TOKEN AT A TIME</span>
            <span>交互图示</span>
          </div>
        </div>
      </section>
      <div className="topic-strip">
        <div className="page-width">
          <span className="mono">FROM FIRST PRINCIPLES</span>
          <span>Token</span>
          <b>·</b>
          <span>Transformer</span>
          <b>·</b>
          <span>KV Cache</span>
          <b>·</b>
          <span>Scheduling</span>
          <b>·</b>
          <span>Serving</span>
        </div>
      </div>
      <section className="page-width learning-paths">
        <div className="section-heading">
          <div>
            <div className="eyebrow">01 / CHOOSE YOUR PATH</div>
            <h2>
              每一次深入，
              <br />
              都有清晰的起点。
            </h2>
          </div>
          <p>
            选择适合自己的解释深度。
            <br />
            同一张图谱，连接不同阶段的理解。
          </p>
        </div>
        <div className="path-grid">
          {paths.map((p) => (
            <Link to={p.to} className="path-card" key={p.n}>
              <div className="path-top">
                <span className="path-level">{p.level}</span>
                <span className="mono">{p.n}</span>
              </div>
              <span className="path-en mono">{p.en}</span>
              <h3>{p.text}</h3>
              <p>{p.desc}</p>
              <div className="path-bottom mono">{p.tags}</div>
            </Link>
          ))}
        </div>
      </section>
      <section className="page-width explore-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">02 / OPEN THE BLACK BOX</div>
            <h2>
              理解系统，
              <br />
              从看见内部开始。
            </h2>
          </div>
          <Link to="/models" className="text-link">
            探索模型结构
          </Link>
        </div>
        <div className="editorial-grid">
          <Link to="/models" className="feature-model">
            <div className="feature-top mono">
              <Layers3 size={20} />
              <span>MODEL ARCHITECTURE</span>
              <span>01</span>
            </div>
            <div className="model-lines" aria-hidden="true">
              {[0, 1, 2, 3, 4, 5].map((i) => (
                <div key={i} style={{ width: 55 + i * 7 + "%" }}>
                  <span>{i === 3 ? "ATTENTION" : "GATED DELTANET"}</span>
                  <i />
                </div>
              ))}
            </div>
            <div className="feature-copy">
              <span className="eyebrow">Dense × MoE</span>
              <h3>两种结构，同一条推理主线。</h3>
              <p>
                以 Qwen3.8-27B 与 Qwen3.6-35B-A3B
                为例，展开每一层的计算与数据流。
              </p>
            </div>
          </Link>
          <div className="feature-side">
            <Link to="/lab/kv-cache" className="feature-mini">
              <MemoryStick size={26} />
              <span className="mono">MEMORY & CACHE</span>
              <h3>显存里，发生了什么？</h3>
              <p>看见 KV Cache 的增长、分页与前缀复用。</p>
            </Link>
            <Link to="/lab/continuous-batching" className="feature-mini">
              <SlidersHorizontal size={26} />
              <span className="mono">INFERENCE LAB</span>
              <h3>改变参数，看见代价。</h3>
              <p>在机制实验中对比调度、量化与投机解码。</p>
            </Link>
          </div>
        </div>
      </section>
      <section className="source-section">
        <div className="page-width">
          <div className="source-heading">
            <div>
              <div className="eyebrow">03 / READ THE ENGINE</div>
              <h2>
                原理的另一面，
                <br />
                是工程。
              </h2>
            </div>
            <p>
              沿着真实源码的模块与请求流，
              <br />
              理解高性能推理框架如何工作。
            </p>
          </div>
          <div className="framework-home-grid">
            <Link to="/frameworks/vllm" className="framework-home">
              <div>
                <span className="framework-logo">
                  v<span>LLM</span>
                </span>
                <span className="source-version mono">SOURCE TOUR</span>
              </div>
              <h3>从请求调度，到分页缓存。</h3>
              <p>Engine Core · Scheduler · KV Cache · Model Runner</p>
              <Code2 size={24} />
            </Link>
            <Link to="/frameworks/sglang" className="framework-home">
              <div>
                <span className="framework-logo sglang">
                  SGLang<span className="sg-star">✳</span>
                </span>
                <span className="source-version mono">SOURCE TOUR</span>
              </div>
              <h3>沿共享前缀，走进运行时。</h3>
              <p>Tokenizer · Scheduler · Radix Cache · Model Runner</p>
              <Network size={24} />
            </Link>
          </div>
        </div>
      </section>
      <section className="page-width closing-note">
        <span className="mono">KEEP EXPLORING.</span>
        <h2>
          看懂一个细节，
          <br />
          连接整个系统。
        </h2>
        <Link to="/learn" className="button primary">
          打开学习地图
        </Link>
        <p>
          机制模拟与理论估算均标明边界。
          <br />
          真实模型配置，真实源码入口。
        </p>
      </section>
    </div>
  );
}
