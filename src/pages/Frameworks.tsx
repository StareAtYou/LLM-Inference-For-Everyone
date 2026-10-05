import { Link, useParams, useSearchParams } from "react-router";
import { ArrowUpRight, Code2 } from "lucide-react";
import { moduleGuide } from "../content/depthGuides";
import { frameworks } from "../content/frameworks";
import { topics } from "../content/topics";
import { parseRouteState } from "../lib/urlState";
import DepthSwitch, { useLearningDepth } from "../components/DepthSwitch";
import SourceNote from "../components/SourceNote";
import ShareConfig from "../components/ShareConfig";
import FrameworkGraph from "../visualizations/FrameworkGraph";
import NotFound from "./NotFound";
export default function Frameworks() {
  const { framework: id } = useParams();
  const definition = frameworks.find((f) => f.id === id);
  const [params, setParams] = useSearchParams();
  const depth = useLearningDepth();
  if (!definition) return <NotFound />;
  const moduleValues = params.getAll("module");
  const selected =
    moduleValues.length === 1
      ? definition.modules.find((m) => m.id === moduleValues[0])
      : undefined;
  const module = selected ?? definition.modules[0];
  const warning =
    params.has("module") && !selected
      ? "模块参数无效，已回到请求入口。"
      : parseRouteState(params).warning;
  function select(id: string) {
    const q = new URLSearchParams(params);
    q.set("module", id);
    setParams(q, { replace: true });
  }
  return (
    <div className="page-width content-bottom">
      <header className="page-header framework-header">
        <div className="eyebrow">FROM CONCEPT TO CODE / 05</div>
        <div className="page-heading-line">
          <div>
            <h1>
              从一个请求，
              <br />
              走进真实的源码。
            </h1>
            <p>
              沿模块职责与数据关系阅读，不必从第一行开始。每个入口都固定到同一个提交。
            </p>
          </div>
          <DepthSwitch />
        </div>
      </header>
      <div className="framework-toolbar">
        <div className="framework-tabs">
          {frameworks.map((f) => (
            <Link
              key={f.id}
              className={f.id === id ? "selected" : ""}
              to={"/frameworks/" + f.id + "?depth=" + depth}
            >
              <strong>{f.id === "vllm" ? "vLLM" : "SGLang"}</strong>
              <span className="mono">{f.release}</span>
            </Link>
          ))}
        </div>
        <span className="version-stamp mono">
          COMMIT {definition.commit.slice(0, 7)} / 2026-10-04
        </span>
      </div>
      {warning && <p className="notice">{warning}</p>}
      <div className="framework-layout">
        <section className="dark-board framework-board">
          <div className="board-heading">
            <span className="mono">{id?.toUpperCase()} / MODULE ATLAS</span>
            <span className="board-label">职责导览</span>
          </div>
          <FrameworkGraph
            framework={definition}
            selected={module.id}
            onSelect={select}
          />
          <p className="dark-description">
            常规文本路径的模块概览，省略部分协议层、Worker
            和后端。连线标签说明中间层及部署差异，不是完整运行时调用追踪。
          </p>
        </section>
        <section className="source-inspector">
          <div className="eyebrow">
            MODULE{" "}
            {String(definition.modules.indexOf(module) + 1).padStart(2, "0")} /{" "}
            {definition.modules.length}
          </div>
          <h2>{module.name}</h2>
          <p className="module-responsibility">{module.responsibility}</p>
          <div className="module-io">
            <div>
              <span className="mono">INPUTS</span>
              {module.inputs.map((s) => (
                <p key={s}>{s}</p>
              ))}
            </div>
            <div>
              <span className="mono">OUTPUTS</span>
              {module.outputs.map((s) => (
                <p key={s}>{s}</p>
              ))}
            </div>
          </div>
          <div className="source-code">
            <div>
              <span>
                <Code2 size={14} />{" "}
                {module.snippet.kind === "source"
                  ? "实际源码片段"
                  : "教学伪代码"}
              </span>
              <a
                href={module.sources[0].url}
                target="_blank"
                rel="noreferrer"
                aria-label="在 GitHub 查看源码"
              >
                L{module.sources[0].startLine} <ArrowUpRight size={14} />
              </a>
            </div>
            <pre tabIndex={0} aria-label="源码片段">
              <code>{module.snippet.code}</code>
            </pre>
          </div>
          <div
            className="code-explanation depth-guide"
            data-testid="module-depth-guide"
          >
            <span className="eyebrow">
              {depth === "beginner"
                ? "模块如何协作"
                : depth === "advanced"
                  ? "输入输出与状态"
                  : "源码契约与边界"}
            </span>
            {moduleGuide(module, depth).map((text) => (
              <p key={text}>{text}</p>
            ))}
          </div>
          <div className="module-topics">
            <span className="eyebrow">CONNECT THE CONCEPTS</span>
            <div>
              {module.relatedTopics.map((t) => {
                const topic = topics.find((x) => x.id === t)!;
                return (
                  <Link key={t} to={"/learn/" + t + "?depth=" + depth}>
                    {topic.englishTerms[0]} <ArrowUpRight size={12} />
                  </Link>
                );
              })}
            </div>
          </div>
          <SourceNote sources={module.sources} />
          <ShareConfig />
        </section>
      </div>
      <div className="source-scope">
        <div>
          <span className="eyebrow">SCOPE & ATTRIBUTION</span>
          <h2>带着原理，回到实现。</h2>
        </div>
        <p>
          vLLM 与 SGLang 源码均依 Apache-2.0
          引用，版权归各项目贡献者。模块职责由本网站整理，短片段只用于解释。模型注册、硬件后端与部署支持需按所选版本另行验证；这里没有启动推理服务。
        </p>
        <a
          className="text-link"
          href={
            id === "vllm"
              ? "https://docs.vllm.ai/en/stable/design/arch_overview/"
              : "https://docs.sglang.io/"
          }
          target="_blank"
          rel="noreferrer"
        >
          官方文档 <ArrowUpRight size={15} />
        </a>
      </div>
    </div>
  );
}
