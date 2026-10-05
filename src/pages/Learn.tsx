import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { Search, Bookmark, Clock, Layers3 } from "lucide-react";
import { parseRouteState } from "../lib/urlState";
import { topics } from "../content/topics";
import { searchTopics, searchFrameworkModules } from "../lib/search";
import DepthSwitch, { useLearningDepth } from "../components/DepthSwitch";
import { usePreferences } from "../hooks/usePreferences";
const areaCopy: Record<string, string> = {
  推理基础: "从文字到生成，理解一次请求的生命线。",
  模型计算: "展开每一层，连接数学、结构与数据流。",
  显存与调度: "观察缓存与请求如何共享有限的资源。",
  推理优化: "理解性能收益，也理解条件和代价。",
  工程与框架: "把机制放回真实的系统实现。",
  "多 GPU 并行": "从切分轴到数据归属，看清每张 GPU 负责什么。",
  通信原语: "用可核对的输入输出，区分求和、拼接与数据交换。",
};
export default function Learn() {
  const [query, setQuery] = useState("");
  const [area, setArea] = useState("全部领域");
  const [view, setView] = useState("全部主题");
  const depth = useLearningDepth();
  const { preferences, toggleBookmark } = usePreferences();
  const [params] = useSearchParams();
  const warning = parseRouteState(params).warning;
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (params.get("focus") === "search") input.current?.focus();
  }, [params]);
  let found = searchTopics(topics, query, depth, area);
  if (view === "我的收藏")
    found = found.filter((t) => preferences.bookmarks.includes(t.id));
  if (view === "最近访问")
    found = found
      .filter((t) => preferences.recent.includes(t.id))
      .sort(
        (a, b) =>
          preferences.recent.indexOf(a.id) - preferences.recent.indexOf(b.id),
      );
  const modules =
    view === "全部主题" && ["全部领域", "工程与框架"].includes(area)
      ? searchFrameworkModules(query)
      : [];
  const areas = Object.keys(areaCopy);
  return (
    <div className="page-width content-bottom">
      <header className="page-header">
        <div className="eyebrow">THE LEARNING ATLAS / 01</div>
        <div className="page-heading-line">
          <div>
            <h1>你的推理学习地图。</h1>
            <p>沿着知识之间的联系探索，也可以从一个好奇的细节开始。</p>
          </div>
          <DepthSwitch />
        </div>
      </header>
      {warning && <p className="notice">{warning}</p>}
      <div className="atlas-intro">
        <span className="mono">THE PATH IS CONNECTED.</span>
        <div className="atlas-chain">
          {["输入与生成", "模型计算", "缓存与调度", "性能优化", "框架实现"].map(
            (n, i) => (
              <span key={n}>
                <small>0{i + 1}</small>
                {n}
              </span>
            ),
          )}
        </div>
      </div>
      <div className="atlas-toolbar">
        <div className="search-field">
          <Search size={19} />
          <input
            ref={input}
            type="search"
            aria-label="搜索知识与术语"
            placeholder="搜索主题、术语或源码模块…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <span className="keycap">⌘ K</span>
        </div>
        <div className="segment">
          {["全部主题", "我的收藏", "最近访问"].map((v) => (
            <button
              key={v}
              className={view === v ? "selected" : ""}
              onClick={() => setView(v)}
            >
              {v === "我的收藏" ? (
                <Bookmark size={14} />
              ) : v === "最近访问" ? (
                <Clock size={14} />
              ) : null}
              {v}
            </button>
          ))}
        </div>
      </div>
      <div className="area-filters" role="group" aria-label="知识领域">
        {["全部领域", ...areas].map((a) => (
          <button
            key={a}
            className={area === a ? "selected" : ""}
            onClick={() => setArea(a)}
          >
            {a}
          </button>
        ))}
        <span>{found.length} 个主题</span>
      </div>
      {modules.length > 0 && (
        <section className="atlas-section module-search-results">
          <div className="atlas-section-heading">
            <div>
              <h2>源码模块</h2>
              <p>定位到固定版本的模块与源码入口。</p>
            </div>
          </div>
          <div className="topic-grid">
            {modules.map((m) => (
              <Link
                className="topic-card"
                key={m.to}
                aria-label={m.name}
                to={m.to + "&depth=" + depth}
              >
                <div className="eyebrow">SOURCE MODULE</div>
                <h3>{m.name}</h3>
                <p>{m.description}</p>
              </Link>
            ))}
          </div>
        </section>
      )}
      {!found.length && !modules.length ? (
        <div className="empty-results">
          <Layers3 size={32} />
          <h2>没有找到相关主题</h2>
          <p>试试英文术语，或者清除筛选后重新探索。</p>
          <button
            className="button secondary"
            onClick={() => {
              setQuery("");
              setArea("全部领域");
              setView("全部主题");
            }}
          >
            清除筛选
          </button>
        </div>
      ) : (
        areas.map((a, i) => {
          const list = found.filter((t) => t.area === a);
          if (!list.length) return null;
          return (
            <section className="atlas-section" key={a}>
              <div className="atlas-section-heading">
                <span className="mono">0{i + 1}</span>
                <div>
                  <h2>{a}</h2>
                  <p>{areaCopy[a]}</p>
                </div>
                <span className="mono">
                  {list.length.toString().padStart(2, "0")} TOPICS
                </span>
              </div>
              <div className="topic-grid">
                {list.map((t) => (
                  <article className="topic-card" key={t.id}>
                    <div className="topic-card-top">
                      <span className="mono">{t.englishTerms[0]}</span>
                      <button
                        className="icon-button"
                        aria-label={
                          (preferences.bookmarks.includes(t.id)
                            ? "取消收藏："
                            : "收藏：") + t.title
                        }
                        onClick={() => toggleBookmark(t.id)}
                      >
                        <Bookmark
                          size={17}
                          fill={
                            preferences.bookmarks.includes(t.id)
                              ? "currentColor"
                              : "none"
                          }
                        />
                      </button>
                    </div>
                    <Link
                      aria-label={t.title}
                      to={"/learn/" + t.id + "?depth=" + depth}
                    >
                      <h3>{t.title}</h3>
                      <p>{t.levels[depth].summary}</p>
                    </Link>
                    <div className="topic-prerequisite">
                      {t.prerequisites.length
                        ? "前置 · " +
                          t.prerequisites
                            .map(
                              (id) =>
                                topics.find((x) => x.id === id)
                                  ?.englishTerms[0],
                            )
                            .join(" / ")
                        : "从这里开始，无需前置知识"}
                      {preferences.recent.includes(t.id) && <span>已访问</span>}
                    </div>
                  </article>
                ))}
              </div>
            </section>
          );
        })
      )}
      <div className="atlas-footnote">
        <p>访问记录保存在这台设备上，代表阅读足迹，不代表掌握程度。</p>
        <Link to="/frameworks/vllm">继续探索框架源码</Link>
      </div>
    </div>
  );
}
