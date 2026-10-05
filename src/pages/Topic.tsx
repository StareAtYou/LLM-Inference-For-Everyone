import { useEffect, lazy, Suspense } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { Bookmark, BookOpen } from "lucide-react";
import { topics } from "../content/topics";
import { usePreferences } from "../hooks/usePreferences";
import DepthSwitch, {
  useLearningDepth,
  depthLabels,
} from "../components/DepthSwitch";
import SourceNote from "../components/SourceNote";
import MechanismPlayer from "../mechanisms/MechanismPlayer";
import NotFound from "./NotFound";
import { parseRouteState } from "../lib/urlState";
import {
  parallelTopicIds,
  collectiveTopicIds,
} from "../content/distributedTopics";
import type { ParallelStrategy } from "../distributed/parallel";
import type { CollectiveOperation } from "../distributed/collectives";
import "../styles/distributed-page.css";
const ParallelExplorer = lazy(() => import("../distributed/ParallelExplorer"));
const CollectiveExplorer = lazy(
  () => import("../distributed/CollectiveExplorer"),
);
export default function TopicPage() {
  const { topic: id } = useParams();
  const topic = topics.find((t) => t.id === id);
  const depth = useLearningDepth();
  const { preferences, toggleBookmark, recordVisit } = usePreferences();
  const [params] = useSearchParams();
  useEffect(() => {
    if (topic) recordVisit(topic.id);
  }, [topic, recordVisit]);
  if (!topic) return <NotFound />;
  const content = topic.levels[depth];
  const saved = preferences.bookmarks.includes(topic.id);
  const warning = parseRouteState(params).warning;
  return (
    <div className="page-width content-bottom">
      <header className="page-header topic-header">
        <Link to="/learn" className="breadcrumb">
          学习地图 / {topic.area}
        </Link>
        <div className="page-heading-line">
          <div>
            <span className="mono topic-kicker">
              {topic.englishTerms.join(" / ")}
            </span>
            <h1>{topic.title}</h1>
            <p>{content.summary}</p>
          </div>
          <div className="topic-header-actions">
            <DepthSwitch />
            <button
              className="icon-button"
              aria-label={saved ? "取消收藏" : "收藏主题"}
              onClick={() => toggleBookmark(topic.id)}
            >
              <Bookmark size={21} fill={saved ? "currentColor" : "none"} />
            </button>
          </div>
        </div>
      </header>
      {warning && <p className="notice">{warning}</p>}
      {(parallelTopicIds.includes(topic.id) ||
        collectiveTopicIds.includes(topic.id)) && (
        <section className="topic-distributed-demo">
          <h2>跟随这个主题的数据对象</h2>
          <Suspense fallback={<p role="status">正在展开独立演示…</p>}>
            {parallelTopicIds.includes(topic.id) ? (
              <div data-testid="parallel-topic-demo">
                <ParallelExplorer
                  key={topic.id}
                  initialStrategy={topic.id as ParallelStrategy}
                />
              </div>
            ) : (
              <div data-testid="collective-topic-demo">
                <CollectiveExplorer
                  key={topic.id}
                  initialOperation={topic.id as CollectiveOperation}
                />
              </div>
            )}
          </Suspense>
        </section>
      )}
      <div className="topic-layout">
        <article className="reading-body">
          <div className="reading-label">
            <BookOpen size={16} /> {depthLabels[depth]}讲解
          </div>
          <p className="topic-explanation">{content.explanation}</p>
          {!parallelTopicIds.includes(topic.id) &&
            !collectiveTopicIds.includes(topic.id) && (
              <section className="topic-mechanism">
                <h2>把这个原理展开来看</h2>
                <MechanismPlayer key={topic.id} id={topic.id} compact />
              </section>
            )}
          <h2>连接关键细节</h2>
          <ul className="detail-list">
            {content.details.map((detail, i) => (
              <li key={detail}>
                <span className="detail-number mono">0{i + 1}</span>
                {detail}
              </li>
            ))}
          </ul>
          <h2>在交互中继续理解</h2>
          <div className="related-links">
            {topic.links.map((link) => (
              <Link
                key={link.to}
                to={
                  link.to +
                  (link.to.includes("?") ? "&" : "?") +
                  "depth=" +
                  depth
                }
              >
                {link.label}
              </Link>
            ))}
          </div>
          <SourceNote sources={topic.sources} />
        </article>
        <aside className="topic-sidebar">
          <div className="panel">
            <span className="eyebrow">BEFORE THIS TOPIC</span>
            <h3>先连接这些概念</h3>
            {topic.prerequisites.length ? (
              topic.prerequisites.map((pid) => {
                const t = topics.find((x) => x.id === pid)!;
                return (
                  <Link key={pid} to={"/learn/" + pid + "?depth=" + depth}>
                    {t.title}
                  </Link>
                );
              })
            ) : (
              <p>这是一个起点，可以直接开始。</p>
            )}
          </div>
          <div className="topic-side-note">
            <span className="mono">ONE IDEA. THREE DEPTHS.</span>
            <p>先建立直觉，再看数学与工程约束。随时切换讲解深度。</p>
          </div>
          <div className="panel">
            <h3>沿着主线继续</h3>
            {topics
              .filter((t) => t.prerequisites.includes(topic.id))
              .slice(0, 4)
              .map((t) => (
                <Link key={t.id} to={"/learn/" + t.id + "?depth=" + depth}>
                  {t.title}
                </Link>
              ))}
            <Link to="/learn">查看完整学习地图</Link>
          </div>
        </aside>
      </div>
    </div>
  );
}
