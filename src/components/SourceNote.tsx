import type { SourceRef } from "../types";
export default function SourceNote({ sources }: { sources: SourceRef[] }) {
  if (!sources.length) return null;
  return (
    <details className="source-note">
      <summary>参考资料与来源 · {sources.length} 项</summary>
      {sources.map((s) => (
        <a key={s.id} href={s.url} target="_blank" rel="noreferrer">
          {s.title}
          {s.revision ? " · " + s.revision.slice(0, 7) : ""}
          <span className="source-date"> 核对于 {s.checkedAt}</span>
        </a>
      ))}
    </details>
  );
}
