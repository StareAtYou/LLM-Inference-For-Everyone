import { frameworks } from "../content/frameworks";
import type { Topic, Depth } from "../types";
export function searchTopics(
  items: Topic[],
  query: string,
  depth: Depth,
  area?: string,
): Topic[] {
  const q = query.trim().toLowerCase();
  return items.filter(
    (t) =>
      (!area || area === "全部领域" || t.area === area) &&
      (!q ||
        [
          t.title,
          ...t.englishTerms,
          t.levels[depth].summary,
          t.levels[depth].explanation,
        ]
          .join(" ")
          .toLowerCase()
          .includes(q)),
  );
}

export function searchFrameworkModules(query: string) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return frameworks.flatMap((f) =>
    f.modules
      .filter((m) =>
        [
          m.name,
          ...m.relatedTopics,
          ...m.sources.flatMap((s) => [s.symbol ?? "", s.path ?? ""]),
        ]
          .join(" ")
          .toLowerCase()
          .includes(q),
      )
      .map((m) => ({
        name: (f.id === "vllm" ? "vLLM" : "SGLang") + " · " + m.name,
        description: m.responsibility,
        to: "/frameworks/" + f.id + "?module=" + m.id,
      })),
  );
}
