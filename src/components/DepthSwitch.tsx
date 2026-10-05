import { useEffect } from "react";
import { useSearchParams } from "react-router";
import { usePreferences } from "../hooks/usePreferences";
import { parseRouteState } from "../lib/urlState";
import type { Depth } from "../types";
export const depthLabels = {
  beginner: "入门",
  advanced: "深入",
  expert: "精通",
};
export function useLearningDepth() {
  const [params] = useSearchParams();
  const { preferences, setDepth } = usePreferences();
  const route = parseRouteState(params);
  const depth = params.has("depth") ? route.depth : preferences.depth;
  useEffect(() => {
    if (params.has("depth") && !route.warning) setDepth(route.depth);
  }, [params, route.depth, route.warning, setDepth]);
  return depth;
}
export default function DepthSwitch() {
  const depth = useLearningDepth();
  const [params, setParams] = useSearchParams();
  const { setDepth } = usePreferences();
  function change(next: Depth) {
    setDepth(next);
    const query = new URLSearchParams(params);
    query.set("depth", next);
    setParams(query, { replace: true });
  }
  return (
    <div className="segment depth-switch" role="group" aria-label="讲解深度">
      {(Object.keys(depthLabels) as Depth[]).map((d) => (
        <button
          key={d}
          aria-pressed={depth === d}
          className={depth === d ? "selected" : ""}
          onClick={() => change(d)}
        >
          {depthLabels[d]}
        </button>
      ))}
    </div>
  );
}
