import { useSearchParams } from "react-router";
import {
  parseExperimentParams,
  serializeExperimentParams,
} from "../lib/urlState";
export function useExperimentParams(experiment: string) {
  const [params, setParams] = useSearchParams();
  const parsed = parseExperimentParams(experiment, params);
  return {
    ...parsed,
    params,
    update: (key: string, value: number | string) => {
      const next = new URLSearchParams(params);
      const clean = serializeExperimentParams(experiment, {
        ...parsed.values,
        [key]: value,
      });
      for (const [k, v] of clean) next.set(k, v);
      setParams(next, { replace: true });
    },
  };
}
