import {
  createContext,
  createElement,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  type ReactNode,
} from "react";
import type { Preferences, Depth } from "../types";
import {
  browserStorage,
  defaultPreferences,
  readPreferences,
  writePreferences,
} from "../lib/preferences";
const Context = createContext<ReturnType<typeof usePreferencesState> | null>(
  null,
);
function usePreferencesState() {
  const [preferences, setPreferences] = useState<Preferences>(() => {
    const s = browserStorage();
    return s ? readPreferences(s) : defaultPreferences();
  });
  useEffect(() => {
    const s = browserStorage();
    if (s) writePreferences(s, preferences);
  }, [preferences]);
  const setDepth = useCallback(
    (depth: Depth) =>
      setPreferences((p) => (p.depth === depth ? p : { ...p, depth })),
    [],
  );
  const toggleBookmark = useCallback(
    (id: string) =>
      setPreferences((p) => ({
        ...p,
        bookmarks: p.bookmarks.includes(id)
          ? p.bookmarks.filter((x) => x !== id)
          : [...p.bookmarks, id],
      })),
    [],
  );
  const recordVisit = useCallback(
    (id: string) =>
      setPreferences((p) =>
        p.recent[0] === id
          ? p
          : {
              ...p,
              recent: [id, ...p.recent.filter((x) => x !== id)].slice(0, 20),
            },
      ),
    [],
  );
  return useMemo(
    () => ({ preferences, setDepth, toggleBookmark, recordVisit }),
    [preferences, setDepth, toggleBookmark, recordVisit],
  );
}
export function PreferencesProvider({ children }: { children: ReactNode }) {
  const value = usePreferencesState();
  return createElement(Context.Provider, { value }, children);
}
export function usePreferences() {
  const value = useContext(Context);
  if (!value) throw Error("PreferencesProvider is required");
  return value;
}
