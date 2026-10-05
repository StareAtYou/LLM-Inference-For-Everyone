import type { Preferences, Depth } from "../types";
export const STORAGE_KEY = "inference-atlas:preferences";
export const isDepth = (value: unknown): value is Depth =>
  ["beginner", "advanced", "expert"].includes(String(value));
export const defaultPreferences = (): Preferences => ({
  version: 1,
  depth: "beginner",
  bookmarks: [],
  recent: [],
});
function strings(value: unknown) {
  return Array.isArray(value)
    ? [
        ...new Set(
          value.filter(
            (x): x is string => typeof x === "string" && x.length < 100,
          ),
        ),
      ].slice(0, 100)
    : [];
}
export function readPreferences(
  storage: Pick<Storage, "getItem">,
): Preferences {
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return defaultPreferences();
    const p = JSON.parse(raw);
    if (!p || p.version !== 1) return defaultPreferences();
    return {
      version: 1,
      depth: isDepth(p.depth) ? p.depth : "beginner",
      bookmarks: strings(p.bookmarks),
      recent: strings(p.recent).slice(0, 20),
    };
  } catch {
    return defaultPreferences();
  }
}
export function writePreferences(
  storage: Pick<Storage, "setItem">,
  value: Preferences,
) {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}
export function browserStorage(): Storage | undefined {
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}
