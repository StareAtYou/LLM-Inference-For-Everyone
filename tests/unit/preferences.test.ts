import { test, expect } from "vitest";
import { readPreferences, writePreferences } from "../../src/lib/preferences";
test("corrupt_disabled_and_old_storage_recover", () => {
  expect(readPreferences({ getItem: () => "{" }).bookmarks).toEqual([]);
  expect(
    readPreferences({
      getItem: () => {
        throw Error("disabled");
      },
    }).depth,
  ).toBe("beginner");
  expect(
    readPreferences({
      getItem: () => JSON.stringify({ version: 0, depth: "expert" }),
    }).depth,
  ).toBe("beginner");
});
test("persisted_values_are_validated_and_deduplicated", () => {
  const p = readPreferences({
    getItem: () =>
      JSON.stringify({
        version: 1,
        depth: "expert",
        bookmarks: ["kv-cache", "kv-cache", 42],
        recent: ["attention"],
      }),
  });
  expect(p.depth).toBe("expert");
  expect(p.bookmarks).toEqual(["kv-cache"]);
  expect(
    writePreferences(
      {
        setItem: () => {
          throw Error("quota");
        },
      },
      p,
    ),
  ).toBe(false);
});
