import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import CollectiveExplorer from "../../src/distributed/CollectiveExplorer";
import {
  buildCollectiveDemo,
  collectiveOperations,
  collectiveSnapshot,
  runCollective,
} from "../../src/distributed/collectives";

describe("collective numeric semantics", () => {
  it("sums matching positions and broadcasts AllReduce to every rank", () => {
    expect(
      runCollective("all-reduce", [
        [1, 2],
        [10, 20],
      ]),
    ).toEqual([
      [11, 22],
      [11, 22],
    ]);
    expect(
      runCollective("all-reduce", [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]),
    ).toEqual([
      [16, 20],
      [16, 20],
      [16, 20],
      [16, 20],
    ]);
  });
  it("concatenates by source rank instead of reducing", () => {
    expect(
      runCollective("all-gather", [
        [1, 2],
        [10, 20],
      ]),
    ).toEqual([
      [1, 2, 10, 20],
      [1, 2, 10, 20],
    ]);
    expect(runCollective("all-gather", [[1], [2], [3], [4]])).toEqual([
      [1, 2, 3, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4],
    ]);
  });
  it("reduces then assigns rank-indexed equal chunks", () => {
    expect(
      runCollective("reduce-scatter", [
        [1, 2, 3, 4],
        [10, 20, 30, 40],
      ]),
    ).toEqual([
      [11, 22],
      [33, 44],
    ]);
    expect(
      runCollective("reduce-scatter", [
        [1, 2, 3, 4],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
      ]),
    ).toEqual([[7], [8], [9], [10]]);
  });
  it("sends destination chunks and receives chunks ordered by source", () => {
    expect(
      runCollective("all-to-all", [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]),
    ).toEqual([
      [1, 2, 5, 6],
      [3, 4, 7, 8],
    ]);
    const input = [0, 100, 200, 300].map((base) => [
      base,
      base + 1,
      base + 10,
      base + 11,
      base + 20,
      base + 21,
      base + 30,
      base + 31,
    ]);
    const output = runCollective("all-to-all", input);
    expect(output[2]).toEqual([20, 21, 120, 121, 220, 221, 320, 321]);
    expect(output.flat().sort((a, b) => a! - b!)).toEqual(
      input.flat().sort((a, b) => a - b),
    );
  });
  it("uses communicator root rank for Broadcast and Scatter", () => {
    expect(
      runCollective("broadcast", [undefined, [7, 9]], { root: 1 }),
    ).toEqual([
      [7, 9],
      [7, 9],
    ]);
    expect(
      runCollective("scatter", [undefined, [1, 2, 3, 4]], { root: 1 }),
    ).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(
      runCollective(
        "scatter",
        [undefined, undefined, [1, 2, 3, 4], undefined],
        { root: 2 },
      ),
    ).toEqual([[1], [2], [3], [4]]);
  });
  it("leaves nonroot Reduce and Gather outputs undefined", () => {
    expect(
      runCollective(
        "reduce",
        [
          [1, 2],
          [10, 20],
        ],
        { root: 1 },
      ),
    ).toEqual([undefined, [11, 22]]);
    expect(
      runCollective(
        "gather",
        [
          [1, 2],
          [10, 20],
        ],
        { root: 1 },
      ),
    ).toEqual([undefined, [1, 2, 10, 20]]);
    expect(runCollective("gather", [[1], [2], [3], [4]], { root: 2 })).toEqual([
      undefined,
      undefined,
      [1, 2, 3, 4],
      undefined,
    ]);
  });
  it("Send/Recv touches only the chosen sender and receiver", () => {
    expect(
      runCollective("send-recv", [undefined, [7, 8], undefined, undefined], {
        sender: 1,
        receiver: 3,
      }),
    ).toEqual([undefined, undefined, undefined, [7, 8]]);
    const demo = buildCollectiveDemo("send-recv", 4, {
      sender: 1,
      receiver: 3,
    });
    expect(demo.transfers.map((t) => [t.from, t.to])).toEqual([[1, 3]]);
  });
  it.each([2, 4] as const)(
    "ReduceScatter + AllGather equals AllReduce for %i ranks",
    (ranks) => {
      const input = Array.from({ length: ranks }, (_, r) =>
        Array.from({ length: ranks * 2 }, (_, i) => r * 10 + i + 1),
      );
      const pieces = runCollective("reduce-scatter", input);
      const reconstructed = runCollective("all-gather", pieces);
      expect(reconstructed[0]).toEqual(
        ranks === 2 ? [12, 14, 16, 18] : [64, 68, 72, 76, 80, 84, 88, 92],
      );
      expect(reconstructed).toEqual(runCollective("all-reduce", input));
    },
  );
  it("does not mutate or alias caller buffers or replicated outputs", () => {
    const input = [
      [1, 2],
      [3, 4],
    ];
    const before = structuredClone(input);
    const output = runCollective("all-reduce", input);
    output[0]![0] = 99;
    expect(input).toEqual(before);
    expect(output[1]).toEqual([4, 6]);
  });
  it("rejects unmatched counts, types, invalid ranks and indivisible chunks", () => {
    expect(() => runCollective("all-reduce", [[1], [2, 3]])).toThrow();
    expect(() => runCollective("all-reduce", [[1], [NaN]])).toThrow();
    expect(() => runCollective("all-reduce", [[1], [2], [3]])).toThrow();
    expect(() => runCollective("all-reduce", [[], []])).toThrow();
    expect(() =>
      runCollective("reduce-scatter", [
        [1, 2, 3],
        [1, 2, 3],
      ]),
    ).toThrow();
    expect(() =>
      runCollective("all-to-all", [
        [1, 2, 3],
        [1, 2, 3],
      ]),
    ).toThrow();
    expect(() => runCollective("reduce", [[1], [2]], { root: 2 })).toThrow();
    expect(() =>
      runCollective("broadcast", [undefined, [1]], { root: 0 }),
    ).toThrow();
    expect(() =>
      runCollective("send-recv", [[1], undefined], { sender: 0, receiver: 0 }),
    ).toThrow();
  });
});

describe("collective transfer timeline", () => {
  it("renders an accessible nine-operation explorer with pending receive buffers", () => {
    const html = renderToStaticMarkup(
      createElement(CollectiveExplorer, { initialOperation: "all-reduce" }),
    );
    expect(html).toContain('data-testid="collective-explorer"');
    expect(html).toContain('data-operation="all-reduce"');
    for (const operation of [
      "all-reduce",
      "all-gather",
      "reduce-scatter",
      "all-to-all",
      "broadcast",
      "reduce",
      "gather",
      "scatter",
      "send-recv",
    ])
      expect(html).toContain(`aria-label="选择 ${operation} 通信"`);
    expect(html).toContain("等待数据到达");
    expect(html).not.toContain('class="ce-output-values"');
    expect(html).toContain("2026-10-05");
  });
  it("covers nine concrete operations with 2/4-rank numeric fixtures", () => {
    expect(collectiveOperations.map((op) => op.id)).toEqual([
      "all-reduce",
      "all-gather",
      "reduce-scatter",
      "all-to-all",
      "broadcast",
      "reduce",
      "gather",
      "scatter",
      "send-recv",
    ]);
    for (const op of collectiveOperations)
      for (const ranks of [2, 4] as const) {
        const demo = buildCollectiveDemo(op.id, ranks);
        expect(demo.inputs).toHaveLength(ranks);
        expect(demo.outputs).toEqual(
          runCollective(op.id, demo.inputs, demo.options),
        );
        expect(demo.transfers.length).toBeGreaterThan(0);
      }
  });
  it("withholds numeric receive buffers until their data arrives", () => {
    const demo = buildCollectiveDemo("all-to-all", 4);
    expect(collectiveSnapshot(demo, 2.99).outputs).toEqual([
      undefined,
      undefined,
      undefined,
      undefined,
    ]);
    expect(collectiveSnapshot(demo, 3).outputs).toEqual(demo.outputs);
    expect(collectiveSnapshot(demo, 0).outputs).toEqual([
      undefined,
      undefined,
      undefined,
      undefined,
    ]);
    const midpoint = collectiveSnapshot(demo, 2).transfers[0];
    expect(midpoint.progress).toBe(0.5);
    expect(collectiveSnapshot(demo, 2)).toEqual(collectiveSnapshot(demo, 2));
  });
  it("AllReduce retains identity-labelled inputs before and during its sum/distribute route", () => {
    const demo = buildCollectiveDemo("all-reduce", 2);
    expect(
      demo.transfers.filter((t) => t.to === "sum").map((t) => t.values),
    ).toEqual(demo.inputs);
    expect(
      demo.transfers.filter((t) => t.from === "sum").map((t) => t.values),
    ).toEqual(demo.outputs);
    expect(collectiveSnapshot(demo, 1.5).outputs).toEqual([
      undefined,
      undefined,
    ]);
    expect(
      collectiveSnapshot(demo, 2.5).transfers.filter((t) => t.active),
    ).toHaveLength(2);
  });
});
