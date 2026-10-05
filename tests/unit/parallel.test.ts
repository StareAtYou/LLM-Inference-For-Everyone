import { describe, expect, it } from "vitest";
import {
  buildParallelExample,
  contextAttention,
  dataParallel,
  expertParallel,
  matmul,
  pipelineParallel,
  sampleParallelObject,
  sequenceParallel,
  tensorParallel,
} from "../../src/distributed/parallel";

describe("GPU partition teaching arithmetic", () => {
  it.each([2, 4] as const)(
    "TP with %i ranks preserves the full matmul by summing row-shard contributions",
    (ranks) => {
      const result = tensorParallel(
        [[1, 2]],
        [
          [1, 0, 2, 1],
          [0, 1, 1, 2],
        ],
        [
          [1, 0],
          [0, 1],
          [1, 1],
          [2, 0],
        ],
        ranks,
      );
      expect(result.hidden).toEqual([[1, 2, 4, 5]]);
      expect(result.output).toEqual([[15, 6]]);
      expect(result.shards.map((shard) => shard.a[0].length)).toEqual(
        Array(ranks).fill(4 / ranks),
      );
      expect(
        result.shards.reduce((sum, shard) => sum + shard.partial[0][0], 0),
      ).toBe(15);
    },
  );
  it("DP changes only the owning replica's outputs, with no inference collective", () => {
    const a = dataParallel(
      [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ],
      [
        [1, 2],
        [0, 1],
      ],
      2,
    );
    const b = dataParallel(
      [
        [9, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ],
      [
        [1, 2],
        [0, 1],
      ],
      2,
    );
    expect(a.replicas[0].output).toEqual([
      [1, 4],
      [3, 10],
    ]);
    expect(a.replicas[1].output).toEqual([
      [5, 16],
      [7, 22],
    ]);
    expect(b.replicas[1].output).toEqual(a.replicas[1].output);
    expect(b.replicas[0].output[0]).toEqual([9, 20]);
    expect(a.collectives).toEqual([]);
  });
  it("EP returns top-k expert results to token owners before a weighted merge", () => {
    const result = expertParallel(
      [
        [2, 4],
        [3, 1],
        [1, 1],
        [2, 1],
      ],
      2,
      2,
    );
    expect(result.output[0]).toEqual([2.5, 5]);
    expect(result.output[3]).toEqual([6.5, 3.25]);
    expect(
      result.routes
        .filter((route) => route.token === 0)
        .map((route) => [route.expert, route.weight, route.to]),
    ).toEqual([
      [0, 0.75, 0],
      [1, 0.25, 0],
    ]);
    expect(
      result.routes
        .filter((route) => route.token === 3)
        .map((route) => route.to),
    ).toEqual([1, 0]);
    expect(expertParallel([[2, 4]], 4, 1).output).toEqual([[2, 4]]);
  });
  it.each([2, 4] as const)(
    "PP preserves sequential layer application and forwards activation values with %i ranks",
    (ranks) => {
      const result = pipelineParallel(
        [
          [1, 2],
          [3, 4],
        ],
        ranks,
      );
      expect(result.output).toEqual([
        [11, 12],
        [13, 14],
      ]);
      expect(result.schedule).toHaveLength(2 + ranks - 1);
      expect(result.schedule[0].filter((item) => item !== null)).toHaveLength(
        1,
      );
      expect(result.handoffs[0].value).toEqual(ranks === 2 ? [4, 5] : [2, 3]);
      expect(result.bubbleFraction).toBe((ranks - 1) / (2 + ranks - 1));
    },
  );
  it.each([2, 4] as const)(
    "CP with %i ranks uses a global softmax rather than local softmax averages",
    (ranks) => {
      const result = contextAttention(
        [1, 2, 3, 4],
        [0, 1, 2, 3],
        [[1], [10], [100], [1000]],
        ranks,
        false,
      );
      const denominator = 1 + Math.exp(1) + Math.exp(2) + Math.exp(3);
      const expected =
        (1 + 10 * Math.exp(1) + 100 * Math.exp(2) + 1000 * Math.exp(3)) /
        denominator;
      expect(result.output[0][0]).toBeCloseTo(expected, 10);
      expect(result.weights[0].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      expect(result.rounds.at(-1)![0].value[0]).toBeCloseTo(expected, 10);
      expect(result.rounds.at(-1)![0].visited).toHaveLength(ranks);
    },
  );
  it("CP applies a causal mask with global positions across KV shards", () => {
    const result = contextAttention(
      [1, 1, 1, 1],
      [0, 1, 2, 3],
      [[1], [10], [100], [1000]],
      2,
      true,
    );
    expect(result.output[0]).toEqual([1]);
    expect(result.weights[1][2]).toBe(0);
    expect(result.weights[1][3]).toBe(0);
    expect(result.output[1][0]).toBeCloseTo(
      (1 + 10 * Math.E) / (1 + Math.E),
      10,
    );
  });
  it.each([2, 4] as const)(
    "SP with %i ranks AllGathers sequence and ReduceScatters summed TP output",
    (ranks) => {
      const result = sequenceParallel(
        [
          [1, 1],
          [1, 1],
          [1, 1],
          [1, 1],
        ],
        [
          [1, 0, 2, 1],
          [0, 1, 1, 2],
        ],
        [
          [1, 0],
          [0, 1],
          [1, 1],
          [2, 0],
        ],
        ranks,
      );
      expect(result.normalized).toEqual([
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
      ]);
      expect(result.gathered.every((replica) => replica.length === 4)).toBe(
        true,
      );
      expect(result.output).toEqual([
        [10, 4],
        [10, 4],
        [10, 4],
        [10, 4],
      ]);
      expect(result.scattered.flat()).toEqual(result.output);
      expect(
        result.scattered.every((shard) => shard.length === 4 / ranks),
      ).toBe(true);
    },
  );
  it("bounds examples to divisible axes instead of silently dropping elements", () => {
    expect(() =>
      tensorParallel(
        [[1, 1]],
        [
          [1, 2, 3],
          [1, 2, 3],
        ],
        [[1], [1], [1]],
        2,
      ),
    ).toThrow(/整除/);
    expect(() => dataParallel([[1], [2], [3]], [[1]], 2)).toThrow(/整除/);
    expect(() => contextAttention([1], [1], [[1]], 3, false)).toThrow(/2.*4/);
    expect(() => matmul([[1, 2]], [[1]])).toThrow(/维度/);
  });
});

describe("parallel trajectory contracts", () => {
  it.each(["tp", "dp", "ep", "pp", "cp", "sp"] as const)(
    "%s has concrete rank buffers and persistent interpolated objects",
    (strategy) => {
      const example = buildParallelExample(strategy, 4, {
        size: 4,
        topK: 2,
        microbatches: 2,
        causal: false,
      });
      expect(example.stages.length).toBeGreaterThan(2);
      expect(example.stages.every((stage) => stage.ranks.length === 4)).toBe(
        true,
      );
      expect(example.output.length).toBeGreaterThan(0);
      const moving = example.objects.find((object) =>
        object.points.some(
          (point, index) => index > 0 && point.x !== object.points[index - 1].x,
        ),
      );
      expect(moving).toBeDefined();
      const first = moving!.points.findIndex(
        (point, index) => index > 0 && point.x !== moving!.points[index - 1].x,
      );
      const from = sampleParallelObject(moving!, first - 1);
      const to = sampleParallelObject(moving!, first);
      const middle = sampleParallelObject(moving!, first - 0.5);
      expect(middle.x).toBeCloseTo((from.x + to.x) / 2, 10);
      expect(middle.y).toBeCloseTo((from.y + to.y) / 2, 10);
      expect(middle.label).toBe(from.label);
    },
  );
});

describe("communication destinations", () => {
  it("TP completes AllReduce at every rank rather than leaving outputs at a gather point", () => {
    const example = buildParallelExample("tp", 4);
    const last = example.stages.length - 1;
    const resultPackets = example.objects.filter((object) =>
      object.id.startsWith("tp-"),
    );
    expect(
      resultPackets.map((object) => sampleParallelObject(object, last).x),
    ).toEqual([105, 315, 525, 735]);
    expect(
      resultPackets.every((object) =>
        sampleParallelObject(object, last).label.startsWith("Y"),
      ),
    ).toBe(true);
  });
  it("SP delivers copied sequence blocks to all TP ranks before the full-input projection", () => {
    const example = buildParallelExample("sp", 4);
    const copies = example.objects.filter((object) =>
      object.id.startsWith("sp-ag-"),
    );
    expect(copies).toHaveLength(12);
    for (const x of [105, 315, 525, 735]) {
      expect(
        copies.filter((object) => sampleParallelObject(object, 1).x === x),
      ).toHaveLength(3);
    }
    expect(
      example.stages[1].ranks.every(
        (rank) => rank.buffers[0].matrix.length === 4,
      ),
    ).toBe(true);
  });
});
