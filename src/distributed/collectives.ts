export type CollectiveOperation =
  | "all-reduce"
  | "all-gather"
  | "reduce-scatter"
  | "all-to-all"
  | "broadcast"
  | "reduce"
  | "gather"
  | "scatter"
  | "send-recv";
export type RankBuffers = (number[] | undefined)[];
export type CollectiveOptions = {
  root?: number;
  sender?: number;
  receiver?: number;
};
export type CollectiveDefinition = {
  id: CollectiveOperation;
  title: string;
  action: string;
  formula: string;
  purpose: string;
  shape: string;
};
export type CollectiveTransfer = {
  id: string;
  from: number | "sum";
  to: number | "sum";
  values: number[];
  label: string;
  start: number;
  end: number;
};
export type CollectiveDemo = {
  operation: CollectiveOperation;
  ranks: 2 | 4;
  inputs: RankBuffers;
  outputs: RankBuffers;
  options: CollectiveOptions;
  transfers: CollectiveTransfer[];
  steps: string[];
};
export const collectiveSources = {
  collectives:
    "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html",
  pointToPoint:
    "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html",
  groups:
    "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html",
  communicators:
    "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html",
  checkedAt: "2026-10-05",
};
export const collectiveOperations: CollectiveDefinition[] = [
  {
    id: "all-reduce",
    title: "AllReduce",
    action: "逐位置求和 → 所有 rank",
    formula: "yᵣ[i] = Σₛ xₛ[i]",
    shape: "每 rank [N] → 每 rank [N]",
    purpose: "合并各 rank 的局部贡献；每个参与者得到相同结果。",
  },
  {
    id: "all-gather",
    title: "AllGather",
    action: "按 rank 拼接 → 所有 rank",
    formula: "yᵣ = concat(x₀, …, xₚ₋₁)",
    shape: "每 rank [n] → 每 rank [p·n]",
    purpose: "每个 rank 获得按来源 rank 排列的完整分片。",
  },
  {
    id: "reduce-scatter",
    title: "ReduceScatter",
    action: "逐位置求和 → 各收一片",
    formula: "yᵣ = (Σₛ xₛ)[r·n : (r+1)·n]",
    shape: "每 rank [p·n] → 每 rank [n]",
    purpose: "先合并局部贡献，再让 rank r 持有第 r 片。",
  },
  {
    id: "all-to-all",
    title: "AllToAll",
    action: "按目的地交换 → 按来源重排",
    formula: "yᵣ[s·n:(s+1)·n] = xₛ[r·n:(r+1)·n]",
    shape: "每 rank [p·n] → 每 rank [p·n]",
    purpose: "例如把 Token 分送到专家组；交换分片，不做求和。",
  },
  {
    id: "broadcast",
    title: "Broadcast",
    action: "复制 root 的缓冲区",
    formula: "yᵣ = xroot",
    shape: "root [N] → 每 rank [N]",
    purpose: "让通信组的所有参与者得到 root 的同一份数据。",
  },
  {
    id: "reduce",
    title: "Reduce",
    action: "逐位置求和 → 仅 root",
    formula: "yroot[i] = Σₛ xₛ[i]",
    shape: "每 rank [N] → 仅 root [N]",
    purpose: "只让指定 root 接收归约结果；其他 rank 没有结果缓冲区。",
  },
  {
    id: "gather",
    title: "Gather",
    action: "按 rank 拼接 → 仅 root",
    formula: "yroot = concat(x₀, …, xₚ₋₁)",
    shape: "每 rank [n] → 仅 root [p·n]",
    purpose: "把各 rank 的原始片段汇总到 root，保留来源顺序。",
  },
  {
    id: "scatter",
    title: "Scatter",
    action: "切分 root → 各收一片",
    formula: "yᵣ = xroot[r·n:(r+1)·n]",
    shape: "root [p·n] → 每 rank [n]",
    purpose: "root 按目的 rank 切片分发；元素保持原值。",
  },
  {
    id: "send-recv",
    title: "Send / Recv",
    action: "指定发送者 → 指定接收者",
    formula: "yreceiver = xsender",
    shape: "sender [N] → receiver [N]",
    purpose: "演示一对匹配的双边发送与接收；其余 rank 不参与。",
  },
];

/** Calculator for numeric buffers, with undefined denoting no receive result. */
export function runCollective(
  operation: CollectiveOperation,
  inputs: RankBuffers,
  options: CollectiveOptions = {},
): RankBuffers {
  const p = inputs.length;
  if (
    ![2, 4].includes(p) ||
    !collectiveOperations.some((op) => op.id === operation)
  )
    throw Error("请选择已有通信原语与 2/4 个 rank");
  const rank = (value: number) =>
    Number.isInteger(value) && value >= 0 && value < p;
  for (const value of Object.values(options))
    if (value !== undefined && !rank(value))
      throw Error("root / peer 必须是通信组内有效 rank");
  for (const buffer of inputs)
    if (
      buffer !== undefined &&
      (!Array.isArray(buffer) ||
        !buffer.length ||
        !buffer.every(Number.isFinite))
    )
      throw Error("缓冲区必须是非空有限数值数组");
  const source = (index: number): number[] => {
    const buffer = inputs[index];
    if (!buffer) throw Error(`rank ${index} 缺少发送缓冲区`);
    return buffer;
  };
  const root = options.root ?? 0;
  const outputs: RankBuffers = Array.from({ length: p }, () => undefined);
  if (operation === "send-recv") {
    const sender = options.sender ?? 0,
      receiver = options.receiver ?? 1;
    if (sender === receiver) throw Error("发送者与接收者必须不同");
    outputs[receiver] = [...source(sender)];
    return outputs;
  }
  if (operation === "broadcast" || operation === "scatter") {
    const buffer = source(root);
    if (operation === "broadcast") return outputs.map(() => [...buffer]);
    if (buffer.length % p) throw Error("Scatter 总元素数必须能被 rank 数整除");
    const n = buffer.length / p;
    return outputs.map((_, r) => buffer.slice(r * n, (r + 1) * n));
  }
  const buffers = inputs.map((_, r) => source(r));
  const count = buffers[0].length;
  if (buffers.some((buffer) => buffer.length !== count))
    throw Error("参与 rank 必须提供匹配的 count 与数值类型");
  if (operation === "all-gather" || operation === "gather") {
    const concatenated = buffers.flat();
    if (operation === "all-gather") return outputs.map(() => [...concatenated]);
    outputs[root] = concatenated;
    return outputs;
  }
  if (operation === "all-to-all") {
    if (count % p) throw Error("AllToAll 输入必须可切成 p 个等长目的地片段");
    const n = count / p;
    return outputs.map((_, destination) =>
      buffers.flatMap((buffer) =>
        buffer.slice(destination * n, (destination + 1) * n),
      ),
    );
  }
  const sum = buffers[0].map((_, i) =>
    buffers.reduce((value, buffer) => value + buffer[i], 0),
  );
  if (!sum.every(Number.isFinite)) throw Error("求和结果超出有限数值范围");
  if (operation === "all-reduce") return outputs.map(() => [...sum]);
  if (operation === "reduce") {
    outputs[root] = sum;
    return outputs;
  }
  if (count % p) throw Error("ReduceScatter 输入元素数必须能被 rank 数整除");
  const n = count / p;
  return outputs.map((_, r) => sum.slice(r * n, (r + 1) * n));
}

/** Fixed semantic routes: teaching schedule, not an NCCL network algorithm. */
export function buildCollectiveDemo(
  operation: CollectiveOperation,
  ranks: 2 | 4,
  options: CollectiveOptions = {},
): CollectiveDemo {
  const normalized = {
    root: options.root ?? 0,
    sender: options.sender ?? 0,
    receiver: options.receiver ?? 1,
  };
  const root = normalized.root;
  const fullLength = ranks * 2;
  const inputs: RankBuffers = Array.from({ length: ranks }, (_, r) => {
    if (operation === "broadcast")
      return r === root ? [r * 10 + 1, r * 10 + 2] : undefined;
    if (operation === "scatter")
      return r === root
        ? Array.from({ length: fullLength }, (_, i) => i + 1)
        : undefined;
    if (operation === "send-recv")
      return r === normalized.sender ? [r * 10 + 1, r * 10 + 2] : undefined;
    if (operation === "all-to-all")
      return Array.from(
        { length: fullLength },
        (_, i) => r * 100 + Math.floor(i / 2) * 10 + (i % 2),
      );
    const length =
      operation === "all-gather" || operation === "gather" ? 2 : fullLength;
    return Array.from({ length }, (_, i) => r * 10 + i + 1);
  });
  const outputs = runCollective(operation, inputs, normalized);
  const transfers: CollectiveTransfer[] = [];
  const move = (
    from: number | "sum",
    to: number | "sum",
    values: number[],
    label: string,
    start = 1,
    end = 3,
  ) =>
    transfers.push({
      id: `${operation}:${from}:${to}:${transfers.length}`,
      from,
      to,
      values: [...values],
      label,
      start,
      end,
    });
  if (["all-reduce", "reduce-scatter", "reduce"].includes(operation)) {
    inputs.forEach((buffer, r) =>
      move(r, "sum", buffer!, `x${r} · 逐位置贡献`, 1, 2),
    );
    outputs.forEach((buffer, r) => {
      if (buffer)
        move(
          "sum",
          r,
          buffer,
          operation === "reduce-scatter"
            ? `sum 的第 ${r} 片`
            : "sum · 归约结果",
          2,
          3,
        );
    });
  } else if (operation === "all-to-all") {
    inputs.forEach((buffer, source) => {
      for (let destination = 0; destination < ranks; destination++)
        move(
          source,
          destination,
          buffer!.slice(destination * 2, destination * 2 + 2),
          `源 ${source} → 目的 ${destination}`,
        );
    });
  } else if (operation === "all-gather" || operation === "gather") {
    inputs.forEach((buffer, source) => {
      for (let destination = 0; destination < ranks; destination++)
        if (operation === "all-gather" || destination === root)
          move(
            source,
            destination,
            buffer!,
            `源 rank ${source} · 拼接位置 ${source}`,
          );
    });
  } else if (operation === "broadcast" || operation === "scatter") {
    outputs.forEach((buffer, destination) =>
      move(
        root,
        destination,
        buffer!,
        operation === "scatter"
          ? `目的 ${destination} 的切片`
          : `复制 root ${root}`,
      ),
    );
  } else
    move(
      normalized.sender,
      normalized.receiver,
      inputs[normalized.sender]!,
      `Send ${normalized.sender} / Recv ${normalized.receiver}`,
    );
  return {
    operation,
    ranks,
    inputs,
    outputs,
    options: normalized,
    transfers,
    steps: [
      "检查输入与分片",
      "发起匹配通信",
      ["all-reduce", "reduce-scatter", "reduce"].includes(operation)
        ? "逐位置求和 / 分发"
        : "沿路径搬运",
      "提交接收缓冲区",
      "核对输出与 shape",
    ],
  };
}

export function collectiveSnapshot(demo: CollectiveDemo, position: number) {
  const arrived = position >= 3;
  return {
    outputs: demo.outputs.map((buffer) =>
      arrived && buffer ? [...buffer] : undefined,
    ),
    transfers: demo.transfers.map((transfer) => ({
      ...transfer,
      progress: Math.max(
        0,
        Math.min(
          1,
          (position - transfer.start) / (transfer.end - transfer.start),
        ),
      ),
      active: position >= transfer.start && position < transfer.end,
    })),
  };
}
