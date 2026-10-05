export type ParallelStrategy = "tp" | "dp" | "ep" | "pp" | "cp" | "sp";
export type Matrix = number[][];

function matrixShape(matrix: Matrix): [number, number] {
  const width = matrix[0]?.length ?? 0;
  if (
    !matrix.length ||
    !width ||
    matrix.some(
      (row) =>
        row.length !== width || row.some((value) => !Number.isFinite(value)),
    )
  )
    throw new Error("矩阵维度必须非空、规则且包含有限数值");
  return [matrix.length, width];
}
function splitSize(size: number, ranks: number) {
  if (ranks !== 2 && ranks !== 4) throw new Error("教学 GPU 数只支持 2 或 4");
  if (!Number.isInteger(size) || size <= 0 || size % ranks !== 0)
    throw new Error(`切分轴 ${size} 必须被 GPU 数 ${ranks} 整除`);
  return size / ranks;
}
const copy = (matrix: Matrix): Matrix => matrix.map((row) => [...row]);
const rows = (matrix: Matrix, rank: number, width: number) =>
  copy(matrix.slice(rank * width, (rank + 1) * width));
const sumMatrices = (matrices: Matrix[]): Matrix =>
  matrices[0].map((row, i) =>
    row.map((_, j) => matrices.reduce((sum, matrix) => sum + matrix[i][j], 0)),
  );
export function matmul(a: Matrix, b: Matrix): Matrix {
  const [count, inner] = matrixShape(a),
    [bRows, width] = matrixShape(b);
  if (inner !== bRows) throw new Error("矩阵乘法的收缩维度不匹配");
  return Array.from({ length: count }, (_, i) =>
    Array.from({ length: width }, (_, j) =>
      a[i].reduce((sum, value, k) => sum + value * b[k][j], 0),
    ),
  );
}
export function tensorParallel(x: Matrix, a: Matrix, b: Matrix, ranks: number) {
  const [, features] = matrixShape(a);
  const local = splitSize(features, ranks);
  if (matrixShape(b)[0] !== features)
    throw new Error("两次投影的中间维度不匹配");
  const shards = Array.from({ length: ranks }, (_, rank) => {
    const ar = a.map((row) => row.slice(rank * local, (rank + 1) * local));
    const br = rows(b, rank, local),
      hidden = matmul(x, ar);
    return { rank, a: ar, b: br, hidden, partial: matmul(hidden, br) };
  });
  return {
    shards,
    hidden: shards[0].hidden.map((_, row) =>
      shards.flatMap((shard) => shard.hidden[row]),
    ),
    output: sumMatrices(shards.map((shard) => shard.partial)),
  };
}
export function dataParallel(x: Matrix, weight: Matrix, ranks: number) {
  const local = splitSize(matrixShape(x)[0], ranks);
  const replicas = Array.from({ length: ranks }, (_, rank) => {
    const input = rows(x, rank, local);
    return { rank, input, weight: copy(weight), output: matmul(input, weight) };
  });
  return {
    replicas,
    output: replicas.flatMap((replica) => replica.output),
    collectives: [] as string[],
  };
}
export function expertParallel(x: Matrix, ranks: number, topK: number) {
  matrixShape(x);
  const localExperts = splitSize(4, ranks);
  if (topK !== 1 && topK !== 2)
    throw new Error("教学路由只支持 top-k = 1 或 2");
  const routes = x.flatMap((value, token) =>
    Array.from({ length: topK }, (_, k) => {
      const expert = (token + k) % 4;
      return {
        token,
        expert,
        from: Math.min(ranks - 1, Math.floor((token * ranks) / x.length)),
        to: Math.floor(expert / localExperts),
        weight: topK === 1 ? 1 : k === 0 ? 0.75 : 0.25,
        input: [...value],
        output: value.map((v) => v * (expert + 1)),
      };
    }),
  );
  const output = x.map((value, token) =>
    value.map((_, column) =>
      routes
        .filter((route) => route.token === token)
        .reduce((sum, route) => sum + route.weight * route.output[column], 0),
    ),
  );
  return { routes, output, localExperts };
}
export function pipelineParallel(x: Matrix, ranks: number) {
  matrixShape(x);
  const layersPerRank = splitSize(4, ranks);
  const layerAdds = Array.from({ length: ranks }, (_, rank) =>
    Array.from(
      { length: layersPerRank },
      (_, i) => rank * layersPerRank + i + 1,
    ).reduce((a, b) => a + b, 0),
  );
  const states = x.map((input) => {
    const result = [[...input]];
    for (const add of layerAdds)
      result.push(result.at(-1)!.map((v) => v + add));
    return result;
  });
  const schedule = Array.from({ length: x.length + ranks - 1 }, (_, slot) =>
    Array.from({ length: ranks }, (_, rank) => {
      const batch = slot - rank;
      return batch >= 0 && batch < x.length ? batch : null;
    }),
  );
  const handoffs = x.flatMap((_, batch) =>
    Array.from({ length: ranks - 1 }, (_, rank) => ({
      batch,
      from: rank,
      to: rank + 1,
      slot: batch + rank,
      value: [...states[batch][rank + 1]],
    })),
  );
  return {
    states,
    schedule,
    handoffs,
    layersPerRank,
    output: states.map((state) => state.at(-1)!),
    bubbleFraction: (ranks - 1) / (x.length + ranks - 1),
  };
}
export interface AttentionAccumulator {
  m: number;
  l: number;
  o: number[];
  value: number[];
  visited: number[];
}
/** Stable online softmax. Each query remains with its owner while KV blocks visit in a ring. */
export function contextAttention(
  q: number[],
  k: number[],
  v: Matrix,
  ranks: number,
  causal: boolean,
) {
  const local = splitSize(q.length, ranks);
  if (
    k.length !== q.length ||
    matrixShape(v)[0] !== q.length ||
    q.some((n) => !Number.isFinite(n)) ||
    k.some((n) => !Number.isFinite(n))
  )
    throw new Error("Q、K、V 序列维度不匹配");
  let accumulators: AttentionAccumulator[] = q.map(() => ({
    m: -Infinity,
    l: 0,
    o: Array(v[0].length).fill(0),
    value: Array(v[0].length).fill(0),
    visited: [],
  }));
  const rounds: AttentionAccumulator[][] = [];
  for (let round = 0; round < ranks; round++) {
    accumulators = accumulators.map((previous, query) => {
      const owner = Math.floor(query / local),
        block = (owner - round + ranks) % ranks;
      const indices = Array.from(
        { length: local },
        (_, i) => block * local + i,
      ).filter((i) => !causal || i <= query);
      const logits = indices.map((i) => q[query] * k[i]); // head dimension d=1 => sqrt(d)=1
      const m = Math.max(previous.m, ...logits);
      const rescale = previous.l === 0 ? 0 : Math.exp(previous.m - m);
      const exps = logits.map((score) => Math.exp(score - m));
      const l =
        previous.l * rescale + exps.reduce((sum, value) => sum + value, 0);
      const o = previous.o.map(
        (value, col) =>
          value * rescale +
          indices.reduce((sum, index, j) => sum + exps[j] * v[index][col], 0),
      );
      return {
        m,
        l,
        o,
        value: o.map((value) => (l ? value / l : 0)),
        visited: [...previous.visited, block],
      };
    });
    rounds.push(accumulators);
  }
  const weights = q.map((query, index) => {
    const scores = k.map((key, pos) =>
      causal && pos > index ? -Infinity : query * key,
    );
    const max = Math.max(...scores),
      exps = scores.map((score) => Math.exp(score - max)),
      sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((value) => value / sum);
  });
  return {
    rounds,
    weights,
    output: accumulators.map((acc) => acc.value),
    local,
  };
}
export function sequenceParallel(
  x: Matrix,
  a: Matrix,
  b: Matrix,
  ranks: number,
) {
  const local = splitSize(matrixShape(x)[0], ranks);
  // Teaching RMSNorm uses epsilon=0 and unit scale; inputs are bounded away from zero.
  const normalized = x.map((row) => {
    const rms = Math.sqrt(row.reduce((sum, v) => sum + v * v, 0) / row.length);
    if (!rms) throw new Error("教学 RMSNorm 输入必须包含非零通道");
    return row.map((v) => v / rms);
  });
  const normShards = Array.from({ length: ranks }, (_, rank) =>
    rows(normalized, rank, local),
  );
  const gathered = Array.from({ length: ranks }, () =>
    normShards.flatMap((shard) => copy(shard)),
  );
  const tp = tensorParallel(normalized, a, b, ranks);
  const scattered = Array.from({ length: ranks }, (_, rank) =>
    rows(tp.output, rank, local),
  );
  return { normalized, normShards, gathered, tp, scattered, output: tp.output };
}

export interface ParallelBuffer {
  name: string;
  matrix: Matrix;
  shape: string;
  axes?: string;
}
export interface ParallelRank {
  rank: number;
  ownership: string;
  buffers: ParallelBuffer[];
}
export interface ParallelStage {
  title: string;
  explanation: string;
  communication: string;
  ranks: ParallelRank[];
}
export interface ParallelPoint {
  x: number;
  y: number;
  label: string;
  visible?: boolean;
}
export interface ParallelObject {
  id: string;
  kind: "activation" | "token" | "kv" | "partial";
  points: ParallelPoint[];
}
export interface ParallelExample {
  strategy: ParallelStrategy;
  title: string;
  subtitle: string;
  split: string;
  global: string;
  derivation: string[];
  stages: ParallelStage[];
  objects: ParallelObject[];
  output: Matrix;
  benefit: string;
  cost: string;
  limitation: string;
  schedule?: (number | null)[][];
  bubbleFraction?: number;
  routes?: ReturnType<typeof expertParallel>["routes"];
  attention?: ReturnType<typeof contextAttention>;
}
export interface ParallelOptions {
  size?: number;
  topK?: number;
  microbatches?: number;
  causal?: boolean;
}
const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(3));
const vectorLabel = (m: Matrix) =>
  m.length
    ? `[${m[0].map(fmt).join(",")}]${m.length > 1 ? ` × ${m.length}行` : ""}`
    : "空";
const rankX = (rank: number, ranks: number) => 105 + rank * (630 / (ranks - 1));
const point = (
  rank: number,
  ranks: number,
  label: string,
  y = 150,
): ParallelPoint => ({ x: rankX(rank, ranks), y, label });
const buffer = (
  name: string,
  matrix: Matrix,
  shape?: string,
  axes?: string,
): ParallelBuffer => ({
  name,
  matrix,
  shape: shape ?? `[${matrix.length},${matrix[0]?.length ?? 0}]`,
  axes,
});
const allRanks = (
  ranks: number,
  ownership: (rank: number) => string,
  buffers: (rank: number) => ParallelBuffer[],
) =>
  Array.from({ length: ranks }, (_, rank) => ({
    rank,
    ownership: ownership(rank),
    buffers: buffers(rank),
  }));
const projections = (features: number): [Matrix, Matrix] => [
  Array.from({ length: 2 }, (_, r) =>
    Array.from({ length: features }, (_, c) =>
      r === 0 ? (c % 3) + 1 : ((c + 1) % 3) + 1,
    ),
  ),
  Array.from({ length: features }, (_, r) => [(r % 2) + 1, ((r + 1) % 2) + 1]),
];

export function sampleParallelObject(
  object: ParallelObject,
  position: number,
): ParallelPoint {
  const p = Math.max(
    0,
    Math.min(
      object.points.length - 1,
      Number.isFinite(position) ? position : 0,
    ),
  );
  const i = Math.floor(p),
    a = object.points[i],
    b = object.points[Math.min(i + 1, object.points.length - 1)],
    fraction = p - i;
  return {
    ...a,
    x: a.x + (b.x - a.x) * fraction,
    y: a.y + (b.y - a.y) * fraction,
  };
}

export function buildParallelExample(
  strategy: ParallelStrategy,
  ranks: number,
  options: ParallelOptions = {},
): ParallelExample {
  splitSize(4, ranks);
  const size = options.size ?? 4;
  if (size !== 4 && size !== 8)
    throw new Error("教学切分轴只支持 4 或 8，确保可以整除 2/4 GPU");
  const [a, b] = projections(size),
    x: Matrix = [
      [1, 2],
      [2, 1],
    ];
  const base = {
    strategy,
    title: "",
    subtitle: "",
    split: "",
    global: "",
    derivation: [] as string[],
    stages: [] as ParallelStage[],
    objects: [] as ParallelObject[],
    output: [] as Matrix,
    benefit: "",
    cost: "",
    limitation: "",
  };
  if (strategy === "tp") {
    const result = tensorParallel(x, a, b, ranks),
      f = size / ranks;
    const rankBuffers = (stage: number) =>
      allRanks(
        ranks,
        (r) => `F 轴 ${r * f}…${(r + 1) * f - 1}`,
        (r) =>
          stage === 0
            ? [
                buffer("X（复制）", x),
                buffer("Aᵣ（列切）", result.shards[r].a),
                buffer("Bᵣ（行切）", result.shards[r].b),
              ]
            : stage === 1
              ? [buffer("Zᵣ = X Aᵣ", result.shards[r].hidden)]
              : stage === 2 || stage === 3
                ? [buffer("Pᵣ = Zᵣ Bᵣ", result.shards[r].partial)]
                : [buffer("Y = ΣᵣPᵣ", result.output)],
      );
    return {
      ...base,
      title: "TP · 张量并行",
      subtitle: "同一层的矩阵由多张 GPU 合作算完",
      split: "先切 A 的输出列 F，再切 B 的输入行 F；X 在每个 rank 复制。",
      global: `X[2,2] · A[2,${size}] · B[${size},2] → Y[2,2]`,
      derivation: [
        `F/${ranks} = ${size}/${ranks} = ${f}`,
        `[2,2] × [2,${f}] → Zᵣ[2,${f}]；[2,${f}] × [${f},2] → Pᵣ[2,2]`,
        `Y = XAB = Σᵣ (XAᵣ)Bᵣ；Y[0,0] = ${result.shards.map((s) => s.partial[0][0]).join(" + ")} = ${result.output[0][0]}`,
      ],
      stages: [
        {
          title: "部署权重分片",
          explanation:
            "相同输入，互补的权重列/行。F 是两次投影之间的教学通道轴。",
          communication: "部署后权重常驻；本阶段无通信",
          ranks: rankBuffers(0),
        },
        {
          title: "列并行投影",
          explanation:
            "每卡计算自己的 Zᵣ；拼接 Zᵣ 才是完整中间激活，但下一层已按相同 F 轴分片，通常不必先拼接。",
          communication: "本地矩阵乘法",
          ranks: rankBuffers(1),
        },
        {
          title: "行并行部分和",
          explanation: "每个 Pᵣ 都是完整 Y 形状，只包含一部分通道的贡献。",
          communication: "下一阶段 AllReduce(sum)",
          ranks: rankBuffers(2),
        },
        {
          title: "按元素归约",
          explanation:
            "将同形状 Pᵣ 求和得到 Y。图中汇合点只帮助理解求和语义，不代表 AllReduce 的固定 root。各 rank 的输出尚待到达。",
          communication: "AllReduce · 教学求和阶段",
          ranks: rankBuffers(3),
        },
        {
          title: "各 rank 得到完整输出",
          explanation:
            "按元素相加，各 rank 得到同一个完整 Y；这里不是把 Pᵣ 拼接。",
          communication: "AllReduce · 求和并复制结果",
          ranks: rankBuffers(4),
        },
      ],
      objects: Array.from({ length: ranks }, (_, r) => ({
        id: `tp-${r}`,
        kind: "partial",
        points: [
          point(r, ranks, `X ${vectorLabel(x)}`, 110),
          point(r, ranks, `Z${r} ${vectorLabel(result.shards[r].hidden)}`, 155),
          point(
            r,
            ranks,
            `P${r} ${vectorLabel(result.shards[r].partial)}`,
            205,
          ),
          { x: 420, y: 270 + r * 22, label: `Y ${vectorLabel(result.output)}` },
          point(r, ranks, `Y ${vectorLabel(result.output)}`, 150),
        ],
      })),
      output: result.output,
      benefit: "单卡只保存 F 轴的权重分片，让单个模型跨卡容纳。",
      cost: "层内部分结果必须通信；小矩阵、低带宽或同步等待会抵消收益。",
      limitation:
        "这是一对线性层的推理前向示例；省略激活函数、偏置、反向传播与真实算子调度。模型头数/通道等维度也须满足实现约束。",
    };
  }
  if (strategy === "dp") {
    const input = Array.from({ length: size }, (_, t) => [t + 1, 1]),
      weight: Matrix = [
        [1, 2],
        [0, 1],
      ],
      result = dataParallel(input, weight, ranks),
      local = size / ranks;
    return {
      ...base,
      title: "DP · 数据并行",
      subtitle: "每张 GPU 有完整模型，处理不同请求",
      split: "切请求/批次 B 轴；模型 W 完整复制，单条请求不拆开。",
      global: `X[B=${size},H=2] · W[2,2] → Y[${size},2]`,
      derivation: [
        `B/${ranks} = ${size}/${ranks} = ${local}`,
        `rank r: Xᵣ[${local},2] × W[2,2] = Yᵣ[${local},2]`,
        `首请求 [1,1] × [[1,2],[0,1]] = [1,3]；合并请求顺序后输出 [${size},2]`,
      ],
      stages: [
        {
          title: "分派独立请求",
          explanation:
            "请求调度器按原始请求 ID 分配任务；各 GPU 拥有相同且固定的推理权重。",
          communication: "调度器 → rank；没有 rank 间 collective",
          ranks: allRanks(
            ranks,
            (r) => `请求 ${r * local}…${(r + 1) * local - 1}`,
            (r) => [
              buffer("Xᵣ", result.replicas[r].input),
              buffer("W（完整副本）", weight),
            ],
          ),
        },
        {
          title: "副本本地前向",
          explanation:
            "每个副本只依赖自己收到的请求，其他副本的输入不会改变它的输出。",
          communication: "本地矩阵乘法；无 AllReduce",
          ranks: allRanks(
            ranks,
            (r) => `请求 ${r * local}…${(r + 1) * local - 1}`,
            (r) => [buffer("Yᵣ", result.replicas[r].output)],
          ),
        },
        {
          title: "各自返回请求结果",
          explanation:
            "结果返回客户端并按请求 ID 关联；图中的返回汇合点是服务端，不是 GPU 的求和算子。",
          communication: "rank → 请求服务端；无梯度同步",
          ranks: allRanks(
            ranks,
            (r) => "完整模型副本",
            (r) => [buffer("独立 Yᵣ", result.replicas[r].output)],
          ),
        },
      ],
      objects: Array.from({ length: ranks }, (_, r) => ({
        id: `dp-${r}`,
        kind: "activation",
        points: [
          {
            x: 30,
            y: 180 + r * 28,
            label: `R${r * local}… ${vectorLabel(result.replicas[r].input)}`,
          },
          point(
            r,
            ranks,
            `Y${r} ${vectorLabel(result.replicas[r].output)}`,
            145,
          ),
          {
            x: 810,
            y: 200 + r * 28,
            label: `返回 ${vectorLabel(result.replicas[r].output)}`,
          },
        ],
      })),
      output: result.output,
      benefit: "多个请求可以由不同副本并发服务，提高服务容量。",
      cost: "每个副本都需容纳完整模型，且服务端需做负载分配。",
      limitation:
        "固定权重推理没有常规梯度 AllReduce；训练 DDP 才需同步梯度。MoE 的 DP+EP 部署会另有专家通信。此批次每行代表一条单 token 请求。",
    };
  }
  if (strategy === "ep") {
    const input = Array.from({ length: size }, (_, t) => [t + 1, 1]),
      topK = options.topK ?? 2,
      result = expertParallel(input, ranks, topK),
      local = size / ranks;
    const received = (r: number) =>
      result.routes.filter((route) => route.to === r);
    return {
      ...base,
      title: "EP · 专家并行",
      subtitle: "Token 按路由访问专家，再回到原持有者",
      split: "4 个专家按专家 E 轴分卡；每个 token 保留来源 rank 和路由权重。",
      global: `X[T=${size},H=2] · E=4 · top-k=${topK} → Y[${size},2]`,
      derivation: [
        `每卡专家 = 4/${ranks} = ${result.localExperts}；发送记录数 = T×k = ${size * topK}`,
        "教学专家 e: fₑ(x) = (e+1)x；路由选择 t mod 4 及其下一位专家",
        topK === 1
          ? "k=1：yₜ = fₑ(xₜ)，权重 1"
          : "k=2：yₜ = 0.75 fₑ(xₜ) + 0.25 f₍ₑ₊₁₎(xₜ)；t0 → 0.75×[1,1]+0.25×[2,2]=[1.25,1.25]",
      ],
      stages: [
        {
          title: "路由与来源登记",
          explanation:
            "来源 rank 保存 token ID 和 top-k 权重；同一个 token 可形成多条发送记录。",
          communication: "路由计算在本地",
          ranks: allRanks(
            ranks,
            (r) =>
              `专家 ${r * result.localExperts}…${(r + 1) * result.localExperts - 1}`,
            (r) => [buffer("来源 Xᵣ", rows(input, r, local))],
          ),
        },
        {
          title: "分发到专家 owner",
          explanation:
            "可变长度 token 缓冲区按目的 rank 重排，目的 rank 可以拥有多个专家；同卡专家不需要经过网络。",
          communication: "AllToAll · dispatch（实际实现可用变长/专用算子）",
          ranks: allRanks(
            ranks,
            (r) =>
              `专家 ${r * result.localExperts}…${(r + 1) * result.localExperts - 1}`,
            (r) => [
              buffer(
                "收到的专家输入",
                received(r).map((route) => route.input),
                `[Nᵣ=${received(r).length},2]`,
              ),
            ],
          ),
        },
        {
          title: "各专家本地计算",
          explanation: "每条记录按其专家编号执行 fₑ，尚未乘路由权重。",
          communication: "本地专家前向",
          ranks: allRanks(
            ranks,
            (r) => "路由记录保留 token/expert ID",
            (r) => [
              buffer(
                "专家结果 fₑ(x)",
                received(r).map((route) => route.output),
              ),
            ],
          ),
        },
        {
          title: "返回来源并加权合并",
          explanation:
            "反向 AllToAll 返回结果；来源 rank 按 token ID 执行加权和，还原原 token 顺序。",
          communication: "AllToAll · combine + 本地加权求和",
          ranks: allRanks(
            ranks,
            (r) => `来源 token ${r * local}…${(r + 1) * local - 1}`,
            (r) => [buffer("Yᵣ", rows(result.output, r, local))],
          ),
        },
      ],
      objects: result.routes.map((route, i) => ({
        id: `ep-t${route.token}-e${route.expert}`,
        kind: "token",
        points: [
          point(
            route.from,
            ranks,
            `t${route.token}→e${route.expert} w${route.weight}`,
            135 + i * 12,
          ),
          point(
            route.to,
            ranks,
            `t${route.token} e${route.expert}: ${vectorLabel([route.input])}`,
            145 + i * 12,
          ),
          point(
            route.to,
            ranks,
            `e${route.expert}(t${route.token}) ${vectorLabel([route.output])}`,
            145 + i * 12,
          ),
          point(
            route.from,
            ranks,
            `t${route.token} w${route.weight}: ${vectorLabel([route.output])}`,
            135 + i * 12,
          ),
        ],
      })),
      output: result.output,
      routes: result.routes,
      benefit: "每卡只持有部分专家参数，token 只执行被选中的专家。",
      cost: "Token 分发、返回、重排和负载不均；top-k 增大后流量和计算记录随之增加。",
      limitation: `此例不丢 token、不限制专家容量。EP 组可覆盖 attention 的 DP/TP 组：同样 ${ranks} 张卡可以 attention DP=${ranks}, TP=1, EP=${ranks}；不能再把 DP×TP×EP 全乘一遍。训练还需反向路由。`,
    };
  }
  if (strategy === "pp") {
    const count = options.microbatches ?? 2;
    if (![1, 2, 4].includes(count))
      throw new Error("教学微批数只支持 1、2 或 4");
    const input = Array.from({ length: count }, (_, t) => [t + 1, 1]),
      result = pipelineParallel(input, ranks);
    const stages: ParallelStage[] = [
      {
        title: "层块分配",
        explanation:
          "4 个连续教学层分别加 1、2、3、4；每个微批携带 [1,2] 激活，依次经过全部层。",
        communication: "权重常驻各 stage",
        ranks: allRanks(
          ranks,
          (r) =>
            `层 L${r * result.layersPerRank + 1}…L${(r + 1) * result.layersPerRank}`,
          () => [],
        ),
      },
    ];
    result.schedule.forEach((slot, s) =>
      stages.push({
        title: `流水线时隙 ${s + 1}`,
        explanation: `当前 ${slot.filter((v) => v !== null).length}/${ranks} 个 stage 工作；空白格是填充/排空气泡。每一格假定相同计算时间。`,
        communication:
          s === result.schedule.length - 1
            ? "最后一个微批完成"
            : "Send/Recv · 下一 stage 的激活交接",
        ranks: allRanks(
          ranks,
          (r) =>
            `层 L${r * result.layersPerRank + 1}…L${(r + 1) * result.layersPerRank}`,
          (r) =>
            slot[r] === null
              ? []
              : [
                  buffer(`μ${slot[r]} 输入`, [result.states[slot[r]!][r]]),
                  buffer(`μ${slot[r]} 输出`, [result.states[slot[r]!][r + 1]]),
                ],
        ),
      }),
    );
    return {
      ...base,
      title: "PP · 流水线并行",
      subtitle: "一条请求跨过层块，多条微批错位前进",
      split: "沿模型层 L 轴分块；同一微批的依赖仍按层先后执行。",
      global: `4 层 · ${count} 个微批 · 每个激活 [1,2] → 输出 [${count},2]`,
      derivation: [
        `每卡层数 = 4/${ranks} = ${result.layersPerRank}`,
        "μ0: [1,1] +1 +2 +3 +4 = [11,11]；Send/Recv 传激活，不传整套权重",
        `理想等时前向时隙 = m+p−1 = ${count}+${ranks}−1 = ${result.schedule.length}；气泡占比 (p−1)/(m+p−1) = ${(result.bubbleFraction * 100).toFixed(1)}%`,
      ],
      stages,
      objects: input.map((_, batch) => ({
        id: `pp-${batch}`,
        kind: "activation",
        points: stages.map((_, stage) => {
          const completed = Math.max(0, Math.min(ranks, stage - batch));
          return stage <= batch
            ? {
                x: 30,
                y: 160 + batch * 35,
                label: `μ${batch} ${vectorLabel([input[batch]])}`,
              }
            : {
                x: rankX(Math.min(completed - 1, ranks - 1), ranks),
                y: 160 + batch * 35,
                label: `μ${batch} ${vectorLabel([result.states[batch][completed]])}`,
              };
        }),
      })),
      output: result.output,
      schedule: result.schedule,
      bubbleFraction: result.bubbleFraction,
      benefit: "模型的连续层分布到不同 GPU；微批交叠可以减少空闲。",
      cost: "激活交接、阶段负载不均和流水线气泡；单请求仍串行跨 stage。",
      limitation:
        "气泡公式仅适用于此等时、仅前向教学调度，不是 GPU 利用率实测。推理 Decode 还依赖前一 token，训练会增加反向与不同调度。",
    };
  }
  if (strategy === "cp") {
    const q = Array.from({ length: size }, (_, t) => (t + 1) / size),
      k = Array.from({ length: size }, (_, t) => t / 2),
      v = Array.from({ length: size }, (_, t) => [t + 1, 1]),
      causal = options.causal ?? true,
      result = contextAttention(q, k, v, ranks, causal),
      local = size / ranks;
    const stages: ParallelStage[] = [
      {
        title: "Q/KV 按位置分片",
        explanation:
          "Q 固定在其位置 owner。KV 块依环从 r 移到 r+1，逻辑上让每个 Q 访问整条序列的有效 K/V。",
        communication: "尚无通信",
        ranks: allRanks(
          ranks,
          (r) => `Q 位置 ${r * local}…${(r + 1) * local - 1}`,
          (r) => [
            buffer(
              "Qᵣ",
              q.slice(r * local, (r + 1) * local).map((v) => [v]),
            ),
            buffer(
              "Kᵣ",
              k.slice(r * local, (r + 1) * local).map((v) => [v]),
            ),
            buffer("Vᵣ", rows(v, r, local)),
          ],
        ),
      },
    ];
    result.rounds.forEach((round, j) =>
      stages.push({
        title: `KV 轮次 ${j + 1}/${ranks}`,
        explanation: `每个 rank 已处理 ${j + 1} 个 KV 块；以在线 (m,l,o) 累积同一个 softmax。${causal ? "全局位置 j>i 的未来 token 被遮罩。" : "此选择使用无因果遮罩的完整注意力。"}`,
        communication:
          j === 0
            ? "本地 KV 块；下一轮环形 Send/Recv"
            : j === ranks - 1
              ? "最后一个 KV 块到达；全局 softmax 完成"
              : "环形 Send/Recv · KV 块前进一跳",
        ranks: allRanks(
          ranks,
          (r) => `Q 固定；收到 KV 块 ${(r - j + ranks) % ranks}`,
          (r) => [
            buffer(
              "累计 [m,l]",
              round
                .slice(r * local, (r + 1) * local)
                .map((acc) => [acc.m, acc.l]),
            ),
            buffer(
              j === ranks - 1 ? "Oᵣ = o/l（完整）" : "o/l（未完整）",
              round.slice(r * local, (r + 1) * local).map((acc) => acc.value),
            ),
          ],
        ),
      }),
    );
    stages.push({
      title: "保留本地 Q 的完整结果",
      explanation:
        "每个 Q 都已看到全部 KV 块；输出仍按 Q 位置分片。独立计算本地注意力或平均各块 softmax 会产生不同结果。",
      communication: "本地输出；不要求 Gather 所有 Q 输出",
      ranks: allRanks(
        ranks,
        (r) => `输出位置 ${r * local}…${(r + 1) * local - 1}`,
        (r) => [buffer("Oᵣ", rows(result.output, r, local))],
      ),
    });
    return {
      ...base,
      title: "CP · 上下文并行",
      subtitle: "Q 分片留在原地，KV 走遍环，softmax 覆盖全局",
      split: "沿同一序列的位置 S 轴分散输入和激活；每个 Q 仍依赖全局有效 KV。",
      global: `Q[${size},d=1] · K[${size},1] · V[${size},2] → O[${size},2]`,
      derivation: [
        `S/${ranks} = ${size}/${ranks} = ${local}；每轮 KV 块[${local},1+2]`,
        `分数 sᵢⱼ=qᵢkⱼ/√1；m′=max(m,max(s))；l′=e^(m−m′)l+Σe^(s−m′)`,
        `o′=e^(m−m′)o+Σe^(s−m′)v；O=o/l；${causal ? "j≤i" : "所有 j"} 共享同一个分母`,
      ],
      stages,
      objects: Array.from({ length: ranks }, (_, r) => ({
        id: `cp-kv-${r}`,
        kind: "kv",
        points: stages.map((_, stage) =>
          point(
            (r + Math.max(0, Math.min(ranks - 1, stage - 1))) % ranks,
            ranks,
            `KV${r} 位置${r * local}…${(r + 1) * local - 1}`,
            170 + r * 30,
          ),
        ),
      })),
      output: result.output,
      attention: result,
      benefit: "长序列激活和 KV 可按位置分摊到多个 rank。",
      cost: "每个 Q 需访问远端 KV；环传输及稳定 softmax 合并增加通信与调度成本。",
      limitation:
        "本例为 Prefill：S 个 Q 按位置分片。Decode 的新 Q 长度为 1，不能直接照搬 S/P 的 Q 切分，但历史 KV 可采用其他分片方案。展示 CP 的完整注意力前向语义和连续块环；实际系统可用交错块或其他通信。Megatron 常用于长上下文训练，推理实现支持需单独确认。",
    };
  }
  const input = Array.from({ length: size }, (_, t) => [t + 1, 1]),
    result = sequenceParallel(input, a, b, ranks),
    local = size / ranks,
    f = size / ranks;
  const stages: ParallelStage[] = [
    {
      title: "SP 序列布局与本地 RMSNorm",
      explanation:
        "norm 对每个 token 的 H 轴独立计算。此例单位缩放、ε=0；输入无全零行。norm 输出仍按 S 切分。",
      communication: "本地逐 token RMSNorm",
      ranks: allRanks(
        ranks,
        (r) => `S 位置 ${r * local}…${(r + 1) * local - 1}`,
        (r) => [
          buffer("输入 Xᵣ", rows(input, r, local)),
          buffer("Norm(Xᵣ)", result.normShards[r]),
        ],
      ),
    },
    {
      title: "AllGather 完整序列",
      explanation:
        "每个 TP rank 需要所有 token 的归一化输入，沿 S 拼接，数值不求和。",
      communication: "AllGather · 沿 S 拼接",
      ranks: allRanks(
        ranks,
        (r) => `TP 通道 ${r * f}…${(r + 1) * f - 1}`,
        (r) => [
          buffer("完整 Norm(X)", result.gathered[r]),
          buffer("Aᵣ", result.tp.shards[r].a),
        ],
      ),
    },
    {
      title: "TP 列并行线性层",
      explanation:
        "序列 S 完整，F 通道分片；这是 TP 激活布局，不是各卡独立做本地注意力。",
      communication: "本地 X Aᵣ",
      ranks: allRanks(
        ranks,
        (r) => "完整 S · 分片 F",
        (r) => [buffer("Zᵣ", result.tp.shards[r].hidden)],
      ),
    },
    {
      title: "TP 行并行部分贡献",
      explanation: "每卡产出 Pᵣ[S,H]，仍须把 F 分片贡献相加。",
      communication: "本地 Zᵣ Bᵣ；下一步 ReduceScatter",
      ranks: allRanks(
        ranks,
        (r) => "完整 S · 部分和",
        (r) => [buffer("Pᵣ", result.tp.shards[r].partial)],
      ),
    },
    {
      title: "ReduceScatter 恢复 SP 布局",
      explanation:
        "先对 Pᵣ 按元素求和，再沿 S 把对应行留给 owner；无需把完整 Y 复制到每卡。后续 norm/dropout/残差可在本地序列分片上工作。",
      communication: "ReduceScatter(sum) · 求和后按 S 分片",
      ranks: allRanks(
        ranks,
        (r) => `S 位置 ${r * local}…${(r + 1) * local - 1}`,
        (r) => [buffer("Yᵣ", result.scattered[r])],
      ),
    },
  ];
  return {
    ...base,
    title: "SP · 序列并行（Megatron）",
    subtitle: "在序列布局和 TP 布局之间切换，节省部分激活",
    split:
      "norm/dropout 等逐 token 区域沿 S 分片；TP 线性层前恢复 S，层后归约并重新分片。",
    global: `X[S=${size},H=2] → Norm → TP(F=${size}) → Y[${size},2]`,
    derivation: [
      `SP: [S/${ranks},H] = [${local},2]；AllGather → [${size},2]`,
      `TP: [${size},2] × [2,${f}] → [${size},${f}]；再乘 [${f},2] → Pᵣ[${size},2]`,
      `ReduceScatter(ΣᵣPᵣ, S) → Yᵣ[${local},2]；逻辑输出仍为 Y[${size},2]`,
    ],
    stages,
    objects: Array.from({ length: ranks }, (_, r): ParallelObject => ({
      id: `sp-${r}`,
      kind: "activation" as const,
      points: [
        point(r, ranks, `Norm块${r} ${vectorLabel(result.normShards[r])}`, 140),
        point(r, ranks, `完整 S ${vectorLabel(result.normalized)}`, 165),
        point(
          r,
          ranks,
          `Z${r} ${vectorLabel(result.tp.shards[r].hidden)}`,
          160 + r * 25,
        ),
        point(
          r,
          ranks,
          `P${r} ${vectorLabel(result.tp.shards[r].partial)}`,
          160 + r * 25,
        ),
        {
          ...point(
            r,
            ranks,
            `Y块${r} ${vectorLabel(result.scattered[r])}`,
            270,
          ),
          x: rankX(r, ranks),
        },
      ],
    })).concat(
      Array.from({ length: ranks }, (_, from) =>
        Array.from({ length: ranks }, (_, to) => {
          const label = `S块${from} → rank${to} ${vectorLabel(result.normShards[from])}`;
          const y = 135 + (from * ranks + to) * 11;
          return {
            id: `sp-ag-${from}-${to}`,
            kind: "activation" as const,
            points: stages.map((_, stage) => ({
              ...point(stage === 0 ? from : to, ranks, label, y),
              visible: stage === 0,
            })),
          };
        }).filter((_, to) => from !== to),
      ).flat(),
      Array.from({ length: ranks }, (_, from) =>
        Array.from({ length: ranks }, (_, to) => {
          const contribution = rows(result.tp.shards[from].partial, to, local);
          const label = `P${from} 的 S块${to} ${vectorLabel(contribution)}`;
          const y = 135 + (from * ranks + to) * 11;
          return {
            id: `sp-rs-${from}-${to}`,
            kind: "partial" as const,
            points: stages.map((_, stage) => ({
              ...point(stage < 4 ? from : to, ranks, label, y),
              visible: stage === 3,
            })),
          };
        }).filter((_, to) => from !== to),
      ).flat(),
    ),
    output: result.output,
    benefit: "减少 norm/dropout 等区域原本在 TP rank 上复制的激活。",
    cost: "线性层边界需要 AllGather 与 ReduceScatter；通信是否隐藏取决于执行实现。",
    limitation:
      "这里专指 Megatron SP，要求 TP>1；它不是 CP，也不是让各卡只对本地序列独立做完整 attention。单请求 Decode 的新激活 S=1 无法直接等分给 2/4 卡；推理框架可能使用批次打包或其他布局。源文档以训练为主，不代表所有推理框架都开启 SP。",
  };
}
