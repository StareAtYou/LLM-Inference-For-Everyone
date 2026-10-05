import type { Topic } from "../types";

const checkedAt = "2026-10-05";
const parallelSource = {
  id: "megatron-parallelisms",
  title: "NVIDIA · Megatron 并行策略",
  checkedAt,
  url: "https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html",
};
const servingSource = {
  id: "vllm-parallelism",
  title: "vLLM · 推理中的并行与扩展",
  checkedAt,
  url: "https://docs.vllm.ai/en/latest/serving/parallelism_scaling/",
};
const collectiveSource = {
  id: "nccl-collectives",
  title: "NVIDIA NCCL · 集合通信定义",
  checkedAt,
  url: "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html",
};
const p2pSource = {
  id: "nccl-p2p",
  title: "NVIDIA NCCL · 点对点通信",
  checkedAt,
  url: "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html",
};

type Lesson = {
  id: string;
  title: string;
  terms: string[];
  before: string[];
  intuition: string;
  mathematics: string;
  engineering: string;
  detail: string;
  links: string[];
};
const parallelLessons: Lesson[] = [
  {
    id: "tp",
    title: "TP：把一次矩阵计算切到多卡",
    terms: ["TP", "Tensor Parallel", "张量并行"],
    before: ["ffn", "all-reduce"],
    intuition:
      "多张 GPU 各算同一层的一部分，再交换结果，让一次请求使用分散的权重。",
    mathematics:
      "列切分 W=[W₀ … Wₚ₋₁] 时，X[N,H]Wᵣ[H,F/P] 得到 Yᵣ[N,F/P]。下一个线性层沿输入维切分，各卡得到 Zᵣ[N,H]，求和恢复 Z。",
    engineering:
      "列并行后的激活可直接交给行并行层，无需每层都拼接。行并行可用 AllReduce 复制输出，也可用 ReduceScatter 交给 SP。维度、KV 头归属和内核通信方案由实现决定。",
    detail: "求和合并的是部分点积，拼接合并的是不同输出通道；二者不能互换。",
    links: ["all-reduce", "reduce-scatter", "sp"],
  },
  {
    id: "dp",
    title: "DP：让多个模型副本处理不同请求",
    terms: ["DP", "Data Parallel", "数据并行"],
    before: ["batching"],
    intuition: "每组 GPU 保存一份模型，负载均衡器把不同请求交给不同副本。",
    mathematics:
      "全局 B 个请求按请求轴分成 Bᵣ；各副本输入 [Bᵣ,T,H]，独立输出 [Bᵣ,T,H]。不等长请求常以 packed Token 表示，Nᵣ=ΣTᵢ。权重副本不随 DP 数缩小。",
    engineering:
      "推理副本没有梯度同步；训练 DP 的梯度 AllReduce 是另一种场景。MoE 的 attention DP 可与 EP 共享设备，此时专家层仍有跨副本的 Token 派发，不能把这种模式当成完全无通信的独立副本。",
    detail: "DP 主要扩展吞吐与并发，不会让独立副本中的单请求天然按副本数加速。",
    links: ["ep", "batching"],
  },
  {
    id: "ep",
    title: "EP：把 Token 送到它选中的专家",
    terms: ["EP", "Expert Parallel", "专家并行"],
    before: ["moe", "all-to-all"],
    intuition:
      "专家分布在不同 GPU。路由器按 Token 的选择，把向量送到对应专家，计算后按原 Token 聚合。",
    mathematics:
      "X[N,H] 经 Top-k 路由产生 N·k 个派发项。专家 e 收到 Xₑ[Nₑ,H]，ΣNₑ=N·k；局部 FFN 输出 [Nₑ,H]，返回后 yᵢ=ΣₑgᵢₑEₑ(xᵢ)。",
    engineering:
      "Nₑ 随路由与负载变化，并非 N/P。专家容量、padding、共享专家和门控依实现不同。EP 组可能复用 attention TP/DP 的 rank，不能把 TP·DP·EP 直接当成所需卡数。",
    detail: "反向聚合需要 Token ID 与路由权重，不能只按接收顺序把向量相加。",
    links: ["all-to-all", "moe", "dp"],
  },
  {
    id: "pp",
    title: "PP：把不同网络层放到不同 GPU",
    terms: ["PP", "Pipeline Parallel", "流水并行"],
    before: ["ffn", "send-recv"],
    intuition: "请求依次通过各 GPU 上的层段；前一段把隐藏状态交给下一段。",
    mathematics:
      "L 层按层轴分配给 P 个 stage。每段输入/输出通常都是 X[N,H]，层数与本地权重变化，隐藏宽度不因为按层切分变成 H/P。",
    engineering:
      "边界常用 Send/Recv 传递激活。微批可以错峰运行多段，但启动和排空产生流水空泡；Decode 的自回归依赖仍存在，单请求的下一 Token 不能任意提前执行。",
    detail:
      "本地权重可能不均衡；embedding、LM head、异构层成本都会影响层段划分。",
    links: ["send-recv", "decode", "tp"],
  },
  {
    id: "cp",
    title: "CP：切分上下文，保持完整注意力",
    terms: ["CP", "Context Parallel", "上下文并行"],
    before: ["attention", "all-gather"],
    intuition:
      "各 GPU 负责一部分序列位置，但每个 Query 仍要看到所有允许访问的 Key 和 Value。",
    mathematics:
      "逻辑输入 [B,T,H] 按 T 切成 [B,T/P,H]。本地 Q 为 [B,nQ,T/P,d]；注意力需要跨分片的 K/V，逻辑分数为 [B,nQ,T/P,T]，归一化覆盖完整可见 Key 轴。",
    engineering:
      "可通过 KV 交换、ring 或其他方案计算注意力；分块局部结果必须按全局 softmax 统计量合并。示例展示逻辑计算，不代表某个服务框架支持所有 CP 变体。",
    detail:
      "在 Decode 中新 Query 常只有一位置；分片的是历史上下文，不应继续把每张卡的 Query 写成 T/P。",
    links: ["attention", "flash-attention", "sp"],
  },
  {
    id: "sp",
    title: "SP：在 TP 之间切分逐 Token 激活",
    terms: ["SP", "Sequence Parallel", "序列并行", "Megatron SP"],
    before: ["tp", "all-gather", "reduce-scatter"],
    intuition:
      "这里采用 Megatron 的 SP：LayerNorm 等逐 Token 运算处理本地序列片段，进入 TP 线性层前再恢复需要的布局。",
    mathematics:
      "SP 区域持有 [N/P,H]；AllGather 沿 Token 轴恢复 [N,H] 后进入列 TP，得到 [N,F/P]。行 TP 产生部分结果 [N,H]，ReduceScatter 求和并切回 [N/P,H]。",
    engineering:
      "SP 是 TP 布局转换的一部分，不是再独立乘一个设备维度，也不等于 CP 的跨上下文注意力。具体推理服务是否采用该方案取决于内核、batch 和框架实现。",
    detail:
      "AllGather 不求和；ReduceScatter 既合并部分点积，又改变本地 Token 归属。",
    links: ["all-gather", "reduce-scatter", "cp"],
  },
];
const collectiveLessons: Lesson[] = [
  {
    id: "all-reduce",
    title: "AllReduce：所有设备得到同一份归约结果",
    terms: ["AllReduce", "All Reduce", "全归约"],
    before: [],
    intuition:
      "每张卡贡献一组数字，把相同位置的数字相加，再把完整结果交给每张卡。",
    mathematics:
      "P 个 rank 输入各为 xᵣ[N]；sum 归约输出 y[N]，yⱼ=Σᵣxᵣⱼ，每个 rank 都持有 y。输入和每卡输出的元素数不变。",
    engineering:
      "常用于 TP 行切分的部分点积合并；训练也用于梯度同步。算子定义不固定 ring、tree 等传输算法，浮点求和次序可能影响最低有效位。",
    detail:
      "ReduceScatter 后再 AllGather，在相同归约算子与分片布局下恢复 AllReduce 的结果。",
    links: ["tp", "reduce-scatter", "all-gather"],
  },
  {
    id: "all-gather",
    title: "AllGather：把分片拼接到每张卡",
    terms: ["AllGather", "All Gather", "全收集"],
    before: [],
    intuition:
      "每张卡拿出自己的一段，按 rank 顺序拼成完整数据；所有卡都收到同一份拼接结果。",
    mathematics:
      "每 rank 输入 xᵣ[n]；每 rank 输出 y[P·n]=concat(x₀,…,xₚ₋₁)。相同位置不会相加；shape 的收集轴长度增加 P 倍。",
    engineering:
      "用在 SP 到 TP 的激活恢复、某些 KV 交换和参数收集。等长 NCCL AllGather 的 count/dtype 必须一致；变长数据需要 padding 或其他专门协议。",
    detail: "收集不是求和。拼接后仍能辨认每个输入片段的原始 rank。",
    links: ["sp", "cp", "gather"],
  },
  {
    id: "reduce-scatter",
    title: "ReduceScatter：先归约，再分片归属",
    terms: ["ReduceScatter", "Reduce Scatter", "归约散发"],
    before: ["all-reduce"],
    intuition: "先把各 GPU 的相同位置相加，然后每张卡只保留结果中的一段。",
    mathematics:
      "每 rank 输入 [P·n]；先归约得到 [P·n]，rank r 输出 [n]，对应完整结果 [r·n:(r+1)·n]。归约减少副本，散发分配归属。",
    engineering:
      "常用于 TP→SP 的部分点积合并与激活切分；训练中的梯度分片也是用途之一。切分要求受 count/布局约束，不能丢掉尾部元素来凑整。",
    detail:
      "若再做 AllGather，所有卡可恢复完整归约结果；两阶段必须使用相同 rank 顺序。",
    links: ["sp", "all-reduce", "all-gather"],
  },
  {
    id: "all-to-all",
    title: "AllToAll：按目的地交换不同的数据块",
    terms: ["AllToAll", "All To All", "全交换"],
    before: [],
    intuition:
      "每张卡给每个目的地准备不同一段；目的地最终收到来自所有源卡的对应段。",
    mathematics:
      "源 rank i 输入 [P,n]，块 xᵢⱼ[n] 发给目的 rank j。rank j 输出 concat(x₀ⱼ,…,xₚ₋₁ⱼ)，仍为 [P·n]，但数据归属改变。",
    engineering:
      "常见于 EP 的 Token 派发和回传。逻辑全交换可由不同内核/协议实现；变长专家负载不能用等长示例推断真实通信量和耗时。",
    detail: "AllGather 复制所有输入给所有卡；AllToAll 只按目的地分配各块。",
    links: ["ep", "all-gather"],
  },
  {
    id: "broadcast",
    title: "Broadcast：复制 root 的一份数据",
    terms: ["Broadcast", "广播"],
    before: [],
    intuition: "指定 root 提供数据，把同一份数据复制给组内每张卡。",
    mathematics:
      "root 输入 x[N]，所有 rank 输出 x[N]。其他 rank 的旧输入不参与求和，也不会追加到输出。",
    engineering:
      "root 是通信组中的 rank，不一定等于物理 GPU 编号。广播可分发配置或公共数据，但也可与 Reduce 组合表达 AllReduce 的语义。",
    detail: "root 保留原始数据，接收卡得到同样内容；复制不改变元素值。",
    links: ["reduce", "all-reduce"],
  },
  {
    id: "reduce",
    title: "Reduce：仅 root 收到归约结果",
    terms: ["Reduce", "归约到根"],
    before: ["all-reduce"],
    intuition: "所有卡都贡献数据，但只有指定 root 收到相加后的完整结果。",
    mathematics:
      "各 rank 输入 xᵣ[N]；root 输出 Σᵣxᵣ[N]。非 root 没有这个算子的有效输出；不要把它描述成全零结果。",
    engineering:
      "需要明确归约算子、root 和通信组；如果每张卡都要最终结果，可以再 Broadcast，或使用 AllReduce。",
    detail: "参与输入的卡与持有输出的卡是两回事。",
    links: ["broadcast", "all-reduce"],
  },
  {
    id: "gather",
    title: "Gather：仅 root 拼接所有分片",
    terms: ["Gather", "收集到根"],
    before: ["all-gather"],
    intuition:
      "把各卡的一段数据按 rank 顺序收集到 root；其他卡没有完整拼接输出。",
    mathematics:
      "每 rank 输入 [n]；root 输出 [P·n]；非 root 输出未定义。不同 rank 的片段并列排列，不逐元素归约。",
    engineering:
      "root 需要足够大的接收缓冲区。逻辑 Gather 可由集合操作或成组 Send/Recv 实现；是否使用某个 API 由 NCCL 版本和服务实现决定。",
    detail: "与 AllGather 的区别在结果放在哪里，而不是拼接顺序。",
    links: ["all-gather", "scatter", "reduce"],
  },
  {
    id: "scatter",
    title: "Scatter：root 把不同分片交给不同卡",
    terms: ["Scatter", "散发"],
    before: ["gather"],
    intuition:
      "root 把完整数据切成有顺序的多段，每张卡只收到归属于自己的那一段。",
    mathematics:
      "root 输入 [P·n]；rank r 输出 [n]，取 [r·n:(r+1)·n]。不求和，也不把完整输入复制给每张卡。",
    engineering:
      "需保证分片顺序和接收大小一致；示例使用等长块，变长或不规则布局需要额外元数据。与 Broadcast 的完整复制不同。",
    detail: "同一 rank 顺序下 Gather 和 Scatter 可以还原分片布局。",
    links: ["gather", "broadcast"],
  },
  {
    id: "send-recv",
    title: "Send / Recv：在一对设备间交接数据",
    terms: ["Send", "Recv", "Send Recv", "Point to Point", "点对点"],
    before: [],
    intuition:
      "指定发送方和接收方，只有这对设备交换数据，其他卡不接收这份输出。",
    mathematics:
      "发送端提供 x[N]，接收端得到同 shape 的 x[N]；没有求和或拼接，不要求整个通信组都产生输出。",
    engineering:
      "用于 PP 激活交接、环形 KV 交换等。成对通信必须匹配 count、datatype、peer 与执行顺序；并发匹配的 Send/Recv 常需成组提交以避免等待死锁。",
    detail: "Send/Recv 描述数据交接，不代表模型计算阶段会自动并行或消除依赖。",
    links: ["pp", "cp"],
  },
];
function makeTopics(
  lessons: Lesson[],
  kind: "parallel" | "collectives",
): Topic[] {
  return lessons.map((x) => ({
    id: x.id,
    title: x.title,
    englishTerms: x.terms,
    area: kind === "parallel" ? "多 GPU 并行" : "通信原语",
    prerequisites: x.before,
    levels: {
      beginner: {
        summary: x.intuition,
        explanation: x.intuition,
        details: [
          x.detail,
          "先看每张 GPU 的输入与最终输出，再播放数据交接过程。动画中的数字是可核对的教学示例。",
        ],
      },
      advanced: {
        summary: x.mathematics,
        explanation: x.mathematics,
        details: [
          x.detail,
          "先确认轴的含义，再区分复制、拼接、求和和分片。shape 描述逻辑数组，不等于显存中实际内核布局。",
        ],
      },
      expert: {
        summary: x.engineering,
        explanation: x.engineering,
        details: [
          x.detail,
          "通信组中的 rank 编号只在当前组内有效。教学时间轴没有建模网络带宽、同步开销和硬件性能，不用于预测真实吞吐。",
        ],
      },
    },
    links: [
      {
        label: "打开独立工作台",
        to: `/distributed?view=${kind}&${kind === "parallel" ? "parallel" : "collective"}=${x.id}`,
      },
      ...x.links.map((id) => ({
        label: `关联：${[...parallelLessons, ...collectiveLessons].find((t) => t.id === id)?.title ?? id}`,
        to: `/learn/${id}`,
      })),
    ],
    sources:
      kind === "parallel"
        ? [parallelSource, servingSource]
        : [x.id === "send-recv" ? p2pSource : collectiveSource],
  }));
}
export const distributedTopics: Topic[] = [
  ...makeTopics(parallelLessons, "parallel"),
  ...makeTopics(collectiveLessons, "collectives"),
];
export const parallelTopicIds = parallelLessons.map((x) => x.id);
export const collectiveTopicIds = collectiveLessons.map((x) => x.id);
