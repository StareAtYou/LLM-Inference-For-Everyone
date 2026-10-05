import { causalAttention } from "../simulation/attention";
import { buildDeltaTrace } from "../simulation/gatedDelta";
import { quantizeSymmetric } from "../simulation/quantization";
import { filterDistribution, softmax } from "../simulation/sampling";

export type MechanismParameter = {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  initial: number;
  meaning: string;
  choices?: number[];
};
export type MechanismDefinition = {
  id: string;
  title: string;
  subtitle: string;
  category: string;
  view: string;
  source?: { title: string; url: string };
  parameters: MechanismParameter[];
};
export type MechanismPanel = {
  label: string;
  kind:
    | "vector"
    | "matrix"
    | "tokens"
    | "bars"
    | "cache"
    | "queue"
    | "experts"
    | "graph"
    | "rotation"
    | "plot";
  values?: number[];
  matrix?: number[][];
  items?: string[];
  selected?: number[];
  columns?: string[];
  rows?: string[];
  note?: string;
  mask?: boolean;
};
export type MechanismFrame = {
  stage: string;
  caption: string;
  formula: string;
  panels: MechanismPanel[];
  readouts: Record<string, number>;
  output?: number[];
};
const param = (
  key: string,
  label: string,
  min: number,
  max: number,
  step: number,
  initial: number,
  meaning: string,
): MechanismParameter => ({ key, label, min, max, step, initial, meaning });
const amplitude = param(
  "amplitude",
  "输入幅度",
  0,
  2,
  0.25,
  1,
  "等比例缩放教学输入，0 用于观察边界。",
);
const temperature = param(
  "temperature",
  "温度 T",
  0,
  2,
  0.25,
  1,
  "T 越低分布越尖锐；0 演示确定性 argmax。",
);
const entries: [
  string,
  string,
  string,
  string,
  string,
  MechanismParameter[]?,
][] = [
  [
    "tokenization",
    "Tokenization · 文本变成编号",
    "字节片段如何合并为词表中的 Token",
    "基础",
    "tokens",
    [
      param(
        "merges",
        "合并轮数",
        0,
        3,
        1,
        3,
        "固定小词表的 BPE 合并次数；不代表真实 Qwen 分词器。",
      ),
    ],
  ],
  [
    "embedding",
    "Embedding · 编号查向量",
    "按 Token ID 读取词表矩阵的一行",
    "基础",
    "lookup",
  ],
  [
    "rmsnorm",
    "RMSNorm · 控制向量尺度",
    "平方 → 均值 → 开根号 → 逐通道缩放",
    "网络",
    "norm",
    [amplitude],
  ],
  [
    "qkv",
    "Q / K / V · 三次投影",
    "同一个输入，与三个独立权重矩阵相乘",
    "网络",
    "projection",
  ],
  [
    "rope",
    "RoPE · 用旋转编码位置",
    "一对通道随位置旋转，长度保持不变",
    "网络",
    "rotation",
    [
      param(
        "position",
        "位置 p",
        0,
        8,
        1,
        2,
        "示意频率 θ=π/8；真实网络同时使用多组频率。",
      ),
    ],
  ],
  [
    "attention",
    "Attention · 找到相关上下文",
    "因果遮罩、行内 Softmax 与加权求和",
    "网络",
    "attention",
    [
      param(
        "row",
        "查询位置",
        0,
        2,
        1,
        2,
        "选择哪个位置的 Q 去读取过去的 K / V。",
      ),
    ],
  ],
  [
    "softmax",
    "Softmax · 分数变成概率",
    "减去最大值，指数化，再归一化",
    "基础",
    "distribution",
    [temperature],
  ],
  [
    "attention-output",
    "输出投影 · 合并注意力头",
    "拼接各头的结果，再映射回残差宽度",
    "网络",
    "concat",
  ],
  [
    "residual",
    "Residual · 保留输入通路",
    "模块输出与原输入逐元素相加",
    "网络",
    "addition",
    [
      param(
        "branch",
        "分支系数",
        0,
        2,
        0.25,
        1,
        "缩放模块分支，观察直连输入仍然保留。",
      ),
    ],
  ],
  [
    "ffn",
    "Dense FFN · 逐位置非线性",
    "扩维、SiLU 门控、逐元素乘，再降维",
    "网络",
    "gate",
    [amplitude],
  ],
  [
    "gqa",
    "GQA · 多个 Q 共享 K / V",
    "4 个查询头按组共享，缓存按 KV 头计",
    "网络",
    "groups",
    [
      param(
        "kvHeads",
        "KV 头数",
        1,
        4,
        1,
        2,
        "用 4 个 Q 头展示 MQA / GQA / MHA；KV 头数须整除 Q 头数。",
      ),
    ],
  ],
  [
    "gated-deltanet",
    "Gated DeltaNet · 更新递归状态",
    "局部卷积与门控 delta 更新，状态不按上下文增长",
    "状态",
    "state",
    [
      param(
        "alpha",
        "保留门 α",
        0,
        1,
        0.1,
        0.8,
        "先让旧状态衰减；β=0.7 控制预测误差的写入。",
      ),
    ],
  ],
  [
    "moe",
    "MoE · 路由到少量专家",
    "8 个教学专家中选 2 个，加上共享分支",
    "网络",
    "experts",
    [
      param(
        "bias",
        "路由偏置",
        -2,
        2,
        0.5,
        0,
        "改变 E0 的 logit，观察 top-2 选择和归一化权重。",
      ),
    ],
  ],
  [
    "prefill",
    "Prefill · 一起处理输入",
    "完整输入并行经过网络，建立缓存",
    "状态",
    "prefill",
    [
      param(
        "tokens",
        "输入 Token 数",
        2,
        6,
        1,
        4,
        "显示逻辑序列长度；不表示真实 GPU 并行时间。",
      ),
    ],
  ],
  [
    "decode",
    "Decode · 一轮只推进一个位置",
    "读取历史缓存 → 处理新 Token → 预测下一个",
    "状态",
    "decode",
  ],
  [
    "sampling",
    "Sampling · 从概率选 Token",
    "温度 → top-k → 累积概率区间 → 固定随机数",
    "基础",
    "sample",
    [
      temperature,
      param(
        "topK",
        "top-k",
        1,
        4,
        1,
        3,
        "保留概率最高的 k 个候选，随后重新归一化。",
      ),
    ],
  ],
  [
    "lm-head",
    "LM Head · 隐藏向量到词表",
    "与每个词表列相乘，得到下一 Token 的 logits",
    "网络",
    "vocabulary",
  ],
  [
    "kv-cache",
    "KV Cache · 历史只算一次",
    "按层、请求保留已处理位置的 K / V",
    "状态",
    "cache",
  ],
  [
    "paged-attention",
    "Paged Attention · 逻辑页到物理块",
    "序列连续，物理存储可不连续，按块分配",
    "优化",
    "pages",
    [
      param(
        "blockSize",
        "每块 Token 容量",
        1,
        4,
        1,
        2,
        "两个请求共 7 个 Token，改变块大小观察尾块空位。",
      ),
    ],
  ],
  [
    "prefix-caching",
    "Prefix Cache · 相同前缀复用",
    "完全一致的已计算前缀块命中，私有后缀继续算",
    "优化",
    "prefix",
    [
      param(
        "shared",
        "共同前缀长度",
        0,
        4,
        1,
        3,
        "这里按 Token 示意；实际引擎通常以完整块为粒度。",
      ),
    ],
  ],
  [
    "batching",
    "Continuous Batching · 槽位滚动补位",
    "等待、执行、完成、释放与新请求入场",
    "工程",
    "scheduler",
    [
      param(
        "slots",
        "并发槽位",
        1,
        2,
        1,
        2,
        "3 个请求的生成长度分别为 1、2、3；按离散轮次演示。",
      ),
    ],
  ],
  [
    "quantization",
    "Quantization · 浮点压成整数",
    "计算 scale、舍入量化码、重建与误差",
    "优化",
    "quant",
    [
      param(
        "bits",
        "整数位宽",
        4,
        8,
        4,
        4,
        "对称量化，使用 ±(2^(b−1)−1)，不计比例元数据。",
      ),
      amplitude,
    ],
  ],
  [
    "speculation",
    "Speculation · 草稿一次验证多个",
    "主模型并行验证，拒绝处修正，后续草稿丢弃",
    "优化",
    "verify",
    [
      param(
        "accepted",
        "接受草稿数",
        0,
        3,
        1,
        2,
        "固定判定示例；真实采样需要分布校正才能保持目标分布。",
      ),
    ],
  ],
  [
    "flash-attention",
    "Flash Attention · 分块精确归约",
    "在线更新最大值、分母和加权 V，无需保存完整矩阵",
    "优化",
    "tiles",
  ],
  [
    "parallelism",
    "Parallelism · 计算怎样分到设备",
    "张量并行的局部乘积 → 通信 → 同步结果",
    "工程",
    "devices",
    [
      param(
        "devices",
        "教学设备数",
        1,
        2,
        1,
        2,
        "按输入通道分片的 row-parallel 线性层；模拟不计通信时间。",
      ),
    ],
  ],
  [
    "chunked-prefill",
    "Chunked Prefill · 输入分段调度",
    "每轮输入块与 decode 共享 token 预算",
    "工程",
    "chunks",
    [
      param(
        "chunk",
        "Prefill 块长",
        1,
        3,
        1,
        2,
        "6 个输入 Token 分段处理；一个 decode 请求每轮占 1 Token。",
      ),
    ],
  ],
  [
    "cuda-graphs",
    "CUDA Graphs · 复用启动图",
    "先捕获固定形状的执行拓扑，再重放同一图",
    "工程",
    "capture",
  ],
  [
    "metrics",
    "Metrics · 分开看等待与生成",
    "首 Token 延迟、逐 Token 间隔与吞吐口径",
    "工程",
    "metrics",
    [
      param(
        "gap",
        "教学步间隔",
        1,
        4,
        1,
        2,
        "单位是模拟步骤，不是毫秒，也不是 GPU 实测。",
      ),
    ],
  ],
  [
    "frameworks",
    "Serving Stack · 层与层的边界",
    "接口 → 调度 → 执行 → kernel → 流式返回",
    "工程",
    "stack",
  ],
];
const coreSources: Record<string, { title: string; url: string }> = {
  rmsnorm: {
    title: "RMSNorm 原始论文",
    url: "https://arxiv.org/abs/1910.07467",
  },
  qkv: {
    title: "Transformer 原始论文",
    url: "https://arxiv.org/abs/1706.03762",
  },
  residual: {
    title: "Transformer 原始论文",
    url: "https://arxiv.org/abs/1706.03762",
  },
  "lm-head": {
    title: "Transformer 原始论文",
    url: "https://arxiv.org/abs/1706.03762",
  },
  softmax: {
    title: "Transformer 原始论文",
    url: "https://arxiv.org/abs/1706.03762",
  },
  "attention-output": {
    title: "Transformer 原始论文",
    url: "https://arxiv.org/abs/1706.03762",
  },
  "flash-attention": {
    title: "FlashAttention 原始论文",
    url: "https://arxiv.org/abs/2205.14135",
  },
  "cuda-graphs": {
    title: "NVIDIA CUDA Graphs 文档",
    url: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html",
  },
};
export const mechanisms: MechanismDefinition[] = entries.map(
  ([id, title, subtitle, category, view, parameters]) => ({
    id,
    title,
    subtitle,
    category,
    view,
    source: coreSources[id],
    parameters: (parameters ?? []).map((p) =>
      p.key === "kvHeads" ? { ...p, choices: [1, 2, 4] } : p,
    ),
  }),
);
const vector = (
  label: string,
  values: number[],
  note?: string,
  selected?: number[],
): MechanismPanel => ({ label, kind: "vector", values, note, selected });
const matrix = (
  label: string,
  values: number[][],
  note?: string,
  selected?: number[],
): MechanismPanel => ({
  label,
  kind: "matrix",
  matrix: values,
  note,
  selected,
});
const items = (
  label: string,
  values: string[],
  kind: MechanismPanel["kind"] = "tokens",
  selected?: number[],
  note?: string,
): MechanismPanel => ({ label, kind, items: values, selected, note });
const bars = (
  label: string,
  values: number[],
  columns: string[],
  selected?: number[],
): MechanismPanel => ({ label, kind: "bars", values, columns, selected });
const frame = (
  stage: string,
  caption: string,
  formula: string,
  panels: MechanismPanel[],
  readouts: Record<string, number> = {},
  output?: number[],
): MechanismFrame => ({ stage, caption, formula, panels, readouts, output });
const multiply = (x: number[], w: number[][]) =>
  w[0].map((_, j) => x.reduce((sum, n, i) => sum + n * w[i][j], 0));
const X = [1, -1, 2, -2];
const smallW = [
  [1, 0],
  [0, 1],
];
const tokenNames = ["猫", "坐", "垫", "上"];
const logits = [1, -1, 1.5, 0.5];

export function buildMechanismFrames(
  id: string,
  params: Record<string, number> = {},
): MechanismFrame[] {
  const definition = mechanisms.find((m) => m.id === id);
  if (!definition) return [];
  const p: Record<string, number> = {};
  for (const parameter of definition.parameters) {
    const value = params[parameter.key] ?? parameter.initial;
    if (
      !Number.isFinite(value) ||
      value < parameter.min ||
      value > parameter.max ||
      (parameter.choices !== undefined && !parameter.choices.includes(value)) ||
      Math.abs(
        (value - parameter.min) / parameter.step -
          Math.round((value - parameter.min) / parameter.step),
      ) > 1e-8
    )
      throw Error("原理参数超出有限教学范围");
    p[parameter.key] = value;
  }
  const scaled = X.map((x) => x * (p.amplitude ?? 1) || 0);
  switch (id) {
    case "tokenization": {
      const merges = Math.round(p.merges);
      const sequences = [
        ["l", "o", "w", "e", "r"],
        ["lo", "w", "e", "r"],
        ["low", "e", "r"],
        ["low", "er"],
      ];
      const vocab: Record<string, number> = {
        l: 0,
        o: 1,
        w: 2,
        e: 3,
        r: 4,
        lo: 5,
        low: 6,
        er: 7,
      };
      return [
        frame(
          "读取字节片段",
          "教学英文 lower 从字符片段开始。这里固定词表和优先合并规则，不模拟真实 tokenizer。",
          '输入 = "lower"',
          [
            items("初始片段", sequences[0]),
            items(
              "合并规则",
              ["l + o → lo", "lo + w → low", "e + r → er"],
              "graph",
            ),
          ],
        ),
        ...Array.from({ length: 3 }, (_, i) =>
          frame(
            `合并候选 ${i + 1}`,
            i < merges
              ? `应用第 ${i + 1} 条词表合并，把相邻片段合成一个 Token；序列长度随之减少。`
              : `当前只允许 ${merges} 轮合并；这一候选保持为独立片段，已经形成的 Token 不再改变。`,
            "相邻片段 → 词表合并",
            [
              items("当前分词", sequences[Math.min(i + 1, merges)], "tokens", [
                Math.max(
                  0,
                  Math.min(i, sequences[Math.min(i + 1, merges)].length - 1),
                ),
              ]),
            ],
          ),
        ),
        frame(
          "写出 Token ID",
          "每个最终片段查到固定词表编号。后续网络接受这些整数编号，而不是直接计算文字。",
          "Token → vocabulary ID",
          [
            items("最终片段", sequences[merges]),
            vector(
              "Token ID",
              sequences[merges].map((s) => vocab[s]),
            ),
          ],
          { tokens: sequences[merges].length },
          sequences[merges].map((s) => vocab[s]),
        ),
      ];
    }
    case "embedding": {
      const table = [
        [0.2, -0.1, 0.4],
        [0.7, 0.5, -0.2],
        [-0.3, 0.9, 0.1],
        [0.4, -0.2, 0.8],
      ];
      return [
        frame(
          "输入编号",
          "当前输入为 Token ID 2。教学词表只有 4 行，每行代表一个 Token 的 3 维向量。",
          "id = 2",
          [items("词表", tokenNames), vector("输入 ID", [2])],
        ),
        frame(
          "定位词表行",
          "通过整数索引读取第 2 行；Embedding 是查表，这一步不需要与所有词逐一比较。",
          "e = E[id, :]",
          [
            matrix(
              "Embedding table · 4 × 3",
              table,
              "高亮 ID 2 所在的整行",
              [6, 7, 8],
            ),
          ],
        ),
        frame(
          "送入网络",
          "读取出来的向量保留 3 个通道。不同位置可查到相同 Token 向量，位置编码由后续运算处理。",
          "E[2] = [−0.3, 0.9, 0.1]",
          [
            vector("隐藏向量", table[2]),
            items("下一站", ["归一化", "Q / K / V 投影"], "graph"),
          ],
          {},
          table[2],
        ),
      ];
    }
    case "rmsnorm": {
      const squares = scaled.map((x) => x * x),
        mean = squares.reduce((a, b) => a + b, 0) / 4,
        rms = Math.sqrt(mean + 1e-6),
        out = scaled.map((x) => x / rms);
      return [
        frame(
          "保留通道输入",
          "教学向量含正负通道。RMSNorm 不先减去均值，而是用均方根控制整体幅度。",
          "x = [1, −1, 2, −2] × amplitude",
          [vector("输入 x", scaled)],
        ),
        frame(
          "逐通道平方",
          "每个通道平方后都非负。将这些数相加再除以通道数 4，得到均方值。",
          "mean(x²) = Σ xᵢ² / d",
          [vector("平方 x²", squares), vector("均方", [mean])],
          { meanSquare: mean },
        ),
        frame(
          "开根号与稳定项",
          "均方值加上 ε=0.000001 后开根号。全零输入仍有非零分母，避免除零。",
          "r = √(mean(x²) + ε)",
          [vector("稳定分母 r", [rms])],
          { rms },
        ),
        frame(
          "归一化并缩放",
          "逐通道除以同一个分母，再乘可学习缩放 γ。教学 γ 全为 1，向量方向保持不变。",
          "yᵢ = xᵢ / r × γᵢ",
          [
            vector("原始输入", scaled),
            vector("γ", [1, 1, 1, 1]),
            vector("输出 y", out),
          ],
          { rms },
          out,
        ),
      ];
    }
    case "qkv": {
      const x = [1, 2],
        wq = [
          [0, 2],
          [1, -1],
        ],
        wk = [
          [1, 1],
          [0, -1],
        ],
        wv = [
          [1, -1],
          [0, 1],
        ];
      const q = multiply(x, wq),
        k = multiply(x, wk),
        v = multiply(x, wv);
      return [
        frame(
          "一个输入三份权重",
          "隐藏向量 x=[1,2] 同时送往三个独立的线性投影，各自产生查询、键和值。",
          "Q = xWQ · K = xWK · V = xWV",
          [
            vector("输入 x", x),
            matrix("WQ", wq),
            matrix("WK", wk),
            matrix("WV", wv),
          ],
        ),
        frame(
          "展开 Q 的贡献",
          "计算 Q 第一个通道：输入第一项贡献 0，第二项贡献 2；相加得到 Q₀=2。可点选矩阵单元查看。",
          "Q₀ = 1×0 + 2×1 = 2",
          [
            vector("逐项贡献", [0, 2], undefined, [1]),
            matrix("WQ", wq, "高亮第一列", [0, 2]),
            vector("Q", q),
          ],
          { contribution: 2 },
        ),
        frame(
          "独立算 K 和 V",
          "相同 x 乘不同权重得到不同向量。K 用于匹配查询，V 用于向输出贡献内容。",
          "K = [1, −1] · V = [1, 1]",
          [vector("Q", q), vector("K", k), vector("V", v)],
        ),
        frame(
          "交付注意力算子",
          "三个投影结果保持各自语义。Q / K 接下来可应用 RoPE，V 保持内容通道并写入缓存。",
          "[Q, K, V] → position / attention / cache",
          [
            items(
              "三条数据路径",
              [
                "Q → RoPE → 当前查询",
                "K → RoPE → 历史匹配",
                "V → Cache → 加权内容",
              ],
              "graph",
            ),
            vector("拼接读数 Q | K | V", [...q, ...k, ...v]),
          ],
          {},
          [...q, ...k, ...v],
        ),
      ];
    }
    case "rope": {
      const angle = (p.position * Math.PI) / 8,
        out = [Math.cos(angle), Math.sin(angle)];
      return [
        frame(
          "取一对通道",
          "把查询或键的两个通道视为平面坐标。本例初始向量为 (1,0)，其他通道对有各自频率。",
          "pair = (x₀, x₁) = (1, 0)",
          [
            { label: "通道平面", kind: "rotation", values: [1, 0, 0] },
            vector("原始通道对", [1, 0]),
          ],
        ),
        frame(
          "位置决定角度",
          "位置 p 乘上固定教学频率 π/8，形成这一通道对的旋转角；越后的位置旋转越多。",
          "φ = p · θ, θ = π/8",
          [
            vector("位置 p / 角度 φ", [p.position, angle]),
            matrix("旋转矩阵 R(φ)", [
              [Math.cos(angle), -Math.sin(angle)],
              [Math.sin(angle), Math.cos(angle)],
            ]),
          ],
          { angle },
        ),
        frame(
          "旋转 Q / K",
          "使用旋转矩阵得到新坐标。旋转保持长度；Q 和 K 的相对位置会通过内积产生影响。",
          "(x′₀, x′₁) = (cos φ, sin φ)",
          [
            {
              label: "旋转后的通道平面",
              kind: "rotation",
              values: [...out, angle],
            },
            vector("新通道对", out),
          ],
          { length: Math.hypot(...out) },
          out,
        ),
      ];
    }
    case "attention": {
      const q = [
          [1, 0],
          [0.5, 1],
          [0.2, 0.3],
        ],
        k = [
          [1, 0],
          [0.2, 1],
          [-0.4, 0.7],
        ];
      const scores = q.map((query) =>
        k.map(
          (key) =>
            query.reduce((sum, x, i) => sum + x * key[i], 0) / Math.sqrt(2),
        ),
      );
      const row = Math.round(p.row),
        weights = causalAttention(scores),
        v = [
          [1, 0],
          [0, 2],
          [1, 1],
        ],
        out = [0, 1].map((c) =>
          weights[row].reduce((sum, w, i) => sum + w * v[i][c], 0),
        );
      return [
        frame(
          "查询与历史键",
          "选中一个查询位置。每个 Q 与各位置 K 做缩放点积形成分数，未来位置暂时也列在矩阵中。",
          "scores = QKᵀ / √2",
          [
            matrix("Q · 教学向量", q),
            matrix("K · 教学向量", k),
            matrix("缩放点积分数", scores, "由左侧 Q 和 K 真实计算", [
              row * 3,
              row * 3 + 1,
              row * 3 + 2,
            ]),
          ],
        ),
        frame(
          "遮住未来位置",
          "每一行只能读到自身及过去。未来格子被遮罩，不参与 Softmax 分母，最终权重严格为零。",
          "j > i ⇒ masked",
          [
            {
              ...matrix("因果可见区域", scores),
              mask: true,
              selected: [row * 3, row * 3 + 1, row * 3 + 2],
            },
          ],
        ),
        frame(
          "行内概率归一化",
          "对可见分数减最大值、指数化，再按行归一化。每行可见权重之和为 1。",
          "Aᵢ = softmax(scoresᵢ + causal mask)",
          [
            matrix("因果注意力权重", weights, "未来位置 = 0", [
              row * 3,
              row * 3 + 1,
              row * 3 + 2,
            ]),
            bars("选中行概率", weights[row], ["位置 0", "位置 1", "位置 2"]),
          ],
          { weightSum: weights[row].reduce((a, b) => a + b, 0) },
        ),
        frame(
          "加权读取内容",
          "把这一行的权重分别乘各位置 V，再逐通道求和。权重分布直接决定输出融合了多少历史内容。",
          "Oᵢ = Σⱼ Aᵢⱼ Vⱼ",
          [
            matrix("因果注意力权重", weights, undefined, [
              row * 3,
              row * 3 + 1,
              row * 3 + 2,
            ]),
            matrix("V", v),
            vector("当前查询的输出", out),
          ],
          {},
          out,
        ),
      ];
    }
    case "softmax": {
      const max = Math.max(...logits),
        shifted = logits.map((v) => v - max),
        exp =
          p.temperature === 0
            ? softmax(logits, 0)
            : shifted.map((v) => Math.exp(v / p.temperature)),
        probs = softmax(logits, p.temperature);
      return [
        frame(
          "读取原始分数",
          "logits 可以为负，也不必和为 1。概率需要把相对分数变成非负且归一化的权重。",
          "z = [1, −1, 1.5, 0.5]",
          [bars("原始 logits", logits, tokenNames)],
        ),
        frame(
          "稳定平移与温度",
          p.temperature === 0
            ? "温度为零时直接使用 argmax，避免除以零；最高 logit 的候选获得概率 1。"
            : "先减去最大的 logit，保持概率结果不变，再除以温度；指数化时最大项为 1，数值更稳定。",
          p.temperature === 0 ? "T=0 ⇒ argmax(z)" : "eᵢ = exp((zᵢ − max(z))/T)",
          [vector("平移后分数", shifted), bars("非负权重 e", exp, tokenNames)],
          { temperature: p.temperature },
        ),
        frame(
          "除以权重总和",
          "所有非负权重除以相同分母，得到和为 1 的概率。它是注意力聚合或下一 Token 采样的输入。",
          "pᵢ = eᵢ / Σeⱼ",
          [bars("归一化概率", probs, tokenNames), vector("概率向量", probs)],
          { probabilitySum: probs.reduce((a, b) => a + b, 0) },
          probs,
        ),
      ];
    }
    case "attention-output": {
      const heads = [
          [1, 0],
          [0, 2],
        ],
        joined = heads.flat(),
        w = [
          [1, 1],
          [0, 1],
          [1, 0],
          [0, 1],
        ],
        out = multiply(joined, w);
      return [
        frame(
          "各头独立读上下文",
          "两个注意力头各自输出 2 个通道。它们可能聚合不同的上下文信息，此时仍然分开。",
          "head₀=[1,0], head₁=[0,2]",
          [vector("头 0 输出", heads[0]), vector("头 1 输出", heads[1])],
        ),
        frame(
          "沿通道拼接",
          "把两个头沿特征轴拼接为 4 维向量；序列位置不增加，改变的是每个位置的通道表示。",
          "concat(head₀, head₁) = [1,0,0,2]",
          [
            vector(
              "拼接后的 4 个通道",
              joined,
              "左两项来自头 0，右两项来自头 1",
            ),
          ],
        ),
        frame(
          "投影回残差宽度",
          "拼接向量乘 Wₒ，把 4 个通道混合为 2 个输出通道。这不是把多个头简单平均。",
          "O = concat(heads) Wₒ = [1,3]",
          [matrix("Wₒ · 4 × 2", w), vector("输出投影结果", out)],
          {},
          out,
        ),
      ];
    }
    case "residual": {
      const branch = [0.2, 0.5, -0.3, 0.1].map((v) => v * p.branch),
        out = X.map((v, i) => v + branch[i]);
      return [
        frame(
          "保留直连输入",
          "输入走两条路径：一条进入模块，一条保留原值。模块计算结束前，直连分支不做替换。",
          "identity(x) = x",
          [
            vector("直连路径 x", X),
            items("模块路径", ["归一化", "Attention 或 FFN"], "graph"),
          ],
        ),
        frame(
          "计算模块分支",
          "这里使用固定教学模块输出并按系数缩放。直连输入和模块输出必须具有相同的形状。",
          "branch = module(x) × coefficient",
          [vector("直连 x", X), vector("模块分支 Δ", branch)],
          { coefficient: p.branch },
        ),
        frame(
          "逐元素相加",
          "第 i 个输出为 xᵢ+Δᵢ。系数为零时输出等于原始输入；新增信息通过加法写入残差流。",
          "y = x + Δ",
          [
            vector("原输入 x", X),
            vector("变化 Δ", branch),
            vector("相加结果 y", out),
          ],
          {},
          out,
        ),
      ];
    }
    case "ffn": {
      const x = [scaled[0], scaled[1]],
        up = multiply(x, [
          [1, 0.5, -1, 0],
          [0, 1, 0.5, -1],
        ]),
        gate = multiply(x, [
          [0.5, 1, 0, -1],
          [1, 0, -1, 0.5],
        ]),
        silu = gate.map((v) => v / (1 + Math.exp(-v))),
        product = up.map((v, i) => v * silu[i]),
        down = [
          [1, 0],
          [0, 1],
          [0.5, 0.5],
          [-0.5, 0.5],
        ],
        out = multiply(product, down);
      return [
        frame(
          "逐位置输入",
          "每个位置独立进入 FFN，不在这一步读取其他 Token。本例从 2 个通道扩展为 4 个中间通道。",
          "x ∈ ℝ²",
          [vector("输入 x", x)],
        ),
        frame(
          "扩维与门分支",
          "同一输入分别乘 Wup 和 Wgate。一个提供内容，一个提供逐通道的门控信号。",
          "u=xWup · g=xWgate",
          [vector("扩维内容 u", up), vector("门输入 g", gate)],
        ),
        frame(
          "非线性门控",
          "对 gate 应用 SiLU，再与内容逐元素相乘。正负与幅度共同决定哪些通道贡献更多信息。",
          "h = SiLU(g) ⊙ u",
          [
            vector("SiLU(g)", silu),
            vector("内容 u", up),
            vector("门控中间值 h", product),
          ],
        ),
        frame(
          "投影回输入宽度",
          "中间 4 维向量乘 Wdown，恢复 2 维。模块结果随后可与同形状的残差输入相加。",
          "y = hWdown",
          [matrix("Wdown · 4 × 2", down), vector("FFN 输出", out)],
          {},
          out,
        ),
      ];
    }
    case "gqa": {
      const kv = p.kvHeads,
        mapping = Array.from({ length: 4 }, (_, i) => Math.floor((i * kv) / 4));
      return [
        frame(
          "查询头保持独立",
          "4 个 Q 头拥有自己的查询；选择 KV 头数会改变共享分组，但不会减少查询头数量。",
          "nQ = 4",
          [items("Query heads", ["Q0", "Q1", "Q2", "Q3"], "graph")],
        ),
        frame(
          "按组共享键和值",
          `每组查询读取同一组 K / V。当前使用 ${kv} 个 KV 头；共享的是键值，不是查询向量。`,
          "KV group(i) = floor(i × nKV / nQ)",
          [
            items(
              "Q → KV 映射",
              mapping.map((k, i) => `Q${i} → K${k} / V${k}`),
              "graph",
            ),
            items(
              "缓存头组",
              Array.from({ length: kv }, (_, i) => `KV ${i}`),
              "cache",
            ),
          ],
          { kvHeads: kv, queryHeads: 4 },
        ),
        frame(
          "分别计算头输出",
          "每个 Q 与自己组的键计算注意力权重，因此共享同一个 V 并不意味着两个 Q 的结果相同。",
          "Oᵢ = softmax(QᵢKgroupᵀ)Vgroup",
          [
            matrix("示意各 Q 的读取权重", [
              [0.8, 0.2],
              [0.3, 0.7],
              [0.6, 0.4],
              [0.1, 0.9],
            ]),
            items("各头输出", ["O0", "O1", "O2", "O3"], "graph"),
          ],
        ),
        frame(
          "缓存按 KV 头增长",
          "同样的序列长度和头维度下，K / V 缓存元素数量随 KV 头数改变。下方是逻辑元素比例，非实测速度。",
          "cache ratio = nKV / nQ",
          [bars("相对缓存规模", [kv / 4, 1], ["当前共享", "4 KV 对照"])],
          { cacheRatio: kv / 4, kvHeads: kv },
        ),
      ];
    }
    case "gated-deltanet": {
      const trace = buildDeltaTrace(p.alpha, 0.7);
      return [
        frame(
          "有限递归状态",
          "每个请求保存一个固定 2×2 教学状态，以及 3 个输入的局部卷积缓冲。本例采用 value × key 状态约定。",
          "S₀ = 0 · buffer=[0,0,0]",
          [
            matrix("初始 S", trace[0].state),
            vector("卷积缓冲", trace[0].buffer),
          ],
        ),
        ...trace
          .slice(1)
          .map((t, i) =>
            frame(
              `状态更新 ${i + 1}`,
              `第 ${i + 1} 个输入移入局部缓冲；旧状态先乘 α=${p.alpha}，用 value−预测值的误差做秩一更新，再以 q 读取状态。卷积为独立局部支路读数。`,
              "S̄=αS · e=v−S̄k · S=S̄+βekᵀ · o=Sq",
              [
                vector("卷积缓冲 / 局部输入", t.buffer),
                vector("k", t.key),
                vector("误差 e", t.error),
                matrix("衰减状态 αS", t.decayed),
                matrix("更新后状态 S", t.state),
                vector("读取输出 Sq", t.output),
              ],
              { alpha: p.alpha, beta: 0.7, convolution: t.conv },
              t.output,
            ),
          ),
      ];
    }
    case "moe": {
      const router = [1.2 + p.bias, -0.4, 0.6, 1.4, 0.1, -0.8, 0.5, 0.9],
        probs = softmax(router, 1),
        selected = probs
          .map((v, i) => ({ v, i }))
          .sort((a, b) => b.v - a.v)
          .slice(0, 2)
          .map((v) => v.i),
        mass = selected.reduce((s, i) => s + probs[i], 0),
        weights = selected.map((i) => probs[i] / mass),
        expert = selected.map((i) => [(i + 1) * 0.1, 0.2 - i * 0.05]),
        routed = [0, 1].map((c) =>
          weights.reduce((s, w, j) => s + w * expert[j][c], 0),
        ),
        shared = [0.15, -0.05],
        out = routed.map((v, i) => v + shared[i]);
      return [
        frame(
          "路由器给专家打分",
          "教学路由器给 8 个专家产生 logits；只改变 E0 偏置即可观察选中专家的切换。真实专家数量见模型结构页。",
          "router logits → softmax",
          [
            bars(
              "8 个专家概率",
              probs,
              probs.map((_, i) => `E${i}`),
            ),
          ],
        ),
        frame(
          "选出 top-2",
          "选择得分最高的两个专家，并在选中集合内重新归一化。其余专家本轮不参与此 Token 的计算。",
          "w = selected_probability / selected_mass",
          [
            {
              label: "专家选择",
              kind: "experts",
              values: probs,
              items: probs.map((_, i) => `E${i}`),
              selected,
            },
            vector("选中专家的权重", weights),
          ],
          { selectedExperts: 2 },
        ),
        frame(
          "并行计算选中专家",
          "两位专家对同一输入产生各自的教学输出；共享专家始终参与，和路由专家分支分别计算。",
          "Ei(x), shared(x)",
          [
            vector(`E${selected[0]} 输出`, expert[0]),
            vector(`E${selected[1]} 输出`, expert[1]),
            vector("共享专家输出", shared),
          ],
        ),
        frame(
          "加权融合与共享相加",
          "先用路由权重融合两位专家输出，再加上共享分支。这里明确展示 8 选 2 教学模型，非真实 Qwen 权重。",
          "y = Σ wᵢEi(x) + shared(x)",
          [
            vector("路由融合结果", routed),
            vector("共享分支", shared),
            vector("MoE 输出", out),
          ],
          {},
          out,
        ),
      ];
    }
    case "prefill": {
      const n = Math.round(p.tokens),
        tokens = Array.from({ length: n }, (_, i) => `t${i}`),
        visibility = Array.from({ length: n }, (_, i) =>
          Array.from({ length: n }, (_, j) => (j <= i ? 1 : 0)),
        );
      return [
        frame(
          "接收完整输入序列",
          "输入的全部 Token 已知，可以把多个位置的投影合成矩阵计算。逻辑形状为 [1,T,d]。",
          `T = ${n}`,
          [
            items("输入 Token", tokens),
            matrix(
              "教学隐藏表示 · T × 2",
              tokens.map((_, i) => [i * 0.2, 1 - i * 0.1]),
            ),
          ],
        ),
        frame(
          "输入位置并行经过层",
          "多个输入位置同一轮进入网络，但因果遮罩仍然保证每个位置只看过去和自身，不能看未来。",
          "mask(i,j) = 1 if j ≤ i",
          [
            matrix("各位置可见性", visibility),
            items("逻辑计算路径", ["QKV", "RoPE / Attention", "FFN"], "graph"),
          ],
        ),
        frame(
          "建立每层 KV",
          "全注意力层为每个已经计算的输入位置保存 K 和 V。递归层则更新自己的固定状态，不保存相同的全量 KV。",
          "cache positions = processed input positions",
          [items("已写入 KV 的位置", tokens, "cache")],
          { cachedTokens: n },
        ),
        frame(
          "最后位置预测下一词",
          "最后一个输入位置的隐藏向量用于预测第一个生成 Token。此时采样出的新 Token 尚未成为缓存中的输入。",
          "h[T−1] → LM Head → sample",
          [
            items("最后位置", [tokens.at(-1)!], "tokens", [0]),
            items("第一个生成 Token", ["g0 · 尚未写入缓存"], "tokens"),
          ],
          { cachedTokens: n, sampledTokens: 1 },
        ),
      ];
    }
    case "decode":
      return [
        frame(
          "读取单个新输入",
          "上一轮采样出的 Token 作为这一轮输入；历史位置的 K / V 已存在，当前只投影新位置。",
          "input = previous sampled token",
          [
            items("本轮输入", ["g0"], "tokens", [0]),
            items("旧 KV", ["t0", "t1", "t2"], "cache"),
          ],
          { cachedTokens: 3 },
        ),
        frame(
          "追加当前位置 KV",
          "新输入经过 QKV 和位置运算，其 K / V 追加到本请求、当前层缓存。缓存此时包含已处理 g0。",
          "KV ← KV ∪ {K(g0),V(g0)}",
          [
            items("已处理缓存", ["t0", "t1", "t2", "g0"], "cache", [3]),
            vector("当前 Q", [0.4, 0.8]),
          ],
          { cachedTokens: 4 },
        ),
        frame(
          "单行读取全部历史",
          "当前 Q 与缓存中的全部 K 比较，读取历史 V。查询长度为 1，KV 长度为 4，而不是重算 4 个输入。",
          "Q: [1,d] · K: [4,d]",
          [
            bars(
              "这一行注意力",
              [0.1, 0.2, 0.3, 0.4],
              ["t0", "t1", "t2", "g0"],
            ),
            vector("聚合输出", [0.7, 0.6]),
          ],
          { queryPositions: 1, cachedTokens: 4 },
        ),
        frame(
          "预测并流式返回",
          "LM Head 和采样得到 g1，先返回给用户。g1 下一轮才作为输入进入缓存，循环直到停止条件满足。",
          "sample → stream → next input",
          [
            items("生成输出", ["g0", "g1"], "tokens", [1]),
            items("缓存仍止于 g0", ["t0", "t1", "t2", "g0"], "cache"),
          ],
          { cachedTokens: 4, sampledTokens: 2 },
        ),
      ];
    case "sampling": {
      const probs = softmax(logits, p.temperature),
        filtered = filterDistribution(probs, Math.round(p.topK), 1),
        random = 0.42;
      let sum = 0,
        choice = 0;
      for (let i = 0; i < filtered.length; i++) {
        sum += filtered[i];
        if (random < sum) {
          choice = i;
          break;
        }
      }
      return [
        frame(
          "温度调整 logits",
          "对同一组 logits 使用温度 Softmax。温度零选择最高分候选；正温度得到可采样的概率分布。",
          "p = softmax(logits / T)",
          [bars("温度后的概率", probs, tokenNames)],
          { temperature: p.temperature },
        ),
        frame(
          "只保留 top-k",
          "去掉排名低于 k 的候选，再对保留概率重新归一化。这里 top-p 固定为 1，不再裁剪累计质量。",
          `k=${Math.round(p.topK)} · top-p=1`,
          [
            bars("裁剪前", probs, tokenNames),
            bars("裁剪并归一化后", filtered, tokenNames),
          ],
        ),
        frame(
          "用固定随机数落区间",
          "按词表顺序累加概率，固定随机数 0.42 落入哪个区间就选择哪个 Token。相同输入与参数可确定地回放。",
          "choose first i with Σⱼ≤ᵢ pⱼ > u",
          [
            bars("采样概率区间", filtered, tokenNames, [choice]),
            vector(
              "累计概率",
              filtered.map((_, i) =>
                filtered.slice(0, i + 1).reduce((a, b) => a + b, 0),
              ),
            ),
          ],
          { random, sampledId: choice },
        ),
        frame(
          "返回所选 Token",
          "采样输出是一个离散 Token ID，不能直接添加到 KV；必须先在下一轮经过对应层的计算。",
          "selected ID → stream / next decode",
          [
            items(
              "采样结果",
              [`${tokenNames[choice]} · ID ${choice}`],
              "tokens",
              [0],
            ),
            vector("最终分布", filtered),
          ],
          { sampledId: choice },
          filtered,
        ),
      ];
    }
    case "lm-head": {
      const h = [1, 0.5],
        w = [
          [1, -1, 1, 0],
          [0, 0, 1, 1],
        ],
        out = multiply(h, w),
        probs = softmax(out, 1);
      return [
        frame(
          "最后位置的隐藏向量",
          "最终归一化后的隐藏向量被送入 LM Head。本例只有 2 个隐藏通道和 4 个词表候选。",
          "h = [1,0.5]",
          [vector("最后位置 h", h), items("教学词表", tokenNames)],
        ),
        frame(
          "逐候选点积",
          "每个词表候选对应投影矩阵的一列。第 2 个候选的 logit 为 1×1+0.5×1=1.5。",
          "logitⱼ = Σᵢ hᵢWᵢⱼ",
          [
            matrix("词表投影 W · 2 × 4", w, undefined, [2, 6]),
            vector("候选 2 的贡献", [1, 0.5]),
          ],
          { contribution: 1.5 },
        ),
        frame(
          "写出词表 logits",
          "四个点积组成原始分数向量；这里还没有采样。LM Head 将隐藏宽度映射为词表大小。",
          "z=hW = [1,−1,1.5,0.5]",
          [bars("logits", out, tokenNames), vector("原始输出 z", out)],
          {},
          out,
        ),
        frame(
          "把分数交给采样",
          "logits 经过温度、概率归一化和筛选，才会选出离散 Token。图中列出 T=1 的概率作衔接。",
          "logits → Softmax → sampling",
          [
            bars("T=1 概率", probs, tokenNames),
            items("输出链路", ["logits", "概率", "Token ID"], "graph"),
          ],
          {},
          out,
        ),
      ];
    }
    case "kv-cache":
      return [
        frame(
          "输入先经过层",
          "三个输入位置先经过当前全注意力层。缓存保存的是处理后的 K / V，而不是尚未计算的文本或 Token ID。",
          "t0,t1,t2 → K/V projection",
          [
            items("输入位置", ["t0", "t1", "t2"]),
            items("空缓存", ["未分配"], "cache"),
          ],
          { cachedTokens: 0 },
        ),
        frame(
          "写入已处理输入",
          "本请求当前层写入三个位置的键和值。不同请求、不同层分别维护缓存，不能直接互相读取。",
          "request A / layer ℓ / positions 0…2",
          [
            matrix("K cache", [
              [1, 0],
              [0, 1],
              [0.5, 0.5],
            ]),
            matrix("V cache", [
              [1, 1],
              [0.2, 0.4],
              [0.5, 0.8],
            ]),
          ],
          { cachedTokens: 3 },
        ),
        frame(
          "采样尚未写入",
          "网络预测并采样出 g0，但尚未处理 g0 对应的隐藏状态。缓存长度保持 3，不能提前写入新 Token 的 K / V。",
          "sampled g0 ≠ cached g0",
          [
            items("缓存", ["t0", "t1", "t2"], "cache"),
            items("待处理输入", ["g0"], "tokens", [0]),
          ],
          { cachedTokens: 3 },
        ),
        frame(
          "下一轮写入",
          "下一 decode 轮把 g0 作为输入，算出它的 K / V 后追加到缓存；历史三个位置的投影被复用。",
          "cache length: 3 → 4",
          [items("缓存", ["t0", "t1", "t2", "g0"], "cache", [3])],
          { cachedTokens: 4 },
        ),
        frame(
          "停止与释放",
          "停止条件满足后，请求完成。为该请求分配的缓存块回到可用池；其他请求的缓存不受影响。",
          "finished → free(request A blocks)",
          [
            items("请求状态", ["A · 已完成"], "queue"),
            items("缓存池", ["空闲块", "空闲块"], "cache"),
          ],
          { cachedTokens: 0, releasedTokens: 4 },
        ),
      ];
    case "paged-attention": {
      const size = Math.round(p.blockSize),
        counts = [5, 2],
        allocations = counts.map((n) => Math.ceil(n / size)),
        blocks = allocations.reduce((a, b) => a + b, 0),
        physical = [3, 0, 5, 1, 6, 2, 4],
        table = allocations.map((n, r) =>
          Array.from(
            { length: n },
            (_, i) => physical[(r ? allocations[0] : 0) + i],
          ),
        );
      return [
        frame(
          "请求拥有逻辑序列",
          "请求 A 有 5 个已处理 Token，请求 B 有 2 个。逻辑位置连续，但不要求物理存储紧挨在一起。",
          "A:5 positions · B:2 positions",
          [
            items("A 的逻辑位置", ["A0", "A1", "A2", "A3", "A4"]),
            items("B 的逻辑位置", ["B0", "B1"]),
          ],
        ),
        frame(
          "按容量划为块",
          `每块容量 ${size} 个 Token；每个请求独立向上取整。尾块可能尚未填满，块内空位属于该请求的分配。`,
          "nblocks = ceil(length / blockSize)",
          [
            items(
              "逻辑块",
              table.flatMap((row, r) =>
                row.map((_, i) => `${r ? "B" : "A"} · 块${i}`),
              ),
              "cache",
            ),
          ],
          { blocks, capacity: blocks * size, slack: blocks * size - 7 },
        ),
        frame(
          "页表映射物理存储",
          "页表记录逻辑块对应的物理块编号。示例物理编号刻意不连续，注意力读取通过页表定位正确位置。",
          "logical block → block table → physical block",
          [
            items(
              "A 页表",
              table[0].map((v, i) => `${i} → 物理 ${v}`),
              "graph",
            ),
            items(
              "B 页表",
              table[1].map((v, i) => `${i} → 物理 ${v}`),
              "graph",
            ),
            items(
              "物理池 · 示意",
              physical.map((v) => `块 ${v}`),
              "cache",
              physical.map((_, i) => i).slice(0, blocks),
            ),
          ],
        ),
        frame(
          "读取与回收以块为单位",
          "调度器按页表读取 K / V；完成请求释放自己的物理块。以下读数仅算教学容量和尾部空位。",
          "read(table, position) · free(request blocks)",
          [
            bars(
              "逻辑 Token 与已分配容量",
              [7, blocks * size],
              ["有效 Token", "容量"],
            ),
            items(
              "A / B 分配块数",
              allocations.map((n, i) => `${i ? "B" : "A"}: ${n} 块`),
              "queue",
            ),
          ],
          { blocks, capacity: blocks * size, slack: blocks * size - 7 },
        ),
      ];
    }
    case "prefix-caching": {
      const n = Math.round(p.shared),
        common = ["你", "是", "助手", "。"].slice(0, n),
        a = [...common, "讲", "猫"],
        b = [...common, "讲", "狗"];
      return [
        frame(
          "第一请求建立前缀",
          "A 的输入先经过网络并保存计算状态。可复用缓存必须关联完全一致的 Token 前缀及相容的模型配置。",
          "A → compute → cached prefix",
          [
            items("请求 A", a),
            items("缓存前缀", common.length ? common : ["无共同前缀"], "cache"),
          ],
          { computedTokens: a.length },
        ),
        frame(
          "新请求比对前缀",
          "B 从开头逐项匹配；一旦不一致，后续位置需要重新计算。这里只指定已验证的共同前缀长度。",
          "match token IDs from start",
          [
            items(
              "请求 B",
              b,
              "tokens",
              common.map((_, i) => i),
            ),
            items("可匹配路径", common.length ? common : ["没有命中"], "graph"),
          ],
          { matchedTokens: n },
        ),
        frame(
          "命中共享已有计算",
          "共同前缀使用已计算的缓存，私有后缀归 B 单独计算。这里只共享只读前缀，不能混用两条请求的后续状态。",
          "reused prefix + private suffix",
          [
            items("共享前缀", common.length ? common : ["0 个 Token"], "cache"),
            items("B 私有后缀", ["讲", "狗"], "tokens", [0, 1]),
          ],
          { reusedTokens: n, newlyComputedTokens: 2 },
        ),
        frame(
          "只继续计算新后缀",
          "B 的后缀读取相同前缀上下文并继续计算。生成分叉后，各请求后续缓存独立；命中减少计算，不给出速度承诺。",
          "new compute = total input − reused prefix",
          [bars("B 处理的逻辑位置", [n, 2], ["复用", "新计算"])],
          { reusedTokens: n, newlyComputedTokens: 2 },
        ),
      ];
    }
    case "batching": {
      const slots = Math.round(p.slots),
        queue = ["A", "B", "C"],
        remaining = { A: 1, B: 2, C: 3 },
        active: string[] = [],
        done: string[] = [],
        result: MechanismFrame[] = [];
      result.push(
        frame(
          "三个请求进入队列",
          "A、B、C 的生成长度分别为 1、2、3。调度器最多同时执行指定槽位数，请求保持独立缓存。",
          `slots=${slots}`,
          [
            items("等待队列", queue, "queue"),
            items(
              "空闲槽位",
              Array.from({ length: slots }, (_, i) => `槽 ${i} 空闲`),
              "cache",
            ),
          ],
          { waiting: 3, allocated: 0, completed: 0 },
        ),
      );
      let tick = 0;
      while (done.length < 3) {
        tick++;
        while (active.length < slots && queue.length)
          active.push(queue.shift()!);
        const running = [...active];
        running.forEach((r) => remaining[r as keyof typeof remaining]--);
        const ended = active.filter(
          (r) => remaining[r as keyof typeof remaining] === 0,
        );
        done.push(...ended);
        const panels = [
          items(
            "这一轮执行",
            running.map(
              (r) => `${r} · 剩余 ${remaining[r as keyof typeof remaining]}`,
            ),
            "queue",
            running.map((_, i) => i),
          ),
          items("等待入场", queue.length ? [...queue] : ["队列空"], "queue"),
          items(
            "本轮完成并释放",
            ended.length ? [...ended] : ["暂无"],
            "cache",
          ),
        ];
        result.push(
          frame(
            `调度轮 ${tick}`,
            ended.length
              ? `这一轮执行 ${running.join("、")}；${ended.join("、")} 完成并释放自己的缓存。等待请求在下一轮使用空出的槽位。`
              : `这一轮执行 ${running.join("、")}，每个活动请求推进一个生成位置；等待请求不会读取活动请求的缓存。`,
            "admit → execute one token / request → finish → free",
            panels,
            {
              waiting: queue.length,
              allocated: active.length - ended.length,
              completed: done.length,
            },
          ),
        );
        active.splice(
          0,
          active.length,
          ...active.filter((r) => !ended.includes(r)),
        );
      }
      result.push(
        frame(
          "所有请求结束",
          "三个请求全部完成，等待队列和活动槽位为空。请求完成时释放分配，下一批请求可以使用相同的物理池。",
          "queue=0 · active=0 · allocated=0",
          [
            items("已完成请求", [...done], "queue"),
            items(
              "缓存池",
              Array.from({ length: slots }, (_, i) => `槽 ${i} 空闲`),
              "cache",
            ),
          ],
          { waiting: 0, allocated: 0, completed: 3 },
        ),
      );
      return result;
    }
    case "quantization": {
      const w = [-1, -0.63, -0.1, 0.17, 0.72, 1].map((v) => v * p.amplitude),
        result = quantizeSymmetric(w, p.bits === 8 ? 8 : 4);
      return [
        frame(
          "浮点教学权重",
          "6 个小权重包含正负与零附近的数。这里演示对称整向量量化；真实系统的分组、激活和 kernel 有更多差异。",
          "weights ∈ ℝ⁶",
          [
            bars(
              "原始权重",
              w,
              w.map((_, i) => `w${i}`),
            ),
          ],
        ),
        frame(
          "确定 scale",
          "最大绝对权重除以整数正边界得到 scale。全零权重使用 scale=1，量化码全为零且重建保持为零。",
          "s = max|w| / (2^(b−1)−1)",
          [
            vector("scale", [result.scale]),
            vector("位宽与整数边界", [p.bits, 2 ** (p.bits - 1) - 1]),
          ],
          { scale: result.scale },
        ),
        frame(
          "舍入得到整数码",
          "每个权重除以 scale、四舍五入并裁剪到整数范围。位宽改变会重新计算全部量化码，不只改变图形。",
          "q = clamp(round(w/s), −bound, bound)",
          [vector("原始权重 w", w), vector("整数码 q", result.codes)],
        ),
        frame(
          "重建并计算误差",
          "整数码乘回 scale 得到近似权重。统计均方误差与打包后的整数 payload；不包括 scale 或其他元数据。",
          "ŵ=q·s · MSE=mean((w−ŵ)²)",
          [
            {
              label: "原始 / 重建权重",
              kind: "plot",
              matrix: [w, result.reconstructed],
            },
            vector("重建 ŵ", result.reconstructed),
          ],
          {
            mse: result.meanSquaredError,
            payloadBytes: result.payloadBytes,
            scale: result.scale,
          },
          result.reconstructed,
        ),
      ];
    }
    case "speculation": {
      const n = Math.round(p.accepted),
        draft = ["猫", "坐", "垫"],
        accepted = draft.slice(0, n),
        corrected = n < 3 ? "上" : "。";
      return [
        frame(
          "草稿模型提出候选",
          "小草稿模型顺序提出 3 个候选；草稿只是建议，必须由目标模型验证后才能进入最终输出。",
          "draft = [猫, 坐, 垫]",
          [items("已确认前缀", ["一只"]), items("候选 Token", draft)],
        ),
        frame(
          "目标模型并行验证",
          "目标模型处理草稿候选的条件位置，得到用于各个候选验证的分布。下方判定为固定教学 fixture，不是猜测相同字即算完整采样校正。",
          "target distributions at draft positions",
          [
            items(
              "候选验证",
              draft.map(
                (s, i) =>
                  `${s} · ${i < n ? "接受" : i === n ? "拒绝" : "丢弃"}`,
              ),
              "queue",
              Array.from({ length: n }, (_, i) => i),
            ),
          ],
          { accepted: n },
        ),
        frame(
          "在首次拒绝处停止",
          n < 3
            ? `前 ${n} 个候选被接受，第 ${n + 1} 个拒绝；后续草稿不能越过拒绝点，相关未确认状态也要回滚。`
            : "三个候选全部被接受，目标模型可以再提供一个 bonus Token。最终步进数量与接受长度直接相关。",
          "accepted prefix + corrected / bonus token",
          [
            items(
              "保留候选",
              accepted.length ? accepted : ["没有草稿被接受"],
              "tokens",
            ),
            items(
              n < 3 ? "修正 Token" : "bonus Token",
              [corrected],
              "tokens",
              [0],
            ),
          ],
          { accepted: n, discarded: 3 - n },
        ),
        frame(
          "确认输出和状态",
          "只保留已确认的候选前缀与目标模型给出的新 Token；下一轮从确认边界继续。速度取决于接受率和执行代价，示例不估算加速。",
          "commit verified prefix · rollback rejected suffix",
          [
            items("本轮确认输出", [...accepted, corrected], "tokens"),
            items("下一轮边界", ["已确认位置之后"], "cache"),
          ],
          { committed: n + 1 },
        ),
      ];
    }
    case "flash-attention": {
      const scores = [0.5, 1, -0.5, 1.5],
        v = [1, 2, -1, 0.5];
      let m = -Infinity,
        l = 0,
        o = 0;
      const tileFrames: MechanismFrame[] = [];
      for (let i = 0; i < 2; i++) {
        const s = scores.slice(i * 2, i * 2 + 2),
          values = v.slice(i * 2, i * 2 + 2),
          nextM = Math.max(m, ...s),
          rescale = m === -Infinity ? 0 : Math.exp(m - nextM),
          exp = s.map((x) => Math.exp(x - nextM));
        l = l * rescale + exp.reduce((a, b) => a + b, 0);
        o = o * rescale + exp.reduce((sum, e, j) => sum + e * values[j], 0);
        m = nextM;
        tileFrames.push(
          frame(
            `归约第 ${i + 1} 个 tile`,
            `第 ${i + 1} 个 tile 的分数和 V 载入局部工作区。最大值更新为 ${m}；重缩放旧累积值，再加入这一块的新贡献，保持精确 Softmax 归约。`,
            "m′=max(m,tile) · l′=l·exp(m−m′)+Σexp(s−m′)",
            [
              vector("当前分数 tile", s),
              vector("当前 V tile", values),
              vector("在线累积 m / l / o", [m, l, o]),
            ],
            { max: m, denominator: l, numerator: o },
          ),
        );
      }
      return [
        frame(
          "完整行按 tile 拆开",
          "一行 4 个分数拆成两个小块；算法不需要把全局注意力概率矩阵写回显存，仍然计算精确注意力而不是稀疏近似。",
          "scores = [0.5,1,−0.5,1.5]",
          [
            matrix("分数 tiles", [
              [0.5, 1],
              [-0.5, 1.5],
            ]),
            vector("V 内容", v),
          ],
        ),
        ...tileFrames,
        frame(
          "最后一次归一化",
          "所有 tile 处理完后用累积加权和除以累积分母。与完整 Softmax 加权 V 相同，差异主要来自浮点求和顺序。",
          "output = o / l",
          [
            vector("在线结果", [o / l]),
            vector("完整计算对照", [
              softmax(scores, 1).reduce((sum, a, i) => sum + a * v[i], 0),
            ]),
          ],
          { denominator: l },
          [o / l],
        ),
      ];
    }
    case "parallelism": {
      const x = [1, 2, 3, 4],
        w = [
          [1, 0],
          [0, 1],
          [0.5, 0.5],
          [-0.5, 1],
        ],
        devices = Math.round(p.devices),
        local =
          devices === 1
            ? [multiply(x, w)]
            : [
                multiply(x.slice(0, 2), w.slice(0, 2)),
                multiply(x.slice(2), w.slice(2)),
              ],
        out = [0, 1].map((c) => local.reduce((s, v) => s + v[c], 0));
      return [
        frame(
          "确定张量分片",
          "本例演示按输入通道切分的 row-parallel 线性层。输入向量与权重行同步分片，逻辑输出仍是同一个向量。",
          "y = xW · split input dimension",
          [vector("全局输入", x), matrix("全局权重", w)],
        ),
        frame(
          "每设备计算局部贡献",
          "每台设备只计算自己输入通道对应的部分点积；结果都是输出宽度 2 的局部贡献，尚不是完整结果。",
          "partialᵣ = xᵣWᵣ",
          local.map((v, i) => vector(`设备 ${i} 局部贡献`, v)),
          { devices },
        ),
        frame(
          "通信归约到全局结果",
          "对局部输出做 sum 归约，恢复与单设备计算相同的结果。通信是算法必需部分，示例不测量链路延迟。",
          "y = all-reduce-sum(partials)",
          [
            ...local.map((v, i) => vector(`设备 ${i}`, v)),
            vector("合并输出", out),
          ],
          { devices },
          out,
        ),
        frame(
          "保持并行方案的边界",
          "张量并行切分同一层；流水线并行切分层；数据并行复制模型处理不同请求。当前示例仅计算张量并行的线性层。",
          "tensor: channels · pipeline: layers · data: requests",
          [
            items(
              "三种分工",
              [
                "张量：设备共同计算一层",
                "流水线：层间传递隐藏状态",
                "数据：副本独立处理请求",
              ],
              "graph",
            ),
          ],
          { devices },
          out,
        ),
      ];
    }
    case "chunked-prefill": {
      const chunk = Math.round(p.chunk),
        total = 6,
        rounds = Math.ceil(total / chunk);
      return [
        frame(
          "长输入与解码同时等待",
          "A 需要处理 6 个输入位置；B 已处在 decode，每轮需要 1 个新位置。调度器为两类工作安排 Token 预算。",
          "A prefill=6 · B decode=1 token / round",
          [
            items(
              "A 待处理输入",
              Array.from({ length: 6 }, (_, i) => `A${i}`),
            ),
            items("B 解码请求", ["B · 继续生成"], "queue"),
          ],
        ),
        ...Array.from({ length: rounds }, (_, i) => {
          const start = i * chunk,
            consumed = Math.min(total, start + chunk),
            count = consumed - start;
          return frame(
            `输入块调度 ${i + 1}`,
            `A 这一轮处理位置 ${start} 到 ${consumed - 1}；B 同轮推进 1 个 decode 位置。A 的缓存只增长已经完成的输入块。`,
            `round token budget = ${count} prefill + 1 decode`,
            [
              items(
                "本轮 A 输入块",
                Array.from({ length: count }, (_, j) => `A${start + j}`),
                "tokens",
                Array.from({ length: count }, (_, j) => j),
              ),
              items("B 本轮", [`B${i} · decode`], "queue"),
              items(
                "A 已完成缓存",
                Array.from({ length: consumed }, (_, j) => `A${j}`),
                "cache",
              ),
            ],
            {
              processedInput: consumed,
              decodeSteps: i + 1,
              tokenBudget: count + 1,
            },
          );
        }),
        frame(
          "输入结束才开始首个采样",
          "A 的 6 个输入位置全部完成，最后输入隐藏状态才能预测首个生成 Token。分块改变调度粒度，不跳过输入网络运算。",
          "prefill complete → first output",
          [
            items("A 状态", ["输入完成", "首 Token 采样"], "graph"),
            items("B 状态", [`已推进 ${rounds} 轮 decode`], "queue"),
          ],
          { processedInput: 6, decodeSteps: rounds },
        ),
      ];
    }
    case "cuda-graphs":
      return [
        frame(
          "通常的逐次启动",
          "在适合捕获的固定形状执行路径上，多次 kernel 启动组成依赖链。图示仅表示主机启动步骤，不代表 kernel 执行时间。",
          "host launch → K1 → K2 → K3",
          [
            items(
              "逐个启动",
              ["主机 → QKV", "主机 → Attention", "主机 → FFN"],
              "graph",
            ),
          ],
          { hostLaunchCalls: 3 },
        ),
        frame(
          "捕获图与依赖",
          "在满足捕获限制的路径上记录节点、依赖和缓冲区地址。先执行必要的准备，之后才能复用图实例。",
          "capture(nodes, dependencies, memory addresses)",
          [
            items(
              "捕获的 DAG",
              ["QKV → Attention", "Attention → FFN", "FFN → 输出"],
              "graph",
            ),
            items("形状契约", ["固定张量形状", "稳定内存地址"], "cache"),
          ],
        ),
        frame(
          "实例化后重放",
          "新一轮输入写入适当的缓冲区，再调用图重放。kernel 的数学运算仍执行，只是启动组织方式被复用。",
          "copy/update input → graph launch",
          [
            items("本轮输入缓冲", ["新 Token 数据"], "tokens", [0]),
            items("一次图启动", ["QKV → Attention → FFN"], "graph"),
          ],
          { hostLaunchCalls: 1 },
        ),
        frame(
          "形状改变需匹配其他路径",
          "不同批大小或张量形状可能匹配另一份已捕获图，也可能走普通执行路径。图捕获不自动让任意动态形状成立。",
          "new shape → compatible graph or eager path",
          [
            items(
              "调度分支",
              ["匹配形状 → 重放对应图", "未匹配 → 常规执行 / 重新准备"],
              "graph",
            ),
          ],
          { hostLaunchCalls: 1 },
        ),
      ];
    case "metrics": {
      const gap = p.gap,
        first = 3,
        times = [first, first + gap, first + 2 * gap],
        duration = times[2],
        throughput = 3 / duration;
      return [
        frame(
          "定义测量边界",
          "请求在教学步骤 0 到达，步骤 1 开始执行。等待时间属于端到端延迟；这里所有单位均为模拟步骤。",
          "arrival=0 · execution start=1",
          [items("事件起点", ["0 到达", "1 开始执行"], "graph")],
          { arrival: 0, start: 1, waiting: 1 },
        ),
        frame(
          "首 Token 返回",
          "首个 Token 在步骤 3 返回，所以端到端 TTFT 为 3 步，包含排队和 prefill；不能仅用执行开始后的时间代替。",
          "TTFT = first token time − arrival",
          [bars("首 Token 延迟组成", [1, 2], ["等待步骤", "执行步骤"])],
          { ttft: first, waiting: 1 },
        ),
        frame(
          "生成间隔与单请求速率",
          "后续两个 Token 以设定的教学间隔返回。ITL 使用相邻 Token 返回时间差，首 Token 本身不计入后续间隔。",
          "ITL = (last − first) / (N−1)",
          [
            vector("三个 Token 返回步骤", times),
            bars("后续 Token 间隔", [gap, gap], ["1→2", "2→3"]),
          ],
          { itl: gap },
        ),
        frame(
          "全窗口吞吐口径",
          "以到达至最后 Token 的完整窗口计，本例吞吐为 3 除以窗口步长。真实系统需注明请求集合、计时窗口及是否含输入 Token。",
          "output throughput = output tokens / observation window",
          [
            bars(
              "不同统计口径",
              [3 / duration, 1 / gap],
              ["全窗口 Token/步", "后续 Token/步"],
            ),
          ],
          { throughput, ttft: first, itl: gap, outputTokens: 3 },
        ),
      ];
    }
    case "frameworks":
      return [
        frame(
          "接口解析与请求入队",
          "接口层解析输入、模型与生成参数，转成标准请求。API 风格相似不意味着后端执行路径相同。",
          "HTTP / API → request object",
          [
            items(
              "接口层",
              ["输入文本", "模型 ID", "采样 / 停止参数"],
              "tokens",
            ),
            items("请求队列", ["A · 等待调度"], "queue"),
          ],
        ),
        frame(
          "调度器管理生命周期",
          "调度器管理等待、prefill、decode 和完成，决定这一轮活动请求及缓存预算。它与底层 kernel 分工不同。",
          "queue + token / memory budget → batch",
          [
            items("调度器", ["选择请求", "分配缓存", "构造批次"], "graph"),
            items("本轮批次", ["A · prefill", "B · decode"], "queue"),
          ],
        ),
        frame(
          "执行器运行网络与 kernel",
          "执行器组织网络、设备和缓存布局；kernel 执行矩阵乘、注意力等算子。可选编译或图捕获服务于具体形状和设备。",
          "executor → model ops → kernels",
          [
            items(
              "执行层级",
              ["执行器 / 模型图", "算子 / 通信", "kernel / 设备"],
              "graph",
            ),
            matrix("教学计算结果", smallW),
          ],
        ),
        frame(
          "流式输出并回收",
          "采样后的 Token 转成文本增量并回到客户端；未完成请求回到调度循环，完成请求释放资源。示意架构不声称所有框架实现完全相同。",
          "sample → stream → continue / finish",
          [
            items("客户端流", ["新 Token", "文本增量", "完成原因"], "tokens"),
            items("调度反馈", ["继续 → 下一轮", "结束 → 释放缓存"], "graph"),
          ],
        ),
      ];
    default:
      return [];
  }
}
