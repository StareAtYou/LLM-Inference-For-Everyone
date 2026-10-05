import type { Topic } from "../types";
import { models } from "./models";
export const topics: Topic[] = [
  {
    id: "tokenization",
    title: "Token：文字如何变成模型输入",
    englishTerms: ["Tokenization", "Tokenizer"],
    area: "推理基础",
    prerequisites: [],
    levels: {
      beginner: {
        summary: "模型先把文字切成词表里的片段，再把片段换成整数。",
        explanation:
          "模型先把文字切成词表里的片段，再把片段换成整数。一个字不一定对应一个 Token，英文单词也可能被拆开。",
        details: ["先观察文字被切成几个片段：长度按片段数统计，而不是按字数。"],
      },
      advanced: {
        summary: "分词器用固定词表和编码规则把文本映射为 token IDs。",
        explanation:
          "分词器用固定词表和编码规则把文本映射为 token IDs。聊天模板还会加入角色、消息边界与生成提示；输入长度必须在模板完成后统计。",
        details: [
          "编码顺序是聊天模板 → 文本分词 → 特殊 Token 合并。角色标记也占长度；同一句话放入不同模板，前缀 Token 序列可能不同。",
        ],
      },
      expert: {
        summary: "服务端与客户端应使用同一 tokenizer 和聊天模板。",
        explanation:
          "服务端与客户端应使用同一 tokenizer 和聊天模板。前缀缓存匹配的是完整 Token 序列，字符相似不保证编码一致；特殊 Token 与停止条件也要成对核对。",
        details: [
          "排查缓存不命中时，逐项比较 tokenizer revision、模板、特殊 Token 与 Token ID，而非仅比较字符串。若客户端自行截断，应确认是否破坏消息边界与停止标记。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/pipeline",
      },
    ],
    sources: [
      {
        id: "1706.03762",
        title: "原始研究 · Tokenization",
        url: "https://arxiv.org/abs/1706.03762",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "embedding",
    title: "Embedding：从整数到向量",
    englishTerms: ["Embedding"],
    area: "模型计算",
    prerequisites: ["tokenization"],
    levels: {
      beginner: {
        summary: "Token ID 像词典页码。",
        explanation:
          "Token ID 像词典页码。Embedding 用这个页码取出一行数字，形成模型可以计算的向量。",
        details: [
          "一个编号只负责找到词表中的一行；这一行向量随后会被网络不断改写。",
        ],
      },
      advanced: {
        summary: "查表结果形状为 [序列长度, hidden_size]。",
        explanation:
          "查表结果形状为 [序列长度, hidden_size]。随后每层通过注意力和 FFN 改变表示，输出头将最后位置的表示投影回词表 logits。",
        details: [
          "词表矩阵为 [V,H]，输入索引 [B,T] 查表得到 [B,T,H]；输出头通常把 [B,H] 投影为 [B,V]。查表不是把 Token ID 当连续数值相乘。",
        ],
      },
      expert: {
        summary: "Embedding、输出头与中间激活的精度和形状应分别核对。",
        explanation:
          "Embedding、输出头与中间激活的精度和形状应分别核对。Qwen 全注意力的 head_dim 由配置指定，不能从 hidden_size / Q头数反推。",
        details: [
          "检查输入 embedding 与 lm_head 是否共享权重、词表补齐与张量并行分片，以及 logits 输出精度。模型配置决定投影尺寸，不能假定所有注意力投影宽度都等于 H。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/models",
      },
    ],
    sources: [
      {
        id: "1706.03762",
        title: "原始研究 · Embedding",
        url: "https://arxiv.org/abs/1706.03762",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "attention",
    title: "Attention：让信息在位置之间流动",
    englishTerms: ["Attention", "Softmax"],
    area: "模型计算",
    prerequisites: ["embedding"],
    levels: {
      beginner: {
        summary: "每个位置提出问题 Q，其他位置给出线索 K 和内容 V。",
        explanation:
          "每个位置提出问题 Q，其他位置给出线索 K 和内容 V。相关度决定读取多少内容。因果模型只能看到自己和过去。",
        details: ["把每个位置当成一位读者：它只能根据已经出现的文字寻找线索。"],
      },
      advanced: {
        summary: "标准注意力为 softmax(QKᵀ / √d + mask)V。",
        explanation:
          "标准注意力为 softmax(QKᵀ / √d + mask)V。因果 mask 把未来位置设为负无穷，再做 softmax；每个可见行的权重和为 1。",
        details: [
          "Prefill 的 Q/K/V 带序列轴 T，Decode 的新 Q 只有一位置，K/V 仍覆盖历史 T。softmax 必须在可见 Key 轴归一化；先遮挡再归一化。",
        ],
      },
      expert: {
        summary:
          "Prefill 的分数矩阵规模随长度平方增长，但 FlashAttention 可避免把完整矩阵写入显存。",
        explanation:
          "Prefill 的分数矩阵规模随长度平方增长，但 FlashAttention 可避免把完整矩阵写入显存。Decode 只有新 Query，仍需读取历史 K/V；矩阵图是教学展示，不是内核存储布局。",
        details: [
          "核对 mask、缩放、softmax 精度与 KV 索引的一致性。融合内核可边读取分块边归一化，数学上的 T×T 分数矩阵不要求在 HBM 完整实体化。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/pipeline",
      },
    ],
    sources: [
      {
        id: "1706.03762",
        title: "原始研究 · Attention",
        url: "https://arxiv.org/abs/1706.03762",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "rope",
    title: "RoPE：用旋转表达相对位置",
    englishTerms: ["RoPE", "Rotary Position Embedding"],
    area: "模型计算",
    prerequisites: ["attention"],
    levels: {
      beginner: {
        summary: "让向量随位置旋转，模型就能从两个方向的差异感知相对位置。",
        explanation: "让向量随位置旋转，模型就能从两个方向的差异感知相对位置。",
        details: ["同一向量在不同位置会被旋转，让模型区分“前面”和“后面”。"],
      },
      advanced: {
        summary:
          "每一对通道按位置 m 与频率 θ 旋转：x′=(x₁cos mθ−x₂sin mθ, x₁sin mθ+x₂cos mθ)。",
        explanation:
          "每一对通道按位置 m 与频率 θ 旋转：x′=(x₁cos mθ−x₂sin mθ, x₁sin mθ+x₂cos mθ)。Q 与 K 的内积包含相对位置差。",
        details: [
          "一个通道对 [x₀,x₁] 按位置 m 的角度 mθ 旋转；Q 与 K 使用各自的位置。不同通道对使用不同频率，点积因此包含相对位置关系。",
        ],
      },
      expert: {
        summary: "实际实现包含频率基数、部分旋转维度和长上下文扩展策略。",
        explanation:
          "实际实现包含频率基数、部分旋转维度和长上下文扩展策略。二维演示只展示一个通道对，不能据此断言外推后的上下文质量。",
        details: [
          "缓存位置偏移、旋转通道布局、partial rotary 与长上下文缩放要和实现一致。二维图只有一个频率，不能据此推断 Qwen 全部通道或外推长度效果。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/models",
      },
    ],
    sources: [
      {
        id: "2104.09864",
        title: "原始研究 · RoPE",
        url: "https://arxiv.org/abs/2104.09864",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "ffn",
    title: "FFN：每个位置的非线性变换",
    englishTerms: ["FFN", "SwiGLU", "Dense"],
    area: "模型计算",
    prerequisites: ["embedding"],
    levels: {
      beginner: {
        summary:
          "注意力负责位置之间的交流，FFN 则对每个位置的内容做进一步处理。",
        explanation:
          "注意力负责位置之间的交流，FFN 则对每个位置的内容做进一步处理。Dense 模型的每个 Token 都经过同一组 FFN 权重。",
        details: ["注意力交换不同位置的信息，FFN 则在每个位置内部加工向量。"],
      },
      advanced: {
        summary:
          "门控 FFN 通常包含 gate_proj、up_proj 和 down_proj：silu(xWg) ⊙ (xWu) 再映射回 hidden_size，并接回残差流。",
        explanation:
          "门控 FFN 通常包含 gate_proj、up_proj 和 down_proj：silu(xWg) ⊙ (xWu) 再映射回 hidden_size，并接回残差流。",
        details: [
          "门控 FFN 先做两条 H→I 投影，再将 SiLU(gate) 与 up 逐元素相乘，最后 I→H。输入序列的各位置共享同一组参数。",
        ],
      },
      expert: {
        summary: "Dense 与 MoE 在本网站指 FFN 结构，而不是注意力类型。",
        explanation:
          "Dense 与 MoE 在本网站指 FFN 结构，而不是注意力类型。矩阵维度决定计算量和权重读取；融合门控算子可减少中间张量与内核启动。",
        details: [
          "比较融合 gate/up 投影、激活内核和 down 投影的形状、布局与精度。Dense 与 MoE 的 FFN 路径不同，但残差连接仍要求输出回到 H。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/models",
      },
    ],
    sources: [
      {
        id: "2002.05202",
        title: "GLU Variants Improve Transformer",
        url: "https://arxiv.org/abs/2002.05202",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "gqa",
    title: "GQA：多个 Query 共享 K/V",
    englishTerms: ["GQA", "Grouped Query Attention"],
    area: "模型计算",
    prerequisites: ["attention"],
    levels: {
      beginner: {
        summary:
          "多个 Query 头共用一组 K/V 头，减少需要保存的历史数据，但仍保留多个提问视角。",
        explanation:
          "多个 Query 头共用一组 K/V 头，减少需要保存的历史数据，但仍保留多个提问视角。",
        details: ["多组问题可以共用同一份历史线索，减少需要保存的副本。"],
      },
      advanced: {
        summary: "Q 头按组映射到 K/V 头。",
        explanation:
          "Q 头按组映射到 K/V 头。Qwen3.8-27B 的全注意力配置为 Q=24、KV=4、head_dim=256；Qwen3.6-35B-A3B 为 Q=16、KV=2。",
        details: [
          "当 Q 头数为 nQ、KV 头数为 nKV，每组 KV 对应 nQ/nKV 个 Query；可视化按组展开。缓存字节依赖 nKV 而非 nQ。",
        ],
      },
      expert: {
        summary: "KV 显存按 KV头数计算，而不是 Q头数。",
        explanation:
          "KV 显存按 KV头数计算，而不是 Q头数。共享方案是模型训练结构的一部分，不能在已训练模型上任意改变而保持输出等价。",
        details: [
          "核对头分组、张量并行下 KV 头复制/切分及内核布局。选定 Qwen 的 head_dim 独立配置，不应使用 H/nQ 代替官方 head_dim。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/models",
      },
    ],
    sources: [
      {
        id: "2305.13245",
        title: "原始研究 · GQA",
        url: "https://arxiv.org/abs/2305.13245",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "gated-deltanet",
    title: "Gated DeltaNet：压缩历史为状态",
    englishTerms: ["Gated DeltaNet", "Linear Attention"],
    area: "模型计算",
    prerequisites: ["attention"],
    levels: {
      beginner: {
        summary: "并非每层都存下所有历史 K/V。",
        explanation:
          "并非每层都存下所有历史 K/V。有些层把历史逐步更新进一个状态，读取时使用这个压缩状态。",
        details: [
          "全注意力保存历史位置；线性层把历史压进一份固定大小的记忆，并逐步更新。",
        ],
      },
      advanced: {
        summary:
          "Gated DeltaNet 结合衰减门控与 Delta 更新，维护递归矩阵状态，并使用短卷积等局部状态。",
        explanation:
          "Gated DeltaNet 结合衰减门控与 Delta 更新，维护递归矩阵状态，并使用短卷积等局部状态。本网站两个 Qwen 配置每四层有三层此结构、一层全注意力。",
        details: [
          "先衰减旧矩阵，再按当前 Key 读取预测 Value，用预测误差作定向写入。短卷积还保存最近几个输入，两类状态生命周期不同。",
        ],
      },
      expert: {
        summary: "它不是标准 softmax Attention 的等价缓存压缩。",
        explanation:
          "它不是标准 softmax Attention 的等价缓存压缩。递归状态和卷积状态需单独管理；全注意力 KV 公式不能用于这些层，前缀复用也需要对应状态快照支持。",
        details: [
          "在 value×key 记法中 Sₜ=αₜSₜ₋₁(I−βₜkₜkₜᵀ)+βₜvₜkₜᵀ。服务端需同时管理卷积与递归状态，并区分 Prefill 批量算法与 Decode 单步更新；不能将状态等同于随 T 增长的 KV。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/models",
      },
    ],
    sources: [
      {
        id: "2412.06464",
        title: "原始研究 · Gated DeltaNet",
        url: "https://arxiv.org/abs/2412.06464",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "moe",
    title: "MoE：为每个 Token 选择专家",
    englishTerms: ["MoE", "Mixture of Experts", "Router"],
    area: "模型计算",
    prerequisites: ["ffn"],
    levels: {
      beginner: {
        summary: "专家像多组不同的 FFN。",
        explanation:
          "专家像多组不同的 FFN。路由器为当前 Token 选择一小部分专家，而不是运行所有专家。",
        details: ["每个 Token 找一小组专家处理，另有共享专家提供共同能力。"],
      },
      advanced: {
        summary:
          "路由 logits 经 top-k 选择与归一化，再把所选专家输出按权重合并。",
        explanation:
          "路由 logits 经 top-k 选择与归一化，再把所选专家输出按权重合并。Qwen3.6-35B-A3B 有 256 个路由专家，每 Token 选 8 个，并有 1 个共享专家。",
        details: [
          "路由 logits 选 Top-k；图中的激活权重在选中集合中归一化，再把各专家输出加权合并。共享专家独立计算，不计入 routed Top-k。",
        ],
      },
      expert: {
        summary: "激活参数少不代表模型驻留权重少，也不保证固定倍数加速。",
        explanation:
          "激活参数少不代表模型驻留权重少，也不保证固定倍数加速。专家并行需要派发与聚合；负载不均、All-to-All 通信和小矩阵效率会改变吞吐。",
        details: [
          "检查路由归一化、共享专家门控、专家并行通信和权重驻留。教学 Top-k 用通用 softmax，不冒充 Qwen 的完整路由实现；稀疏激活不等于所有未选权重可从显存删除。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/models",
      },
    ],
    sources: [
      {
        id: "2101.03961",
        title: "原始研究 · MoE",
        url: "https://arxiv.org/abs/2101.03961",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "prefill",
    title: "Prefill：一起理解整段输入",
    englishTerms: ["Prefill", "TTFT"],
    area: "推理基础",
    prerequisites: ["tokenization", "attention"],
    levels: {
      beginner: {
        summary: "模型先处理整段提示，为后续生成建立历史状态。",
        explanation:
          "模型先处理整段提示，为后续生成建立历史状态。第一个输出 Token 通常就从这一步最后位置的分布中选出。",
        details: ["模型先一起阅读整段提示词，准备历史状态并预测第一个输出。"],
      },
      advanced: {
        summary: "因果 mask 下各输入位置可以并行计算。",
        explanation:
          "因果 mask 下各输入位置可以并行计算。全注意力层写入输入 K/V，线性层更新递归状态；最后输入位置的 logits 用来采样首个生成 Token。",
        details: [
          "长度 T 的输入产生 T 个位置的表示及各层状态；通常用最后位置 logits 采样首 Token。之后每轮只输入一个新 Token，因此首 Token 不属于 Decode 新读入的输入。",
        ],
      },
      expert: {
        summary: "长提示通常带来较高算术强度，但具体瓶颈取决于矩阵尺寸和硬件。",
        explanation:
          "长提示通常带来较高算术强度，但具体瓶颈取决于矩阵尺寸和硬件。TTFT 还包含排队、分词和通信，不能把 Prefill 时间直接当作端到端 TTFT。",
        details: [
          "分析首 Token 延迟时区分队列、Tokenizer、Prefill 和输出处理。长 Prefill 分块涉及调度与缓存预留，不改变因果依赖；吞吐收益要在相同请求负载下比较。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/pipeline",
      },
    ],
    sources: [
      {
        id: "1706.03762",
        title: "原始研究 · Prefill",
        url: "https://arxiv.org/abs/1706.03762",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "decode",
    title: "Decode：逐步续写与停止",
    englishTerms: ["Decode", "Autoregressive Generation"],
    area: "推理基础",
    prerequisites: ["prefill", "kv-cache"],
    levels: {
      beginner: {
        summary: "每轮把上一次选出的 Token 喂回模型，再选择下一个 Token。",
        explanation:
          "每轮把上一次选出的 Token 喂回模型，再选择下一个 Token。生成是逐步进行的，不会一次算出整段回答。",
        details: ["每生成一个新片段，就把它作为下一步输入，继续使用历史记忆。"],
      },
      advanced: {
        summary:
          "每轮为新位置计算 Q/K/V，读取已有 KV 或递归状态，更新状态后从 logits 中选 Token。",
        explanation:
          "每轮为新位置计算 Q/K/V，读取已有 KV 或递归状态，更新状态后从 logits 中选 Token。达到 EOS、停止字符串或长度上限时结束。",
        details: [
          "每轮的新输入长度通常为 1，Full Attention 缓存追加该输入的 K/V；刚采样出的下一个 Token 尚未被读入。EOS 或长度上限触发停止。",
        ],
      },
      expert: {
        summary:
          "单请求 Decode 常受权重或 KV 读取带宽限制；批处理能改变算术强度。",
        explanation:
          "单请求 Decode 常受权重或 KV 读取带宽限制；批处理能改变算术强度。采样是模型前向之外的步骤；流式传输还涉及分词边界与网络缓冲。",
        details: [
          "Decode 的效率受 KV 读取、权重读取、批次规模与调度影响。诊断时区分缓存中的已处理 Token 与已输出 Token，不用输出字数推算缓存长度。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/pipeline",
      },
    ],
    sources: [
      {
        id: "1706.03762",
        title: "原始研究 · Decode",
        url: "https://arxiv.org/abs/1706.03762",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "sampling",
    title: "采样：从概率到下一个 Token",
    englishTerms: ["Sampling", "Temperature", "Top-k", "Top-p"],
    area: "推理基础",
    prerequisites: ["decode"],
    levels: {
      beginner: {
        summary: "模型给出候选分数，采样规则决定选择哪一个。",
        explanation:
          "模型给出候选分数，采样规则决定选择哪一个。温度越低，选择越集中；温度为零时直接取最大分数。",
        details: ["模型给候选片段打分，采样规则决定下一片段选谁。"],
      },
      advanced: {
        summary: "正温度使用 softmax(logits / T)。",
        explanation:
          "正温度使用 softmax(logits / T)。Top-k 保留最高 k 项，Top-p 按概率降序保留累计质量达到阈值的最小集合，再重新归一化。",
        details: [
          "先按温度缩放 logits，再进行 Top-k / Top-p 截断并重新归一化。Top-p 保留累积概率达到阈值所需的最少前缀；温度 0 使用 argmax。",
        ],
      },
      expert: {
        summary: "不同实现中多个处理器的顺序可能影响结果。",
        explanation:
          "不同实现中多个处理器的顺序可能影响结果。本网站先温度、再 top-k、再 top-p，并使用固定教学样本；温度为零处理为 argmax，避免除零。",
        details: [
          "采样运算顺序、随机数流、重复惩罚和停止条件会改变复现结果。本版使用固定分位序列，不报告随机输出多样性；低温时采用稳定 softmax，避免指数溢出。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/pipeline",
      },
    ],
    sources: [
      {
        id: "1904.09751",
        title:
          "The Curious Case of Neural Text Degeneration · Nucleus Sampling",
        url: "https://arxiv.org/abs/1904.09751",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "kv-cache",
    title: "KV Cache：复用历史计算",
    englishTerms: ["KV Cache", "Key Value Cache"],
    area: "显存与调度",
    prerequisites: ["attention", "prefill"],
    levels: {
      beginner: {
        summary: "旧 Token 的 K/V 可以留下来，下一轮只计算新 Token 的部分。",
        explanation:
          "旧 Token 的 K/V 可以留下来，下一轮只计算新 Token 的部分。缓存节省重复计算，却会随着上下文变长占用显存。",
        details: [
          "缓存保存过去已经算好的 K 和 V，生成下一片段时可以直接读取。",
        ],
      },
      advanced: {
        summary:
          "全注意力 KV 字节数 = 2 × 全注意力层数 × Token数 × batch × KV头数 × head_dim × 元素字节数。",
        explanation:
          "全注意力 KV 字节数 = 2 × 全注意力层数 × Token数 × batch × KV头数 × head_dim × 元素字节数。前面的 2 分别对应 K 和 V。",
        details: [
          "Full Attention KV 字节 = 2 × 层数 × B × T × KV头数 × head_dim × 每元素字节。将各轴与控件对应，观察长度或并发翻倍时结果如何线性增长。",
        ],
      },
      expert: {
        summary:
          "公式只估算有效 K/V 数据，未包含对齐、分配粒度、元数据和线性层状态。",
        explanation:
          "公式只估算有效 K/V 数据，未包含对齐、分配粒度、元数据和线性层状态。混合 Qwen 必须只计 full_attention 层；权重和激活也不在此分项内。",
        details: [
          "混合模型必须按层类型分项：线性层递归/卷积状态不沿 T 同样增长。FP8/INT4 缓存另含尺度、对齐与内核支持，预留块和实际 Token 占用也不是同一个量。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/lab/kv-cache",
      },
    ],
    sources: [
      {
        id: "2305.13245",
        title: "原始研究 · KV Cache",
        url: "https://arxiv.org/abs/2305.13245",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "paged-attention",
    title: "分页缓存：把逻辑位置映射到物理块",
    englishTerms: ["PagedAttention", "Block Table"],
    area: "显存与调度",
    prerequisites: ["kv-cache"],
    levels: {
      beginner: {
        summary: "把历史 Token 分成固定大小的小块，按需放进显存中的空闲位置。",
        explanation:
          "把历史 Token 分成固定大小的小块，按需放进显存中的空闲位置。逻辑顺序连续，物理位置可以分散。",
        details: ["每个请求看到连续的位置编号，背后却可以使用分散的物理块。"],
      },
      advanced: {
        summary:
          "块表把请求逻辑块索引映射到物理块 ID，长度 N 需要 ceil(N / blockSize) 个块。",
        explanation:
          "块表把请求逻辑块索引映射到物理块 ID，长度 N 需要 ceil(N / blockSize) 个块。释放请求后，物理块可返回池中复用。",
        details: [
          "请求 T 个 Token 需要 ceil(T/blockSize) 块；末块剩余空间为块数×blockSize−T。释放后物理块回到空闲池，逻辑块表把顺序映射到物理索引。",
        ],
      },
      expert: {
        summary: "分页减少预留和外部碎片，仍有尾块内部浪费。",
        explanation:
          "分页减少预留和外部碎片，仍有尾块内部浪费。内核必须支持块表寻址；引用计数、共享、写时复制与并发一致性属于真实系统额外职责。",
        details: [
          "核对预留、引用计数、驱逐与共享块的写入保护。资源检查必须发生在修改分配前；生产缓存还涉及滑窗、混合状态与可用预算，本版块池只演示分配不等同完整框架缓存。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/lab/paged-attention",
      },
    ],
    sources: [
      {
        id: "2309.06180",
        title: "原始研究 · PagedAttention",
        url: "https://arxiv.org/abs/2309.06180",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "prefix-caching",
    title: "前缀复用：共享已经处理过的输入",
    englishTerms: ["Prefix Caching", "Radix Cache"],
    area: "显存与调度",
    prerequisites: ["kv-cache", "tokenization"],
    levels: {
      beginner: {
        summary:
          "多个请求开头相同，可以复用此前已计算的状态，从共同前缀之后继续处理。",
        explanation:
          "多个请求开头相同，可以复用此前已计算的状态，从共同前缀之后继续处理。",
        details: ["多个请求有相同开头时，可以复用已经读过的那部分。"],
      },
      advanced: {
        summary: "本演示用 Token 前缀树显示匹配长度。",
        explanation:
          "本演示用 Token 前缀树显示匹配长度。vLLM 使用块哈希识别可复用完整块；SGLang 的 RadixCache 用压缩前缀树组织缓存键与位置。",
        details: [
          "前缀命中依赖完整 Token 序列、位置与计算条件。块哈希缓存通常按完整块匹配；压缩前缀树把连续共享路径合成节点，分叉后只复用共同部分。",
        ],
      },
      expert: {
        summary: "命中需要考虑模型、适配器、Token、缓存配置等身份条件。",
        explanation:
          "命中需要考虑模型、适配器、Token、缓存配置等身份条件。命中通常减少 Prefill 工作，而不是跳过新 Token 的 Decode；线性状态复用需相应实现支持。",
        details: [
          "检查缓存键是否包含模型、适配器等影响计算的身份，以及部分块、状态快照与释放策略。命中省去已有前缀的 Prefill 计算，但不消除新后缀的计算或所有后续 Decode 成本。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/lab/paged-attention",
      },
    ],
    sources: [
      {
        id: "2309.06180",
        title: "原始研究 · Prefix Caching",
        url: "https://arxiv.org/abs/2309.06180",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "batching",
    title: "连续批处理：让空出的槽位继续工作",
    englishTerms: ["Continuous Batching", "Iteration-level Scheduling"],
    area: "显存与调度",
    prerequisites: ["decode"],
    levels: {
      beginner: {
        summary:
          "一个请求结束后，把等待请求放进空位，无需等同一批的其他请求都完成。",
        explanation:
          "一个请求结束后，把等待请求放进空位，无需等同一批的其他请求都完成。长短请求就能更灵活地共用计算资源。",
        details: ["一个请求完成后，让等待中的请求尽快进入空出的计算槽位。"],
      },
      advanced: {
        summary: "调度器在迭代边界重组活动请求。",
        explanation:
          "调度器在迭代边界重组活动请求。本网站把每个 Decode 步设为一个模拟单位，对比静态批与连续批在相同请求到达和输出长度下的占用。",
        details: [
          "静态批等整批结束才补位；连续批在每个 Decode tick 检查空槽。比较同一请求到达与输出长度序列，分别记录等待时间、完成时间和槽位占用。",
        ],
      },
      expert: {
        summary:
          "实际调度受 Token预算、KV容量、Prefill/Decode混排与公平性限制。",
        explanation:
          "实际调度受 Token预算、KV容量、Prefill/Decode混排与公平性限制。较高利用率不自动代表更好的尾延迟；演示不把模拟单位标成 GPU 毫秒。",
        details: [
          "真实调度按 Token 预算、KV 容量与 Prefill/Decode 资源共同限制，并非只数槽位。网站时钟忽略 Prefill、抢占及通信，因此利用率只描述教学占槽，不代表 GPU 利用率。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/lab/continuous-batching",
      },
    ],
    sources: [
      {
        id: "orca-osdi22",
        title: "Orca · OSDI 2022 原始论文",
        url: "https://www.usenix.org/conference/osdi22/presentation/yu",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "quantization",
    title: "量化：用更少的位表达数值",
    englishTerms: ["Quantization", "INT4", "INT8", "GPTQ"],
    area: "推理优化",
    prerequisites: ["ffn"],
    levels: {
      beginner: {
        summary:
          "把连续数值映射到有限的整数格点，存得更少，但恢复出来可能有偏差。",
        explanation:
          "把连续数值映射到有限的整数格点，存得更少，但恢复出来可能有偏差。",
        details: [
          "用更少的刻度表示数字，可以节省存储，但原值不一定正好落在刻度上。",
        ],
      },
      advanced: {
        summary:
          "教学对称量化使用 scale=max|w|/(2^(b−1)−1)，q=round(w/scale)，反量化值为 q×scale。",
        explanation:
          "教学对称量化使用 scale=max|w|/(2^(b−1)−1)，q=round(w/scale)，反量化值为 q×scale。图中误差和存储字节由同一组权重计算。",
        details: [
          "对称量化令 s=max|w|/qmax，q=round(w/s) 并裁剪，反量化 ŵ=sq。INT4 用 ±7，INT8 用 ±127；均方误差衡量重建偏差。",
        ],
      },
      expert: {
        summary: "真实格式还包含 scale、分组、零点、打包和内核约束。",
        explanation:
          "真实格式还包含 scale、分组、零点、打包和内核约束。权重量化、激活量化与 KV量化是不同对象；减少位宽不能直接推出端到端加速倍数。",
        details: [
          "离群值会增大共同尺度，挤压小权重分辨率。生产方案常按组求尺度，并依赖校准与内核；小数组 MSE 和载荷比不能直接预测模型准确率、速度或完整显存。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/lab/quantization",
      },
    ],
    sources: [
      {
        id: "2210.17323",
        title: "原始研究 · Quantization",
        url: "https://arxiv.org/abs/2210.17323",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "speculation",
    title: "投机解码：先猜，再批量验证",
    englishTerms: ["Speculative Decoding", "MTP", "Draft"],
    area: "推理优化",
    prerequisites: ["decode", "sampling"],
    levels: {
      beginner: {
        summary: "便宜的草稿先猜几个 Token，目标模型一起检查。",
        explanation:
          "便宜的草稿先猜几个 Token，目标模型一起检查。猜对的前缀可以一起接受，第一次不匹配后需要修正。",
        details: ["小模型先猜几步，大模型一次检查；猜错时按大模型的结果修正。"],
      },
      advanced: {
        summary:
          "本演示是 greedy 验证：比较草稿与目标 argmax，接受首个错误前的连续前缀，丢弃剩余草稿并输出目标修正；全接受时目标可再补一个 Token。",
        explanation:
          "本演示是 greedy 验证：比较草稿与目标 argmax，接受首个错误前的连续前缀，丢弃剩余草稿并输出目标修正；全接受时目标可再补一个 Token。",
        details: [
          "本版限定贪心序列：从左到右接受匹配前缀，首个不匹配由目标修正并丢弃后续草稿；全部接受时可追加目标 bonus Token。输出始终等于目标序列。",
        ],
      },
      expert: {
        summary: "概率采样下的无损投机需要接受概率与残差分布校正，此处未实现。",
        explanation:
          "概率采样下的无损投机需要接受概率与残差分布校正，此处未实现。接受率、草稿成本和验证效率共同决定收益；MTP 与独立草稿模型也不是同一种执行机制。",
        details: [
          "随机采样的分布保持需要接受概率与残差分布，不能直接使用贪心匹配。成本要包括草稿、批量验证、缓存回滚与接受长度；教学验证每轮的固定成本不是 GPU 加速保证。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/lab/speculative-decoding",
      },
    ],
    sources: [
      {
        id: "2211.17192",
        title: "原始研究 · Speculative Decoding",
        url: "https://arxiv.org/abs/2211.17192",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "flash-attention",
    title: "FlashAttention：少搬运中间结果",
    englishTerms: ["FlashAttention", "IO-aware Attention"],
    area: "推理优化",
    prerequisites: ["attention"],
    levels: {
      beginner: {
        summary: "计算注意力时，减少把大块中间矩阵反复写入和读出显存。",
        explanation:
          "计算注意力时，减少把大块中间矩阵反复写入和读出显存。它仍计算精确注意力，而不是减少可见历史。",
        details: ["分块读取和计算注意力，尽量少把中间结果来回搬进显存。"],
      },
      advanced: {
        summary:
          "分块把 Q/K/V 送入片上存储，用 online softmax 维护归一化统计与输出累积，避免物化完整 N×N 分数与概率矩阵。",
        explanation:
          "分块把 Q/K/V 送入片上存储，用 online softmax 维护归一化统计与输出累积，避免物化完整 N×N 分数与概率矩阵。",
        details: [
          "内核分块计算 QK 与在线 softmax，再累计加权 V；保留每行最大值与归一化量，避免完整分数矩阵写回 HBM。因果 mask 仍参与每块计算。",
        ],
      },
      expert: {
        summary: "性能收益依赖序列长度、数据类型、硬件和具体内核。",
        explanation:
          "性能收益依赖序列长度、数据类型、硬件和具体内核。FlashAttention 与分页 KV解决不同问题，可在适配内核中组合；内存复杂度下降不等于 FLOPs同倍下降。",
        details: [
          "数学等价不表示浮点逐位相同；检查精度、head_dim、布局、mask 与硬件支持。Prefill 大矩阵与 Decode 单 Query 的瓶颈不同，不能把同一收益比例迁移到所有阶段。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/pipeline",
      },
    ],
    sources: [
      {
        id: "2205.14135",
        title: "原始研究 · FlashAttention",
        url: "https://arxiv.org/abs/2205.14135",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "parallelism",
    title: "并行策略：把模型与工作分到多卡",
    englishTerms: ["Tensor Parallel", "Pipeline Parallel", "Expert Parallel"],
    area: "推理优化",
    prerequisites: ["ffn", "moe"],
    levels: {
      beginner: {
        summary:
          "模型或工作放不下一张卡时，可以按矩阵、层或专家拆开，并让设备合作。",
        explanation:
          "模型或工作放不下一张卡时，可以按矩阵、层或专家拆开，并让设备合作。",
        details: [
          "模型可以让多张卡分担计算或保存不同的参数部分，但它们需要交换数据。",
        ],
      },
      advanced: {
        summary:
          "张量并行拆分线性层并通信聚合；流水并行把不同层放在不同设备；专家并行按专家归属派发 Token；数据并行运行独立副本服务不同请求。",
        explanation:
          "张量并行拆分线性层并通信聚合；流水并行把不同层放在不同设备；专家并行按专家归属派发 Token；数据并行运行独立副本服务不同请求。",
        details: [
          "Tensor Parallel 切分张量运算并通信；Pipeline Parallel 按层分段；Expert Parallel 分配专家并派发 Token。每种划分改变数据传输与批次组织。",
        ],
      },
      expert: {
        summary: "拆分会增加通信和同步。",
        explanation:
          "拆分会增加通信和同步。低延迟场景、网络拓扑、批大小、专家负载和 KV布局决定可行策略，不能简单用卡数乘出吞吐。本版只提供原理概览。",
        details: [
          "评估扩展时同时记录拓扑、通信字节、同步点和流水线空泡。KV 头数较小时可能复制而非均匀分片；权重分布与缓存分布需要分别确认。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/frameworks/vllm",
      },
    ],
    sources: [
      {
        id: "1909.08053",
        title: "原始研究 · Tensor Parallel",
        url: "https://arxiv.org/abs/1909.08053",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "chunked-prefill",
    title: "Chunked Prefill：把长输入分段调度",
    englishTerms: ["Chunked Prefill", "Token Budget"],
    area: "推理优化",
    prerequisites: ["prefill", "batching"],
    levels: {
      beginner: {
        summary: "长输入不必一次占住整个调度窗口。",
        explanation:
          "长输入不必一次占住整个调度窗口。分成块处理，可以在块之间为其他生成请求留出机会。",
        details: ["长提示词可以分成多段阅读，让正在生成的请求有机会穿插执行。"],
      },
      advanced: {
        summary:
          "调度器在每轮 Token预算下混合 Decode位置与部分 Prefill区间，处理完当前区间后保存进度与状态，下轮继续。",
        explanation:
          "调度器在每轮 Token预算下混合 Decode位置与部分 Prefill区间，处理完当前区间后保存进度与状态，下轮继续。",
        details: [
          "在总 Token 预算内分配 Prefill chunk 和 Decode 工作，下一 chunk 必须接续正确历史状态。分块改变调度粒度，不把提示词变成彼此独立的请求。",
        ],
      },
      expert: {
        summary:
          "块大小影响 TTFT、ITL与内核效率的权衡；共享前缀会改变剩余输入。",
        explanation:
          "块大小影响 TTFT、ITL与内核效率的权衡；共享前缀会改变剩余输入。分块是调度与执行的合作，不改变原始 Token顺序。此处提供概览并链接调度源码。",
        details: [
          "比较 TTFT 与 TPOT 的共同变化，并记录 chunk 大小、负载、抢占和缓存预留。过小 chunk 带来更多启动/调度开销，过大 chunk 可能增加 Decode 等待，需真实服务验证。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/frameworks/vllm",
      },
    ],
    sources: [
      {
        id: "2308.16369",
        title: "SARATHI · Chunked Prefill 原始研究",
        url: "https://arxiv.org/abs/2308.16369",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "cuda-graphs",
    title: "融合与 CUDA Graph：减少启动开销",
    englishTerms: ["CUDA Graph", "Operator Fusion"],
    area: "推理优化",
    prerequisites: ["decode"],
    levels: {
      beginner: {
        summary:
          "把连续的小操作合并，或录下一段固定执行流程再重放，可以减少每一步启动和组织计算的开销。",
        explanation:
          "把连续的小操作合并，或录下一段固定执行流程再重放，可以减少每一步启动和组织计算的开销。",
        details: [
          "把重复的 GPU 操作流程录下来，以后重放，减少每次启动的组织成本。",
        ],
      },
      advanced: {
        summary:
          "算子融合减少中间读写与内核调用；CUDA Graph捕获满足条件的执行图并复用，通常需要稳定的地址、控制流与适配的形状桶。",
        explanation:
          "算子融合减少中间读写与内核调用；CUDA Graph捕获满足条件的执行图并复用，通常需要稳定的地址、控制流与适配的形状桶。",
        details: [
          "捕获图要求稳定执行结构和可复用地址；动态批次可通过分桶、填充或部分图处理。图重放减少 CPU 发射开销，并不会自动减少模型的数学运算。",
        ],
      },
      expert: {
        summary: "动态批与动态长度可能需要填充、分桶或分段图；捕获也会占资源。",
        explanation:
          "动态批与动态长度可能需要填充、分桶或分段图；捕获也会占资源。编译、融合和 CUDA Graph不是同义词，收益需由特定硬件上的实测验证。本版展示源码入口。",
        details: [
          "核对捕获范围、缓冲区地址、动态图形与 graph pool 显存。编译/算子融合/图重放解决不同问题；模型、后端和版本变化可能改变支持边界。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/frameworks/vllm",
      },
    ],
    sources: [
      {
        id: "vllm-cuda-graphs",
        title: "vLLM 官方 CUDA Graph 设计说明",
        url: "https://docs.vllm.ai/en/stable/design/cuda_graphs/",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "metrics",
    title: "性能指标：把延迟与吞吐分开看",
    englishTerms: ["TTFT", "ITL", "Throughput", "Latency"],
    area: "工程与框架",
    prerequisites: ["prefill", "decode", "batching"],
    levels: {
      beginner: {
        summary: "首字等待、字间等待、整段时间和每秒输出量描述不同体验。",
        explanation:
          "首字等待、字间等待、整段时间和每秒输出量描述不同体验。吞吐更高，单个用户也可能等得更久。",
        details: [
          "第一片段等多久、后续片段隔多久、同时处理多少请求，是不同的问题。",
        ],
      },
      advanced: {
        summary:
          "TTFT从请求开始到首个输出；ITL描述相邻输出间隔；端到端延迟包含排队和传输。",
        explanation:
          "TTFT从请求开始到首个输出；ITL描述相邻输出间隔；端到端延迟包含排队和传输。吞吐需注明 input/output tokens、并发与统计窗口。",
        details: [
          "TTFT 衡量请求到首 Token 的时间；TPOT 描述后续输出间隔；吞吐按请求或 Token 每秒区分。统计延迟同时给出样本数、分位数与输入/输出长度分布。",
        ],
      },
      expert: {
        summary: "比较系统必须固定负载、输入输出长度、模型精度和 SLO。",
        explanation:
          "比较系统必须固定负载、输入输出长度、模型精度和 SLO。平均值隐藏尾延迟；p95/p99与稳态测量更适合评估服务。本网站不展示虚构硬件跑分。",
        details: [
          "明确客户端/服务器计时边界、排队时间、并发策略、预热和失败请求处理。吞吐最大值与单请求延迟最小值通常来自不同负载；没有硬件与负载条件的数字不可直接比较。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/lab/continuous-batching",
      },
    ],
    sources: [
      {
        id: "vllm-metrics",
        title: "vLLM 官方性能指标说明",
        url: "https://docs.vllm.ai/en/stable/usage/metrics/",
        checkedAt: "2026-10-04",
      },
    ],
  },
  {
    id: "frameworks",
    title: "推理框架：把计算组织成服务",
    englishTerms: ["vLLM", "SGLang", "Serving"],
    area: "工程与框架",
    prerequisites: ["batching", "paged-attention"],
    levels: {
      beginner: {
        summary:
          "框架负责把请求变成可执行的批次，管理缓存，运行模型，再把结果送回用户。",
        explanation:
          "框架负责把请求变成可执行的批次，管理缓存，运行模型，再把结果送回用户。",
        details: [
          "框架像一条流水线：接收请求、安排计算、管理记忆，再把文字送回。",
        ],
      },
      advanced: {
        summary:
          "HTTP入口、Token处理、调度器、缓存管理器、模型执行器与输出处理协作。",
        explanation:
          "HTTP入口、Token处理、调度器、缓存管理器、模型执行器与输出处理协作。图中直接调用、进程消息和共享数据用不同标签表示。",
        details: [
          "按请求入口 → Token 处理 → 调度与缓存 → 模型执行 → 输出回传追踪；区分函数调用、跨进程消息和共享数据，不能把所有连线当作直接调用栈。",
        ],
      },
      expert: {
        summary: "本网站固定 vLLM v0.30.0 与 SGLang v0.5.21 的源码提交。",
        explanation:
          "本网站固定 vLLM v0.30.0 与 SGLang v0.5.21 的源码提交。图是职责导览而非完整调用图；升级版本时应重新核对符号、缓存后端与模型注册。",
        details: [
          "固定 commit 后核对符号、注册、IPC 边界与取消/释放路径。源码职责图用于定位，实际模型支持、后端选择和部署表现仍需运行配置与测试佐证。",
        ],
      },
    },
    links: [
      {
        label: "打开对应可视化",
        to: "/frameworks/sglang",
      },
    ],
    sources: [
      {
        id: "2309.06180",
        title: "原始研究 · vLLM",
        url: "https://arxiv.org/abs/2309.06180",
        checkedAt: "2026-10-04",
      },
    ],
  },
];

for (const topic of topics)
  if (
    ["gqa", "gated-deltanet", "moe", "embedding", "ffn", "kv-cache"].includes(
      topic.id,
    )
  )
    topic.sources.push(...models.flatMap((m) => m.sources));
