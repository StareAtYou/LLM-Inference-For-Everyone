import type { Depth } from "../types";
type Guide = { beginner: string[]; advanced: string[]; expert: string[] };
export const labGuides: Record<string, Guide> = {
  "kv-cache": {
    beginner: ["历史 K/V 随已处理 Token 增加。先只改变长度，观察数值和图形。"],
    advanced: [
      "2 × Full Attention 层数 × batch × tokens × KV heads × head_dim × bytes，是当前显示的缓存分项公式。",
      "保持 batch 和精度不变，将 tokens 翻倍，缓存分项应翻倍。",
    ],
    expert: [
      "线性递归/卷积状态单独占用固定形状，本实验没有计入；也没有计入权重、激活、块对齐和元数据。",
      "检查的是逻辑字节，不是完整运行时峰值显存。混合缓存池还需结合框架实现验证。",
    ],
  },
  "paged-attention": {
    beginner: ["分配一个请求，再释放它，观察相同物理块如何重新被使用。"],
    advanced: [
      "逻辑块数 ceil(tokens/blockSize)，尾块浪费 = 块数×blockSize−tokens。",
      "前缀树在分叉前共享路径，Token 序列相等才具备复用条件。",
    ],
    expert: [
      "先检查资源再原子提交分配；OOM 后已有映射不应改变。",
      "真实块复用还需引用计数、驱逐和共享块写保护；前缀树为概念模拟，不计算线上命中率。",
    ],
  },
  "continuous-batching": {
    beginner: ["看哪个请求进入空槽、哪个完成；单步观察比一次播放更容易辨认。"],
    advanced: [
      "静态批在整批完成后补位；连续批每个 tick 填充空槽。",
      "两者共享到达时间、输出长度和 slot 容量，完成时间差来自补位规则。",
    ],
    expert: [
      "占槽利用率 = 已执行 Decode 单位 / (总 tick × 槽容量)。",
      "教学时钟不包含 Prefill、抢占、通信和 GPU 核心利用率，无法直接当真实服务 benchmark。",
    ],
  },
  quantization: {
    beginner: ["先点击一个权重，看原值落在哪个整数刻度，再切换 INT4 / INT8。"],
    advanced: [
      "scale=max|w|/qmax；整数码为 round(w/scale)，再反量化为 q×scale。",
      "误差图按所有权重的 MSE 比较；零权重用有限尺度约定并保持重建值 0。",
    ],
    expert: [
      "当前为单尺度对称权重量化，INT4 限制 ±7，不模拟所有生产编码方式。",
      "字节比只计载荷；尺度、分组、对齐、校准与内核都影响准确率和真实收益。",
    ],
  },
  "speculative-decoding": {
    beginner: ["观察绿色接受前缀与暖色修正点；拒绝后的草稿不会写入结果。"],
    advanced: [
      "首个不匹配后由目标补一个 Token；全接受时可能追加 bonus。",
      "每轮包括草稿成本与目标验证成本，接受长度越短越难摊薄额外成本。",
    ],
    expert: [
      "这是贪心匹配，不是随机采样的分布保持算法。",
      "真实实现需协调目标/草稿缓存的提交与回滚，并记录批量验证时间，不能使用教学 cost 推断硬件速度。",
    ],
  },
};
export function moduleGuide(
  module: {
    id: string;
    name: string;
    responsibility: string;
    inputs: string[];
    outputs: string[];
    snippet: { explanation: string };
  },
  depth: Depth,
): string[] {
  if (depth === "beginner")
    return [
      module.responsibility,
      "先在图中找到 " + module.name + "，再沿相邻连线理解请求去了哪里。",
    ];
  const core: Record<string, string> = {
    api: "确认协议校验、stream 与非 stream 分支，以及请求错误如何映射为 HTTP 响应。",
    client:
      "分清异步请求提交、响应迭代与 EngineCore 消息边界，沿 request ID 对应返回结果。",
    engine:
      "沿 schedule → execute → 更新输出追踪一次迭代，关注哪些工作由调度器决定。",
    scheduler:
      "检查 waiting/running 迁移、Token 预算与缓存约束，调度输出应和执行输入一一对应。",
    cache: "沿分配、前缀匹配与释放追踪缓存生命周期，区别逻辑位置与物理块。",
    blocks: "观察空闲队列与块引用的生命周期，已共享或已缓存的块不能随意覆盖。",
    runner:
      "确认输入准备、缓存索引、模型 forward 与输出处理，勿把 execute_model 等同于只有一个算子。",
    model: "先识别层类型，再比较全注意力与递归状态以及 Dense/MoE FFN 分支。",
    tokenizer:
      "追踪聊天模板、Token ID 和消息打包，明确输入长度与消息格式由谁确认。",
    detokenizer:
      "区分 Token 序列增量解码与文本发送，记录结束标记如何影响最终返回。",
  };
  return depth === "advanced"
    ? [
        `输入：${module.inputs.join("；")}。输出：${module.outputs.join("；")}。`,
        core[module.id],
        module.snippet.explanation,
      ]
    : [
        core[module.id],
        `在固定提交中追踪 ${module.name} 的取消、异常与释放路径，区分直接调用和消息传递；截取的几行不能证明完整路径都已执行。`,
        "核对实际张量布局、后端和资源所有权，再设计带版本、输入长度与并发条件的运行验证。此处只做源码阅读，不提供实测结果。",
      ];
}
