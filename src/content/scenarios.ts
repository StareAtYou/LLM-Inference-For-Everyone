export const scenarios = [
  {
    id: "sky",
    name: "一句直观解释",
    prompt: "为什么天空是蓝色的？",
    tokens: ["为什么", "天空", "是", "蓝色", "的", "？"],
    ids: [101, 204, 37, 508, 12, 9],
    output: ["阳光", "经过", "空气", "发生", "散射", "。"],
  },
  {
    id: "code",
    name: "生成一行代码",
    prompt: "用 Python 打印 hello",
    tokens: ["用", "Python", "打印", "hello"],
    ids: [31, 512, 604, 823],
    output: ["print", "(", '"', "hello", '"', ")"],
  },
  {
    id: "shared",
    name: "带共享前缀的问答",
    prompt: "你是一位耐心的老师。解释 KV Cache",
    tokens: ["你是", "一位", "耐心的", "老师", "。", "解释", "KV", "Cache"],
    ids: [41, 21, 36, 61, 5, 94, 1001, 1002],
    output: ["缓存", "保存", "历史", "键值", "减少", "重算"],
  },
];
export const samplingCandidates = [
  "模型",
  "缓存",
  "计算",
  "Token",
  "状态",
  "。",
];
export const teachingLogits = [3.2, 2.4, 1.8, 1.1, 0.2, -0.7];
export const batchScenarios = [
  {
    id: "mixed",
    name: "长短输出混合",
    requests: [
      { id: "A", arrival: 0, outputTokens: 8 },
      { id: "B", arrival: 0, outputTokens: 2 },
      { id: "C", arrival: 1, outputTokens: 3 },
      { id: "D", arrival: 2, outputTokens: 6 },
      { id: "E", arrival: 3, outputTokens: 2 },
      { id: "F", arrival: 5, outputTokens: 3 },
    ],
  },
  {
    id: "equal",
    name: "同长输出同时到达",
    requests: ["A", "B", "C", "D", "E", "F"].map((id) => ({
      id,
      arrival: 0,
      outputTokens: 4,
    })),
  },
  {
    id: "staggered",
    name: "请求陆续到达",
    requests: ["A", "B", "C", "D", "E", "F"].map((id, i) => ({
      id,
      arrival: i * 2,
      outputTokens: i % 2 === 0 ? 3 : 7,
    })),
  },
];
export const targetTokens = [
  "模型",
  "先",
  "生成",
  "草稿",
  "然后",
  "验证",
  "接受",
  "正确",
  "前缀",
  "并",
  "继续",
  "。",
];
export const speculationScenarios = [
  {
    id: "partial",
    name: "部分接受",
    draft: [
      "模型",
      "先",
      "猜测",
      "草稿",
      "然后",
      "验证",
      "跳过",
      "正确",
      "前缀",
      "并",
      "继续",
      "。",
    ],
  },
  { id: "all", name: "草稿全部匹配", draft: [...targetTokens] },
  {
    id: "low",
    name: "低接受率",
    draft: targetTokens.map((_, i) => ["错误", "猜想", "偏离"][i % 3]),
  },
];
