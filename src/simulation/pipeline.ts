import type { ModelId, Observation } from "../types";
import { getModel } from "../content/models";
import { scenarios } from "../content/scenarios";
import { softmax, filterDistribution } from "./sampling";
export type PipelineFrame = {
  stage: string;
  inputTokens: string[];
  outputTokens: string[];
  cacheTokens: number;
  selectedLayer?: number;
  description: string;
  observations: Observation[];
};
export function buildPipelineFrames(
  modelId: ModelId,
  scenarioId: string,
  sampling: { temperature: number; topK: number; topP: number },
): PipelineFrame[] {
  const model = getModel(modelId);
  if (!model) throw new Error("模型不存在");
  const scenario = scenarios.find((s) => s.id === scenarioId);
  if (!scenario) throw new Error("场景不存在");
  const n = scenario.tokens.length;
  const base = {
    inputTokens: scenario.tokens,
    outputTokens: [] as string[],
    cacheTokens: 0,
    observations: [] as Observation[],
  };
  const frames: PipelineFrame[] = [
    { ...base, stage: "输入", description: "读取固定教学提示，尚未运行模型。" },
    {
      ...base,
      stage: "Tokenization",
      description:
        "教学 Token 与 ID 用来展示编码流程，不是官方分词器的实际输出。",
    },
    {
      ...base,
      stage: "Embedding",
      description: `每个 ID 查表得到 ${model.hiddenSize} 维表示，送入 ${model.layerTypes.length} 层混合网络。`,
    },
  ];
  const outputs: string[] = [];
  for (let i = 0; i < scenario.output.length; i++) {
    const probs = filterDistribution(
      softmax([3.2, 2.4, 1.8, 1.1, 0.2, -0.7], sampling.temperature),
      sampling.topK,
      sampling.topP,
    );
    let mass = 0;
    const u = [0.31, 0.81, 0.16, 0.54, 0.92, 0.25][i];
    let chosen = 0;
    for (let j = 0; j < probs.length; j++) {
      mass += probs[j];
      if (u < mass) {
        chosen = j;
        break;
      }
    }
    outputs.push(
      chosen === 0
        ? scenario.output[i]
        : ["思考", "继续", "说明", "细节", "例子"][chosen - 1],
    );
    const cacheTokens = n + i;
    frames.push({
      inputTokens: scenario.tokens,
      outputTokens: [...outputs],
      cacheTokens,
      stage: i === 0 ? "Prefill" : "Decode",
      selectedLayer: model.layerTypes.length - 1,
      description:
        i === 0
          ? "并行处理输入，保存状态；最后输入位置的 logits 产生首个输出 Token。"
          : "将上一个输出 Token 喂回模型，读取历史状态，选择下一个 Token。",
      observations: [
        {
          label: "已进入全注意力缓存的位置",
          value: cacheTokens,
          unit: "tokens",
          source: "simulation",
          assumptions: ["Token数为教学序列；线性层使用独立递归状态"],
        },
      ],
    });
  }
  frames.push({
    ...frames.at(-1)!,
    stage: "停止",
    description: "达到教学输出长度上限。真实服务还会处理 EOS 与停止字符串。",
  });
  return frames;
}
