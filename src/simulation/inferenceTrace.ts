import type { ModelId } from "../types";
import { models } from "../content/models";
import { samplingCandidates, scenarios } from "../content/scenarios";
import { filterDistribution, softmax } from "./sampling";
export type TraceConfig = {
  modelId: ModelId;
  scenarioId: string;
  mode: "single" | "batch";
  capacity: number;
  temperature: number;
  topK: number;
  topP: number;
  outputLimit: number;
};
export type TraceStage =
  | "receive"
  | "tokenize"
  | "schedule"
  | "embedding"
  | "rmsnorm"
  | "qkv"
  | "rope"
  | "attention"
  | "linear"
  | "attention-output"
  | "residual"
  | "ffn"
  | "moe"
  | "final-norm"
  | "lm-head"
  | "sample"
  | "emit"
  | "feedback"
  | "finish"
  | "release";
export type TraceRequestState = {
  id: string;
  prompt: string;
  inputTokens: string[];
  inputIds: number[];
  outputTokens: string[];
  outputIds: number[];
  processedTokens: number;
  cacheTokens: number;
  blocks: number[];
  status: "waiting" | "prefill" | "decode" | "done";
  phase?: string;
};
export type TraceTensor = {
  name: string;
  shape: string;
  values: number[];
  rows: number;
  cols: number;
  requestId: string;
  note: string;
};
export type TraceFrame = {
  index: number;
  tick: number;
  stage: TraceStage;
  label: string;
  description: string;
  pass: "setup" | "prefill" | "decode" | "mixed" | "finish";
  layer: number | null;
  requestIds: string[];
  requests: TraceRequestState[];
  tensors: TraceTensor[];
  probabilities: number[];
  candidateLabels: string[];
  from: string;
  to: string;
  blockPart?: "attention" | "ffn";
};
export function buildInferenceTrace(config: TraceConfig): TraceFrame[] {
  const model = models.find((candidate) => candidate.id === config.modelId);
  const first = scenarios.findIndex(
    (scenario) => scenario.id === config.scenarioId,
  );
  if (
    !model ||
    first < 0 ||
    !["single", "batch"].includes(config.mode) ||
    !Number.isInteger(config.capacity) ||
    config.capacity < 1 ||
    config.capacity > 3 ||
    !Number.isInteger(config.outputLimit) ||
    config.outputLimit < 1 ||
    config.outputLimit > 4 ||
    !Number.isFinite(config.temperature) ||
    config.temperature < 0 ||
    !Number.isInteger(config.topK) ||
    config.topK < 1 ||
    config.topK > samplingCandidates.length ||
    !Number.isFinite(config.topP) ||
    config.topP <= 0 ||
    config.topP > 1
  ) {
    throw new Error(
      "需要有效模型、场景、单/批模式、1–3 槽位、1–4 输出及有限采样参数",
    );
  }
  const count = config.mode === "single" ? 1 : 3;
  const requests: RuntimeRequest[] = Array.from({ length: count }, (_, i) => {
    const scenario = scenarios[(first + i) % scenarios.length];
    return {
      arrival: i === 2 ? 1 : 0,
      limit: i === 1 ? Math.min(2, config.outputLimit) : config.outputLimit,
      view: {
        id: `R${i + 1}`,
        prompt: scenario.prompt,
        inputTokens: [...scenario.tokens],
        inputIds: [...scenario.ids],
        outputTokens: [],
        outputIds: [],
        processedTokens: 0,
        cacheTokens: 0,
        blocks: [],
        status: "waiting",
        phase: i === 2 ? "第 1 轮到达" : "等待入槽",
      },
      layers: model.layerTypes.map(() => ({
        keys: [],
        values: [],
        recurrent: [
          [
            [0, 0],
            [0, 0],
          ],
          [
            [0, 0],
            [0, 0],
          ],
        ],
        convolution: [],
      })),
    };
  });
  const frames: TraceFrame[] = [];
  // The physical block pool is shared, while logical mappings and all neural
  // states remain per request. These blocks hold full-attention KV only.
  const freeBlocks = Array.from({ length: 12 }, (_, i) => i);
  let tick = 0;
  let blockPart: "attention" | "ffn" = "attention";
  function emit(
    stage: TraceStage,
    label: string,
    description: string,
    group: RuntimeRequest[],
    pass: TraceFrame["pass"],
    layer: number | null,
    tensors: TraceTensor[] = [],
    from: TraceStage = stage,
    to: TraceStage = stage,
    probabilities: number[] = [],
  ) {
    frames.push(
      freeze({
        index: frames.length,
        tick,
        stage,
        label,
        description,
        pass,
        layer,
        blockPart:
          stage === "rmsnorm" || stage === "residual" ? blockPart : undefined,
        requestIds: group.map((r) => r.view.id),
        requests: requests.map(({ view }) => ({
          ...view,
          inputTokens: [...view.inputTokens],
          inputIds: [...view.inputIds],
          outputTokens: [...view.outputTokens],
          outputIds: [...view.outputIds],
          blocks: [...view.blocks],
        })),
        tensors,
        probabilities: [...probabilities],
        candidateLabels: [...samplingCandidates],
        from,
        to,
      }),
    );
  }
  function arrivals() {
    const group = requests.filter((r) => r.arrival === tick);
    if (!group.length) return;
    emit(
      "receive",
      "接收请求",
      "请求到达队列；逻辑 tick 是教学批次轮数，不是毫秒。",
      group,
      "setup",
      null,
      [],
      "receive",
      "tokenize",
    );
    emit(
      "tokenize",
      "Tokenization",
      "固定场景的教学 Token/ID 映射；不调用真实 Qwen 分词器。",
      group,
      "setup",
      null,
      group.map((r) =>
        tensor("输入 Token IDs", [r.view.inputIds], r.view.id, "固定分词示例"),
      ),
      "tokenize",
      "schedule",
    );
  }
  arrivals();
  while (requests.some((r) => r.view.status !== "done")) {
    if (tick > 0) arrivals();
    const active = requests.filter(
      (r) => r.view.status === "prefill" || r.view.status === "decode",
    );
    for (const r of requests) {
      if (active.length >= (config.mode === "single" ? 1 : config.capacity))
        break;
      if (r.view.status === "waiting" && r.arrival <= tick) {
        r.view.status = "prefill";
        active.push(r);
      }
    }
    if (!active.length) {
      tick++;
      continue;
    }
    const work = active.map((r) => {
      const prefill = r.view.processedTokens === 0;
      r.view.status = prefill ? "prefill" : "decode";
      r.view.phase = prefill
        ? "Prefill：整个提示"
        : "Decode：上轮生成的 1 个 Token";
      const ids = prefill ? r.view.inputIds : [r.view.outputIds.at(-1)!];
      return {
        r,
        ids,
        x: ids.map(embedding),
        normalized: [] as Matrix,
        q: [] as Matrix,
        k: [] as Matrix,
        v: [] as Matrix,
        context: [] as Matrix,
        branch: [] as Matrix,
      };
    });
    const pass = work.every((entry) => entry.r.view.status === "prefill")
      ? "prefill"
      : work.every((entry) => entry.r.view.status === "decode")
        ? "decode"
        : "mixed";
    emit(
      "schedule",
      "组成连续批次",
      "独立请求以 packed 方式处理，无 padding；同轮允许新请求 Prefill 与已有请求 Decode。数值按请求展示，不跨请求注意。",
      active,
      pass,
      null,
      [
        tensor(
          "批次 [输入长度, Prefill=1]",
          work.map(({ r, ids }) => [
            ids.length,
            r.view.status === "prefill" ? 1 : 0,
          ]),
          "batch",
          "每行对应 requestIds 顺序；不是 GPU 耗时",
        ),
      ],
      "schedule",
      "embedding",
    );
    emit(
      "embedding",
      "Embedding 查表",
      "Token ID 查找固定的 4 维教学向量；全部输入位置参与运算。",
      active,
      pass,
      null,
      work.map(({ r, x }) =>
        tensor(
          "Embedding",
          x,
          r.view.id,
          `真实隐藏维度 ${model.hiddenSize}；教学 H=4`,
        ),
      ),
      "embedding",
      "rmsnorm",
    );

    for (let layer = 0; layer < model.layerTypes.length; layer++) {
      blockPart = "attention";
      const full = model.layerTypes[layer] === "full_attention";
      const seed = layer + 1;
      for (const entry of work) entry.normalized = entry.x.map(norm);
      emit(
        "rmsnorm",
        "注意力前 RMSNorm",
        "逐 Token：x̂=x/√(mean(x²)+ε)，教学缩放参数 γ=1，ε=10⁻⁶。",
        active,
        pass,
        layer,
        work.flatMap(({ r, x, normalized }) => [
          tensor("输入 x", x, r.view.id, "残差流"),
          tensor("归一化 x̂", normalized, r.view.id, "逐行 RMSNorm"),
        ]),
        "rmsnorm",
        "qkv",
      );
      for (const entry of work) {
        entry.q = entry.normalized.map((x) =>
          project(x, weights(H, H, seed * 7)),
        );
        entry.k = entry.normalized.map((x) =>
          project(x, weights(H, 2, seed * 7 + 1)),
        );
        entry.v = entry.normalized.map((x) =>
          project(x, weights(H, 2, seed * 7 + 2)),
        );
      }
      emit(
        "qkv",
        "Q / K / V 投影",
        "使用各层固定矩阵真实相乘；教学 2 个 Q 头共用 1 个 K/V 头，每头 2 维。",
        active,
        pass,
        layer,
        work.flatMap(({ r, q, k, v }) => [
          tensor("Q", q, r.view.id, "2×2 查询头"),
          tensor("K", k, r.view.id, "1×2 键头"),
          tensor("V", v, r.view.id, "1×2 值头"),
        ]),
        "qkv",
        full ? "rope" : "linear",
      );
      if (full) {
        for (const entry of work) {
          entry.q = entry.q.map((q, i) =>
            rotate(q, entry.r.view.processedTokens + i),
          );
          entry.k = entry.k.map((k, i) =>
            rotate(k, entry.r.view.processedTokens + i),
          );
        }
        emit(
          "rope",
          "RoPE 位置旋转",
          "按该请求绝对位置旋转 Q/K；V 保持投影值。每个 2 维头的教学频率为 1。",
          active,
          pass,
          layer,
          work.flatMap(({ r, q, k }) => [
            tensor("旋转 Q", q, r.view.id, "绝对位置旋转"),
            tensor("旋转 K", k, r.view.id, "缓存存储旋转后的键"),
          ]),
          "rope",
          "attention",
        );
        const tensors: TraceTensor[] = [];
        for (const entry of work) {
          const state = entry.r.layers[layer];
          const result = fullAttention(entry.q, entry.k, entry.v, state);
          entry.context = result.output;
          tensors.push(
            tensor(
              "注意力分数",
              result.scores,
              entry.r.view.id,
              "末输入 Token；2 个头，QKᵀ/√2",
            ),
            tensor(
              "注意力概率",
              result.probabilities,
              entry.r.view.id,
              "每行 softmax；因果位置不读取未来 Token",
            ),
            tensor(
              "KV：K",
              state.keys,
              entry.r.view.id,
              "本请求、本层的旋转键；每个已处理位置一行",
            ),
            tensor(
              "KV：V",
              state.values,
              entry.r.view.id,
              "本请求、本层的值；无跨请求缓存",
            ),
          );
        }
        emit(
          "attention",
          "因果 Full Attention / GQA",
          "顺序追加各输入位置的 KV，2 个 Q 头读取同一组 K/V；softmax 权重乘 V。缓存计数在所有层完成后更新。",
          active,
          pass,
          layer,
          tensors,
          "attention",
          "attention-output",
        );
      } else {
        const tensors: TraceTensor[] = [];
        for (const entry of work) {
          const state = entry.r.layers[layer];
          const result = deltaAttention(
            entry.q,
            entry.k,
            entry.v,
            entry.normalized,
            state,
          );
          entry.context = result.output;
          tensors.push(
            tensor(
              "短卷积末步 QKV",
              [result.convolved],
              entry.r.view.id,
              "三抽头 0.7/0.2/0.1，因果历史仅 2 步",
            ),
            tensor(
              "卷积历史",
              state.convolution,
              entry.r.view.id,
              "保留末 2 个原始 QKV；固定大小，不是 KV Cache",
            ),
            tensor(
              "衰减 α / 写入 β",
              [[result.decay, result.beta]],
              entry.r.view.id,
              "sigmoid 门控",
            ),
            ...state.recurrent.map((matrix, head) =>
              tensor(
                `递归状态 S${head + 1}`,
                matrix,
                entry.r.view.id,
                "2×2 固定大小状态；不是逐 Token KV Cache",
              ),
            ),
          );
        }
        emit(
          "linear",
          "Gated DeltaNet",
          "小网络：S′=αS；S=S′+βk(v−kᵀS′)ᵀ；输出 gate·qᵀS。独立递归状态，线性层不创建 KV 列表。",
          active,
          pass,
          layer,
          tensors,
          "linear",
          "attention-output",
        );
      }
      for (const entry of work)
        entry.branch = entry.context.map((x) =>
          project(x, weights(H, H, seed * 7 + 3)),
        );
      emit(
        "attention-output",
        "注意力输出投影",
        "将两头的 4 维上下文拼接后乘 Wₒ，回到残差隐藏空间。",
        active,
        pass,
        layer,
        work.flatMap(({ r, context, branch }) => [
          tensor("上下文", context, r.view.id, "两头拼接"),
          tensor("输出投影", branch, r.view.id, "context × Wₒ"),
        ]),
        "attention-output",
        "residual",
      );
      for (const entry of work)
        entry.x = entry.x.map((x, i) => add(x, entry.branch[i]));
      emit(
        "residual",
        "注意力残差相加",
        "逐位置 x ← x + Attention(x)。",
        active,
        pass,
        layer,
        work.map(({ r, x }) =>
          tensor("残差输出", x, r.view.id, "注意力分支加法"),
        ),
        "residual",
        "rmsnorm",
      );
      blockPart = "ffn";
      for (const entry of work) entry.normalized = entry.x.map(norm);
      emit(
        "rmsnorm",
        "前馈前 RMSNorm",
        "第二次 RMSNorm 为 Dense FFN 或 MoE 提供归一化输入。",
        active,
        pass,
        layer,
        work.flatMap(({ r, x, normalized }) => [
          tensor("输入 x", x, r.view.id, "注意力残差"),
          tensor("归一化 x̂", normalized, r.view.id, "逐行 RMSNorm"),
        ]),
        "rmsnorm",
        model.family === "moe" ? "moe" : "ffn",
      );
      const ffnTensors: TraceTensor[] = [];
      for (const entry of work) {
        if (model.family === "moe") {
          let routing: Vector = [];
          let routed: Vector = [];
          let shared: Vector = [];
          let selected: Matrix = [];
          let expertOutputs: Matrix = [];
          entry.branch = entry.normalized.map((x) => {
            routing = softmax(project(x, weights(H, 3, seed * 11)), 1);
            const chosen = routing
              .map((p, i) => ({ p, i }))
              .sort((a, b) => b.p - a.p || a.i - b.i)
              .slice(0, 2);
            const mass = chosen.reduce((sum, expert) => sum + expert.p, 0);
            const outputs = chosen.map((expert) => ({
              weight: expert.p / mass,
              output: ffn(x, seed * 11 + expert.i * 3 + 1).output,
            }));
            selected = chosen.map((expert) => [expert.i, expert.p / mass]);
            expertOutputs = outputs.map((expert) => expert.output);
            routed = x.map((_, j) =>
              outputs.reduce(
                (sum, expert) => sum + expert.weight * expert.output[j],
                0,
              ),
            );
            shared = ffn(x, seed * 11 + 10).output.map(
              (value) => value * sigmoid(x[0]),
            );
            return add(routed, shared);
          });
          ffnTensors.push(
            tensor(
              "路由概率",
              [routing],
              entry.r.view.id,
              "教学 3 个专家选 2，选中权重归一；真实专家数量另列",
            ),
            tensor(
              "选中专家 [ID, 权重]",
              selected,
              entry.r.view.id,
              "末输入 Token；教学专家 ID 为 0/1/2，归一权重之和为 1",
            ),
            tensor(
              "选中专家输出",
              expertOutputs,
              entry.r.view.id,
              "每行对应选中专家；先计算门控 FFN 再求加权和",
            ),
            tensor(
              "路由加权和",
              [routed],
              entry.r.view.id,
              "Σ wᵢ Expertᵢ(x)，末输入 Token",
            ),
            tensor(
              "共享专家",
              [shared],
              entry.r.view.id,
              "独立共享 FFN × sigmoid 门控，末输入 Token",
            ),
            tensor(
              "MoE 输出",
              entry.branch,
              entry.r.view.id,
              "路由专家加权和 + 共享专家",
            ),
          );
        } else {
          const results = entry.normalized.map((x) => ffn(x, seed * 11));
          entry.branch = results.map((result) => result.output);
          ffnTensors.push(
            tensor(
              "SiLU 门控",
              results.map((result) => result.gate),
              entry.r.view.id,
              "SiLU(xWgate)；教学中间维度 6",
            ),
            tensor(
              "上投影",
              results.map((result) => result.up),
              entry.r.view.id,
              "xWup",
            ),
            tensor(
              "门控乘积",
              results.map((result) => result.product),
              entry.r.view.id,
              "SiLU(xWgate) ⊙ xWup",
            ),
            tensor(
              "FFN 输出",
              entry.branch,
              entry.r.view.id,
              "门控乘积 × Wdown",
            ),
          );
        }
      }
      const ffnStage = model.family === "moe" ? "moe" : "ffn";
      emit(
        ffnStage,
        model.family === "moe" ? "MoE 路由与专家" : "门控 Dense FFN",
        model.family === "moe"
          ? "每个 Token 计算路由，选择 2/3 教学专家并求加权和，再加共享专家。真实拓扑的专家参数另列。"
          : "真实小数组计算 (SiLU(xWgate) ⊙ xWup)Wdown；不使用真实 Qwen 权重。",
        active,
        pass,
        layer,
        ffnTensors,
        ffnStage,
        "residual",
      );
      for (const entry of work)
        entry.x = entry.x.map((x, i) => add(x, entry.branch[i]));
      emit(
        "residual",
        "前馈残差相加",
        "逐位置 x ← x + FFN(x)，进入下一层或 Final Norm。",
        active,
        pass,
        layer,
        work.map(({ r, x }) =>
          tensor(
            "层输出",
            x,
            r.view.id,
            `第 ${layer + 1}/${model.layerTypes.length} 层`,
          ),
        ),
        "residual",
        layer === model.layerTypes.length - 1 ? "final-norm" : "rmsnorm",
      );
    }

    for (const entry of work) {
      const r = entry.r;
      const requestPass = r.view.status === "prefill" ? "prefill" : "decode";
      r.view.processedTokens += entry.ids.length;
      r.view.cacheTokens = r.view.processedTokens;
      while (r.view.blocks.length < Math.ceil(r.view.cacheTokens / BLOCK_SIZE))
        r.view.blocks.push(freeBlocks.shift()!);
      const hidden = norm(entry.x.at(-1)!);
      emit(
        "final-norm",
        "Final RMSNorm",
        "所有层已处理本轮输入；KV 页按每页 4 个位置分配，线性层只保留固定递归/卷积状态。使用末位置隐藏向量预测下一 Token。",
        [r],
        requestPass,
        null,
        [
          tensor(
            "末位置隐藏向量",
            [entry.x.at(-1)!],
            r.view.id,
            "仅末位置用于下一 Token",
          ),
          tensor("Final Norm", [hidden], r.view.id, "H=4；γ=1"),
        ],
        "final-norm",
        "lm-head",
      );
      const logits = project(hidden, weights(H, samplingCandidates.length, 91));
      emit(
        "lm-head",
        "LM Head",
        "归一化隐藏向量乘词表投影矩阵，得到 6 个教学候选词的 logits。",
        [r],
        requestPass,
        null,
        [
          tensor("隐藏输入", [hidden], r.view.id, "Final Norm 输出"),
          tensor(
            "Logits",
            [logits],
            r.view.id,
            "6 词教学词表；不是场景预写输出",
          ),
        ],
        "lm-head",
        "sample",
      );
      const raw = softmax(logits, config.temperature);
      const probabilities = filterDistribution(raw, config.topK, config.topP);
      // The same quantile for every request makes solo/batch executions agree.
      const quantile = 0.61;
      let cumulative = 0;
      let chosen = probabilities.length - 1;
      for (let i = 0; i < probabilities.length; i++) {
        cumulative += probabilities[i];
        if (quantile < cumulative) {
          chosen = i;
          break;
        }
      }
      emit(
        "sample",
        "Softmax 与采样",
        `温度 ${config.temperature} → Top-K ${config.topK} → Top-P ${config.topP}；固定分位数 0.61 从实际分布取样。新 Token 此时尚未进入模型或缓存。`,
        [r],
        requestPass,
        null,
        [
          tensor(
            "Softmax",
            [raw],
            r.view.id,
            "稳定 softmax；温度为 0 时 argmax",
          ),
          tensor(
            "过滤概率",
            [probabilities],
            r.view.id,
            "Top-K/Top-P 后重新归一化",
          ),
        ],
        "sample",
        "emit",
        probabilities,
      );
      r.view.outputTokens.push(samplingCandidates[chosen]);
      r.view.outputIds.push(VOCAB_IDS[chosen]);
      emit(
        "emit",
        "流式输出 Token",
        `输出「${samplingCandidates[chosen]}」；用户可见输出 +1，已处理输入与 KV 数量不变。`,
        [r],
        requestPass,
        null,
        [
          tensor(
            "新 Token ID",
            [[VOCAB_IDS[chosen]]],
            r.view.id,
            "教学输出 ID",
          ),
        ],
        "emit",
        r.view.outputTokens.length >= r.limit ? "finish" : "feedback",
        probabilities,
      );
      if (r.view.outputTokens.length >= r.limit) {
        r.view.status = "done";
        r.view.phase = `达到输出上限 ${r.limit}`;
        emit(
          "finish",
          "达到输出上限",
          "明确停止条件：教学输出数量上限；没有隐含 EOS。最后生成的 Token 不再进行 Decode。",
          [r],
          "finish",
          null,
          [],
          "finish",
          "release",
        );
        freeBlocks.push(...r.view.blocks);
        freeBlocks.sort((a, b) => a - b);
        r.view.blocks = [];
        r.view.cacheTokens = 0;
        r.layers = [];
        emit(
          "release",
          "释放缓存与槽位",
          "释放全部物理 KV 页与线性递归/卷积状态；保留提示、已处理数量与用户输出。下一轮等待请求可补位。",
          [r],
          "finish",
          null,
          [],
          "release",
          "schedule",
        );
      } else {
        r.view.status = "decode";
        r.view.phase = "等待将新 Token 反馈到下一轮";
        emit(
          "feedback",
          "反馈到下一轮",
          "上轮输出 ID 在下一次 Decode 才做 Embedding、逐层计算并进入 KV/递归状态。",
          [r],
          "decode",
          null,
          [
            tensor(
              "待反馈 ID",
              [[VOCAB_IDS[chosen]]],
              r.view.id,
              "尚未处理或写缓存",
            ),
          ],
          "feedback",
          "embedding",
        );
      }
    }
    tick++;
  }
  return frames;
}

// Real model topology; all values below belong to an explicitly miniature,
// deterministic network (H=4, two query heads, one KV head, head dimension=2).
// These fixture matrices are not Qwen weights or a tokenizer implementation.
const H = 4;
const EPSILON = 1e-6;
const BLOCK_SIZE = 4;
const VOCAB_IDS = [2001, 2002, 2003, 2004, 2005, 2006];
type Vector = number[];
type Matrix = Vector[];
type LayerState = {
  keys: Matrix;
  values: Matrix;
  recurrent: Matrix[];
  convolution: Matrix;
};
type RuntimeRequest = {
  view: TraceRequestState;
  arrival: number;
  limit: number;
  layers: LayerState[];
};
const dot = (a: Vector, b: Vector) =>
  a.reduce((sum, value, i) => sum + value * b[i], 0);
const add = (a: Vector, b: Vector) => a.map((value, i) => value + b[i]);
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const silu = (x: number) => x * sigmoid(x);
function norm(x: Vector): Vector {
  const rms = Math.sqrt(dot(x, x) / x.length + EPSILON);
  return x.map((value) => value / rms);
}
function unit(x: Vector): Vector {
  const length = Math.sqrt(dot(x, x) + EPSILON);
  return x.map((value) => value / length);
}
function weights(rows: number, cols: number, seed: number): Matrix {
  return Array.from({ length: rows }, (_, i) =>
    Array.from(
      { length: cols },
      (_, j) =>
        Math.sin((i + 1) * 1.7 + (j + 1) * 2.3 + seed * 0.71) * 0.18 +
        (i === j ? 0.23 : 0),
    ),
  );
}
function project(x: Vector, matrix: Matrix): Vector {
  return matrix[0].map((_, j) =>
    x.reduce((sum, value, i) => sum + value * matrix[i][j], 0),
  );
}
function embedding(id: number): Vector {
  return Array.from(
    { length: H },
    (_, j) =>
      Math.sin(id * 0.017 + j * 1.3) * 0.7 +
      Math.cos(id * 0.011 - j * 0.8) * 0.3,
  );
}
function rotate(x: Vector, position: number): Vector {
  const result = [...x];
  // Every miniature head contains one 2D pair; its RoPE frequency is 1.
  for (let i = 0; i < x.length; i += 2) {
    result[i] = x[i] * Math.cos(position) - x[i + 1] * Math.sin(position);
    result[i + 1] = x[i] * Math.sin(position) + x[i + 1] * Math.cos(position);
  }
  return result;
}
function freeze<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}
function tensor(
  name: string,
  matrix: Matrix,
  requestId: string,
  note: string,
): TraceTensor {
  // A panel displays at most 24 actual values. Every token still participates in
  // computation; only long matrix displays retain their last rows.
  const cols = matrix[0].length;
  const shown = matrix.slice(-Math.max(1, Math.floor(24 / cols)));
  const cropped =
    shown.length < matrix.length
      ? `；显示末 ${shown.length}/${matrix.length} 行`
      : "";
  return {
    name,
    rows: shown.length,
    cols,
    shape: `${shown.length} × ${cols}`,
    values: shown.flat(),
    requestId,
    note: `小维度教学数值；${note}${cropped}`,
  };
}
function ffn(
  x: Vector,
  seed: number,
): { output: Vector; gate: Vector; up: Vector; product: Vector } {
  const gate = project(x, weights(H, 6, seed)).map(silu);
  const up = project(x, weights(H, 6, seed + 1));
  const product = gate.map((value, i) => value * up[i]);
  return {
    output: project(product, weights(6, H, seed + 2)),
    gate,
    up,
    product,
  };
}
function fullAttention(q: Matrix, k: Matrix, v: Matrix, state: LayerState) {
  const output: Matrix = [];
  let lastScores: Matrix = [];
  let lastProbabilities: Matrix = [];
  for (let token = 0; token < q.length; token++) {
    state.keys.push(k[token]);
    state.values.push(v[token]);
    const scores = [0, 1].map((head) =>
      state.keys.map(
        (key) =>
          dot(q[token].slice(head * 2, head * 2 + 2), key) / Math.sqrt(2),
      ),
    );
    const probabilities = scores.map((row) => softmax(row, 1));
    output.push(
      probabilities.flatMap((row) =>
        [0, 1].map((j) =>
          row.reduce(
            (sum, probability, i) => sum + probability * state.values[i][j],
            0,
          ),
        ),
      ),
    );
    lastScores = scores;
    lastProbabilities = probabilities;
  }
  return { output, scores: lastScores, probabilities: lastProbabilities };
}
function deltaAttention(
  q: Matrix,
  k: Matrix,
  v: Matrix,
  normalized: Matrix,
  state: LayerState,
) {
  const output: Matrix = [];
  let convolved: Vector = [];
  let decay = 0;
  let beta = 0;
  for (let token = 0; token < q.length; token++) {
    // A causal three-tap depthwise short convolution precedes delta state.
    const packed = [...q[token], ...k[token], ...v[token]];
    convolved = packed.map(
      (value, j) =>
        value * 0.7 +
        (state.convolution.at(-1)?.[j] ?? 0) * 0.2 +
        (state.convolution.at(-2)?.[j] ?? 0) * 0.1,
    );
    state.convolution.push(packed);
    state.convolution = state.convolution.slice(-2);
    const key = unit(convolved.slice(4, 6));
    const value = convolved.slice(6, 8);
    decay = sigmoid(normalized[token][0]);
    beta = sigmoid(normalized[token][1]);
    const heads = [0, 1].flatMap((head) => {
      const query = unit(convolved.slice(head * 2, head * 2 + 2));
      const decayed = state.recurrent[head].map((row) =>
        row.map((cell) => cell * decay),
      );
      const prediction = [0, 1].map((j) =>
        key.reduce((sum, cell, i) => sum + cell * decayed[i][j], 0),
      );
      const updated = decayed.map((row, i) =>
        row.map((cell, j) => cell + beta * key[i] * (value[j] - prediction[j])),
      );
      state.recurrent[head] = updated;
      const gate = sigmoid(normalized[token][head + 2]);
      return [0, 1].map(
        (j) =>
          query.reduce((sum, cell, i) => sum + cell * updated[i][j], 0) * gate,
      );
    });
    output.push(heads);
  }
  return { output, convolved, decay, beta };
}
