import type { ModelPreset } from "../types";
import type { TraceFrame, TraceStage } from "../simulation/inferenceTrace";
import {
  buildMechanismFrames,
  mechanisms,
  type MechanismFrame,
  type MechanismPanel,
} from "../mechanisms/catalog";
import denseConfig from "../../docs/references/models/Qwen3.8-27B.config.json";
import moeConfig from "../../docs/references/models/Qwen3.6-35B-A3B.config.json";

export type ShapeObject = {
  name: string;
  kind: "tensor" | "structure";
  axes: string[];
  dimensions: (number | string)[];
  count?: number;
  detail?: string;
};
export type ShapeStep = {
  title: string;
  inputs: ShapeObject[];
  outputs: ShapeObject[];
  formula: string;
  explanation: string;
  substitution: string;
};
export type ShapeLesson = {
  stage: string;
  scope: string;
  axes: { symbol: string; meaning: string; value?: number | string }[];
  steps: ShapeStep[];
  notes: string[];
  currentObjects: ShapeObject[];
  queryLength: number;
  keyLength: number;
  packedTokens: number;
  batchRequests: { id: string; queryLength: number; historyLength: number }[];
  sources?: { title: string; url: string }[];
};
const tensor = (
  name: string,
  axes: string[],
  dimensions: (number | string)[],
  detail?: string,
): ShapeObject => ({ name, axes, dimensions, kind: "tensor", detail });
const structure = (
  name: string,
  count?: number,
  detail?: string,
): ShapeObject => ({
  name,
  kind: "structure",
  axes: [],
  dimensions: [],
  count,
  detail,
});
const step = (
  title: string,
  inputs: ShapeObject[],
  outputs: ShapeObject[],
  formula: string,
  explanation: string,
  substitution: string,
): ShapeStep => ({
  title,
  inputs,
  outputs,
  formula,
  explanation,
  substitution,
});
export const formatShape = (object: ShapeObject) =>
  object.kind === "structure"
    ? `${object.count === undefined ? "结构对象" : `${object.count} 项`}`
    : `[${object.dimensions.join(", ")}]`;
const implementationSource = {
  title: "Transformers · Qwen3.5 系列实现（核查 2026-10-05）",
  url: "https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5",
};

/** Shapes are logical text-backbone dimensions, not allocated GPU buffers. */
export function modelShapeSteps(
  stage: TraceStage,
  model: ModelPreset,
  q: number,
  t: number,
  linear: boolean,
): ShapeStep[] {
  const H = model.hiddenSize,
    nQ = model.qHeads,
    nKV = model.kvHeads,
    d = model.headDim;
  const official = model.id !== "teaching";
  const config =
    model.id === "qwen36-moe" ? moeConfig.text_config : denseConfig.text_config;
  const V = official ? config.vocab_size : 6,
    I = model.ffnSize ?? H;
  const gated = official && config.attn_output_gate;
  const hK = model.linearKeyHeads,
    hV = model.linearValueHeads,
    dk = model.linearHeadDim,
    dv = model.linearHeadDim;
  const keyWidth = hK * dk,
    valueWidth = hV * dv,
    convWidth = 2 * keyWidth + valueWidth;
  const x = tensor("X", ["b", "q", "H"], [1, q, H]);
  const hidden = tensor("Y", ["b", "q", "H"], [1, q, H]);
  const Q = tensor("Q", ["b", "nQ", "q", "d"], [1, nQ, q, d]);
  const Knew = tensor("K 新位置", ["b", "nKV", "q", "d"], [1, nKV, q, d]);
  const Vnew = tensor("V 新位置", ["b", "nKV", "q", "d"], [1, nKV, q, d]);
  const Kcache = tensor("K 缓存", ["b", "nKV", "t", "d"], [1, nKV, t, d]);
  const Vcache = tensor("V 缓存", ["b", "nKV", "t", "d"], [1, nKV, t, d]);
  const scores = tensor("分数 / 概率", ["b", "nQ", "q", "t"], [1, nQ, q, t]);
  const context = tensor("头输出", ["b", "nQ", "q", "d"], [1, nQ, q, d]);
  const outputWidth = linear ? valueWidth : nQ * d;
  switch (stage) {
    case "embedding":
      return [
        step(
          "整数索引查表",
          [
            tensor("Token IDs", ["b", "q"], [1, q]),
            tensor("Embedding E", ["V", "H"], [V, H]),
          ],
          [x],
          "X[b,i,:] = E[ID[b,i],:]",
          "ID 选择词表行，保留全部 H 个通道。查表没有对词表轴做矩阵乘归约。",
          `[1, ${q}] 个 ID → 每个取 ${H} 个通道 → [1, ${q}, ${H}]`,
        ),
      ];
    case "rmsnorm":
    case "final-norm": {
      const rows = stage === "final-norm" ? 1 : q;
      return [
        step(
          "沿隐藏轴求均方根",
          [tensor("x", ["b", "q", "H"], [1, rows, H]), tensor("γ", ["H"], [H])],
          [
            tensor("RMS 分母", ["b", "q", "1"], [1, rows, 1]),
            tensor("归一化输出", ["b", "q", "H"], [1, rows, H]),
          ],
          "r = √(Σₕ xₕ²/H + ε); yₕ = xₕ/r · γₕ",
          "只对 H 轴归约，分母保留单例轴并广播回通道；不混合 Token 位置。Qwen 实现的有效缩放为 1+weight。",
          `每行 ${H} 项平方 → 1 个分母；[1, ${rows}, 1] 广播到 [1, ${rows}, ${H}]`,
        ),
      ];
    }
    case "qkv":
      if (!linear) {
        const width = nQ * d * (gated ? 2 : 1);
        return [
          step(
            "投影：输入通道收缩",
            [
              x,
              tensor(
                gated ? "W_Q+gate" : "W_Q",
                ["H", gated ? "2·nQ·d" : "nQ·d"],
                [H, width],
              ),
              tensor("W_K / W_V", ["H", "nKV·d"], [H, nKV * d]),
            ],
            [
              tensor(
                gated ? "Q 与输出门（未拆头）" : "Q（未拆头）",
                ["b", "q", gated ? "2·nQ·d" : "nQ·d"],
                [1, q, width],
              ),
              tensor("K（未拆头）", ["b", "q", "nKV·d"], [1, q, nKV * d]),
              tensor("V（未拆头）", ["b", "q", "nKV·d"], [1, q, nKV * d]),
            ],
            "Z[b,i,j] = Σₕ X[b,i,h] W[h,j]",
            "收缩共享 H 轴；q 轴保留。数学权重写 [输入,输出]，PyTorch Linear 存储布局是它的转置。",
            `[1, ${q}, ${H}] × [${H}, ${width}] → [1, ${q}, ${width}]`,
          ),
          step(
            "拆开门与头，再换轴",
            [
              tensor(
                gated ? "Q 与输出门（未拆头）" : "Q（未拆头）",
                ["b", "q", gated ? "2·nQ·d" : "nQ·d"],
                [1, q, width],
              ),
              tensor("K（未拆头）", ["b", "q", "nKV·d"], [1, q, nKV * d]),
              tensor("V（未拆头）", ["b", "q", "nKV·d"], [1, q, nKV * d]),
            ],
            [
              Q,
              Knew,
              Vnew,
              ...(gated
                ? [tensor("Full 输出门", ["b", "q", "nQ·d"], [1, q, nQ * d])]
                : []),
            ],
            "Q+gate → split → reshape(b,q,nQ,d) → transpose(q,nQ); K/V → reshape(b,q,nKV,d) → transpose(q,nKV)",
            "门与 Q 各占 nQ·d 通道，K/V 各占 nKV·d；reshape 与换轴不增加元素。Q/K 还沿每头 d 做 RMSNorm，形状不变。Q 宽度无须等于残差宽度 H。",
            `${width} = ${gated ? `2 × ${nQ} × ${d}` : `${nQ} × ${d}`}；K/V = ${nKV} × ${d} = ${nKV * d}`,
          ),
        ];
      } else return linearProjection();
    case "rope":
      return [
        step(
          "按通道对旋转 Q/K",
          [Q, Knew],
          [
            tensor("旋转 Q", Q.axes, Q.dimensions),
            tensor("旋转 K", Knew.axes, Knew.dimensions),
          ],
          "[x′₀,x′₁]ᵀ = R(position·frequency)[x₀,x₁]ᵀ",
          "2×2 旋转收缩通道对的长度 2，Q/K 总形状保持；V 不旋转。所选 Qwen 配置只旋转每头 25% 的通道。",
          `每头 ${d} 通道中 ${official ? d * config.partial_rotary_factor : d} 参与旋转，位置 q=${q} 不变`,
        ),
      ];
    case "attention":
      return [
        step(
          "GQA：共享 KV，再收缩 d",
          [Q, Kcache],
          [scores],
          "S[b,h,i,j] = Σᵣ Q[b,h,i,r] K[b,group(h),j,r] / √d",
          "Q 与 K 在 d 轴点积；保留查询 q 与历史 t 两个轴。每组 Q 读取同一 KV 头；缓存只存 nKV 头，逻辑分组无需复制缓存。",
          `[1, ${nQ}, ${q}, ${d}] × Kᵀ → [1, ${nQ}, ${q}, ${t}]；${nQ}/${nKV}=${nQ / nKV} 个 Q 共用 KV`,
        ),
        step(
          "因果概率收缩历史轴",
          [scores, Vcache],
          [context],
          "A = softmaxₜ(S + causal mask); O[b,h,i,r] = Σⱼ A[b,h,i,j] V[b,group(h),j,r]",
          "Softmax 对可见历史 t 归一化，形状不变；乘 V 时收缩 t，输出每个查询位置的 d 个内容通道。Decode 仍读取历史但只计算一个新 Q。",
          `[${q}, ${t}] × [${t}, ${d}] → [${q}, ${d}]；Prefill 通常 q=t，Decode q=1`,
        ),
      ];
    case "linear":
      return [
        ...linearProjection(),
        step(
          "固定状态的秩一更新",
          [
            tensor(
              "q/k（匹配 value 头后）",
              ["b", "q", "hV", "dk"],
              [1, q, hV, dk],
            ),
            tensor("v", ["b", "q", "hV", "dv"], [1, q, hV, dv]),
            tensor("S 固定状态", ["b", "hV", "dk", "dv"], [1, hV, dk, dv]),
          ],
          [
            tensor("S 固定状态", ["b", "hV", "dk", "dv"], [1, hV, dk, dv]),
            tensor("线性头输出", ["b", "q", "hV", "dv"], [1, q, hV, dv]),
          ],
          "S̄=αS; e=v−kᵀS̄; S=S̄+βkeᵀ; o=qᵀS",
          "采用 key×value 状态约定；kᵀS 收缩 key 轴，k·eᵀ 外积恢复 dk×dv。状态没有 t 轴，历史增长不会扩大状态。",
          `k[${dk}] × eᵀ[${dv}] → S[${dk}, ${dv}]；每请求 ${hV} 头，合计 ${hV * dk * dv} 个状态元素`,
        ),
      ];
    case "attention-output":
      return [
        step(
          "拼接、门控、投影回残差宽度",
          [
            tensor(
              "头拼接",
              ["b", "q", linear ? "hV·dv" : "nQ·d"],
              [1, q, outputWidth],
            ),
            tensor("W_o", [linear ? "hV·dv" : "nQ·d", "H"], [outputWidth, H]),
          ],
          [hidden],
          linear
            ? "Y = (RMSNorm(head output) ⊙ SiLU(z)) W_o"
            : gated
              ? "Y = (concat(heads) ⊙ activation(gate)) W_o"
              : "Y = concat(heads) W_o",
          "先保持 q 轴拼接头通道，再以通道门逐元素调节，最后收缩拼接宽度回到 H。线性输出门 z 的宽度与 value 输出相同。",
          `[1, ${q}, ${outputWidth}] × [${outputWidth}, ${H}] → [1, ${q}, ${H}]`,
        ),
      ];
    case "residual":
      return [
        step(
          "同形状逐元素相加",
          [x, tensor("模块分支 Δ", x.axes, x.dimensions)],
          [hidden],
          "Y[b,i,h] = X[b,i,h] + Δ[b,i,h]",
          "加法要求两条分支形状相同；没有收缩轴，不沿 Token 或隐藏通道拼接。",
          `[1, ${q}, ${H}] + [1, ${q}, ${H}] → [1, ${q}, ${H}]`,
        ),
      ];
    case "ffn":
      return ffnSteps(q, H, I);
    case "moe": {
      const E = model.expertCount ?? 0,
        k = model.activeExperts ?? 0;
      return [
        step(
          "路由与变长派发",
          [
            tensor("选中请求 X", ["q", "H"], [q, H]),
            tensor("W_router", ["H", "E"], [H, E]),
          ],
          [
            tensor("路由 logits", ["q", "E"], [q, E]),
            tensor("top-k ID / 权重", ["q", "k"], [q, k]),
            tensor("专家 e 输入", ["Nₑ", "H"], ["Nₑ", H]),
          ],
          "logits=XW_router; topk → indices, normalized weights; Σₑ Nₑ=q·k",
          "对 H 收缩得到每个专家分数。这里 q 是所选请求的位置数；整批 packed N 另列。每个 Token 产生 k 次派发，Nₑ 随路由选择变化，不能假设平均分给所有专家。",
          `[${q}, ${H}] × [${H}, ${E}] → [${q}, ${E}]；${q}×${k}=${q * k} 次派发`,
        ),
        ...ffnSteps("Nₑ", H, I),
        step(
          "恢复 Token 顺序并融合共享分支",
          [
            tensor("专家输出", ["q", "k", "H"], [q, k, H]),
            tensor("top-k 权重", ["q", "k", "1"], [q, k, 1]),
            tensor("共享分支 X", ["q", "H"], [q, H]),
            tensor("SharedFFN 输出", ["q", "H"], [q, H]),
            tensor("W_shared_gate", ["H", "1"], [H, 1]),
          ],
          [
            tensor("共享门", ["q", "1"], [q, 1]),
            tensor("MoE 输出", ["b", "q", "H"], [1, q, H]),
          ],
          "Yᵢ = Σₑ∈topk wᵢₑ Expertₑ(Xᵢ) + sigmoid(XᵢW_shared_gate)·SharedFFN(Xᵢ)",
          "按原 Token 索引 gather 并沿 k 求和；共享专家始终运行。共享门是每 Token 一个标量，广播到 H；共享 FFN 中间宽度来自 shared_expert_intermediate_size。",
          `[${q}, ${k}, ${H}] 沿 k 求和 → [${q}, ${H}]；共享门 [${q},1] 广播；共享中间宽度 ${moeConfig.text_config.shared_expert_intermediate_size}`,
        ),
      ];
    }
    case "lm-head":
      return [
        step(
          "末位置映射到词表",
          [
            tensor("末位置隐藏", ["b", "H"], [1, H]),
            tensor("W_vocab", ["H", "V"], [H, V]),
          ],
          [tensor("Logits", ["b", "V"], [1, V])],
          "logit[b,v] = Σₕ h_last[b,h] W_vocab[h,v]",
          "使用末个已处理位置预测下一 Token；收缩 H，输出保留整个词表 V，不是把 H 当成词表大小。",
          `[1, ${H}] × [${H}, ${V}] → [1, ${V}]`,
        ),
      ];
    case "sample":
      return [
        step(
          "词表概率到一个 ID",
          [tensor("Logits", ["b", "V"], [1, V])],
          [
            tensor("概率", ["b", "V"], [1, V]),
            tensor("新 Token ID", ["b"], [1]),
          ],
          "p=softmaxᵥ(logits/temperature); ID ~ filtered(p)",
          "沿 V 求归一化分母，但保留 V 项概率；采样后每请求只返回一个整数 ID。新 ID 要到下一轮才生成 KV。",
          `[1, ${V}] logits → [1, ${V}] 概率 → [1] ID`,
        ),
      ];
    default:
      return [];
  }
  function linearProjection() {
    return [
      step(
        "线性层独立投影与短卷积",
        [
          x,
          tensor("W_linear_qkv", ["H", "2·hK·dk+hV·dv"], [H, convWidth]),
          tensor("W_z", ["H", "hV·dv"], [H, valueWidth]),
          tensor("W_a / W_b", ["H", "hV"], [H, hV]),
        ],
        [
          tensor("线性 QKV", ["b", "q", "C"], [1, q, convWidth]),
          tensor("输出门 z", ["b", "q", "hV", "dv"], [1, q, hV, dv]),
          tensor("α / β 门", ["b", "q", "hV"], [1, q, hV]),
          tensor(
            "卷积缓存",
            ["b", "C", "kernel"],
            [1, convWidth, official ? config.linear_conv_kernel_dim : 3],
          ),
        ],
        "QKV=XW_qkv; z=XW_z; a=XW_a; b=XW_b",
        "Q/K 每个 hK×dk 通道，V/z 每个 hV×dv 通道；a/b 是每 value 头标量。深度卷积保持通道宽度，并使用固定长度缓存。",
        `QKV = 2×${hK}×${dk} + ${hV}×${dv} = ${convWidth}；z=${valueWidth}；a/b=${hV}；Q/K 头重复 ${hV / hK} 倍匹配 V`,
      ),
    ];
  }
}
function ffnSteps(N: number | string, H: number, I: number): ShapeStep[] {
  return [
    step(
      "门控前馈扩维",
      [
        tensor("FFN 输入", ["n", "H"], [N, H]),
        tensor("W_gate / W_up", ["H", "I"], [H, I]),
      ],
      [tensor("gate / up / 乘积", ["n", "I"], [N, I])],
      "u=XW_up; g=SiLU(XW_gate); z=g⊙u",
      "两次投影都收缩 H；SiLU 与逐元素乘保持 n×I（n 是当前算子的输入位置数）。FFN 不在 Token 轴之间做注意力。",
      `[${N}, ${H}] × [${H}, ${I}] → [${N}, ${I}]`,
    ),
    step(
      "前馈降维",
      [
        tensor("门控乘积", ["n", "I"], [N, I]),
        tensor("W_down", ["I", "H"], [I, H]),
      ],
      [tensor("FFN 输出", ["n", "H"], [N, H])],
      "Y[n,h] = Σⱼ z[n,j] W_down[j,h]",
      "收缩中间通道 I 回到 H，以便与同宽残差相加；扩维宽度由配置决定。",
      `[${N}, ${I}] × [${I}, ${H}] → [${N}, ${H}]`,
    ),
  ];
}
const committedStages: TraceStage[] = [
  "final-norm",
  "lm-head",
  "sample",
  "emit",
  "feedback",
  "finish",
  "release",
];
const structureStages: TraceStage[] = [
  "receive",
  "tokenize",
  "schedule",
  "emit",
  "feedback",
  "finish",
  "release",
];
export function deriveTraceShapes(
  frame: TraceFrame,
  requestId: string,
  model: ModelPreset,
): ShapeLesson {
  const selected = frame.requests.find((r) => r.id === requestId);
  const networkStage = !structureStages.includes(frame.stage);
  const batchRequests = (networkStage ? frame.requestIds : []).map((id) => {
    const r = frame.requests.find((r) => r.id === id)!;
    const prefill = r.status === "prefill" || frame.pass === "prefill";
    return {
      id,
      queryLength: prefill ? r.inputIds.length : 1,
      historyLength: committedStages.includes(frame.stage)
        ? Math.max(0, r.processedTokens - (prefill ? r.inputIds.length : 1))
        : r.processedTokens,
    };
  });
  const active = batchRequests.find((r) => r.id === requestId);
  const q = active?.queryLength ?? 0;
  const t = active ? active.historyLength + q : 0;
  const linear =
    frame.layer !== null &&
    model.layerTypes[frame.layer] === "linear_attention";
  const packed = batchRequests.reduce((sum, r) => sum + r.queryLength, 0);
  const currentObjects = frame.tensors
    .filter((t) => t.requestId === requestId || t.requestId === "batch")
    .map((t) =>
      t.name.includes("批次")
        ? {
            ...structure(
              t.name,
              t.rows,
              "每行是一项请求记录；两字段为输入长度与 Prefill 标记，不是神经网络激活。",
            ),
            axes: ["请求", "输入长度 / Prefill 标记"],
            dimensions: [t.rows, t.cols],
          }
        : tensor(
            t.name,
            toyAxes(t.name, t.rows, t.cols),
            [t.rows, t.cols],
            t.note,
          ),
    );
  const notes = networkStage
    ? [
        `当前层：${frame.layer === null ? "层外阶段" : `${frame.layer + 1} / ${model.layerTypes.length} · ${linear ? "线性递归" : "Full Attention"}`}。每个请求单独解释，b=1；本轮 packed N=${packed}。`,
        "变长批次按 N=Σqᵢ 打包，没有统一 padding 的 T；注意力与状态不跨请求。",
        `Prefill 一次处理提示的 q 个位置；Decode q=1，历史加当前位置得到 t。已提交缓存计数 ${selected?.cacheTokens ?? 0} 在全部层完成后更新；层内 KV 可先追加。`,
        "下列模型形状来自官方文本配置；实际数值网格来自 H=4 教学网络。教学投影省略部分模型门与归一化，不是 Qwen 激活。",
      ]
    : [
        "当前是请求生命周期阶段，没有执行网络算子；查询长度与 Full KV 注意力长度只在实际执行阶段显示。",
        `所选请求输入 ${selected?.inputIds.length ?? 0} 个 ID；已处理 ${selected?.processedTokens ?? 0} 个位置；已提交缓存 ${selected?.cacheTokens ?? 0} 个位置；块表 ${selected?.blocks.length ?? 0} 项。`,
        "编号序列的 q 表示此对象的序列长度；等待反馈的单个 ID 尚未生成 KV。调度批次表记录请求长度和阶段标记。",
      ];
  if (linear)
    notes.push(
      "模型线性状态采用 key×value；当前教学轨迹同样采用 key×value。短卷积教学历史保留 2 步、8 通道；模型缓存按 kernel=4 保存 C 通道。",
    );
  if (model.id === "qwen38-dense" && !linear)
    notes.push(
      "配置指定 output_gate_type=swish；基础 Qwen3.5 实现使用 sigmoid。门宽度相同，实际激活须以对应版本实现为准。",
    );
  let steps = active ? modelShapeSteps(frame.stage, model, q, t, linear) : [];
  if (!steps.length) {
    const input =
      frame.stage === "tokenize"
        ? structure("输入文本", 1)
        : structure("请求 / 调度状态", frame.requestIds.length);
    const output =
      frame.stage === "tokenize"
        ? tensor("Token IDs", ["q"], [selected?.inputIds.length ?? 0])
        : structure(
            frame.stage === "release"
              ? "回收后的块表"
              : frame.stage === "schedule"
                ? "packed 批次描述"
                : "请求 / 流式输出状态",
            frame.stage === "release"
              ? selected?.blocks.length
              : frame.requestIds.length,
          );
    steps = [
      step(
        frame.label,
        [input],
        [output],
        frame.stage === "tokenize"
          ? "text → tokenizer → IDs"
          : "request state → scheduler / stream / cache lifecycle",
        frame.requestIds.includes(requestId)
          ? "这里操作的是请求、编号序列或块映射。神经网络矩阵形状由进入执行器时的查询长度确定。"
          : "所选请求这一帧没有执行网络计算；其状态保持。选择当前活动请求可查看该阶段模型张量推导。",
        `活动请求 ${frame.requestIds.length}；所选输入 ${selected?.inputIds.length ?? 0} 个 ID；物理块表 ${selected?.blocks.length ?? 0} 项`,
      ),
    ];
  }
  return {
    stage: frame.label,
    scope: structureStages.includes(frame.stage)
      ? "当前请求对象与逻辑数据"
      : "模型逻辑维度 / 实际未运行模型",
    steps,
    notes,
    currentObjects,
    queryLength: q,
    keyLength: t,
    packedTokens: packed,
    batchRequests,
    axes: networkStage
      ? [
          { symbol: "b", meaning: "当前解释的请求数", value: 1 },
          { symbol: "q", meaning: "本轮新查询位置", value: q },
          { symbol: "t", meaning: "历史 + 本轮位置（Full KV）", value: t },
          { symbol: "N", meaning: "整批 packed Token 数", value: packed },
          {
            symbol: "n",
            meaning: "当前算子展平位置数（按请求为 q，按专家为 Nₑ）",
          },
          { symbol: "H", meaning: "残差隐藏通道", value: model.hiddenSize },
          {
            symbol: "nQ / nKV",
            meaning: "Full 查询头 / KV 头",
            value: `${model.qHeads} / ${model.kvHeads}`,
          },
          { symbol: "d", meaning: "Full 每头通道", value: model.headDim },
          {
            symbol: "I",
            meaning: "FFN 或每专家中间通道",
            value: model.ffnSize,
          },
          {
            symbol: "hK / hV",
            meaning: "线性 key / value 头",
            value: `${model.linearKeyHeads} / ${model.linearValueHeads}`,
          },
          {
            symbol: "dk / dv",
            meaning: "线性 key / value 每头通道",
            value: model.linearHeadDim,
          },
        ]
      : [
          { symbol: "b", meaning: "编号序列所属请求数", value: 1 },
          {
            symbol: "q",
            meaning: "当前 ID 对象的序列长度（此阶段不表示网络查询）",
          },
          {
            symbol: "请求 / 字段",
            meaning: "调度记录行 / 输入长度与阶段标记列",
          },
        ],
    sources: [...model.sources, implementationSource],
  };
}
function toyAxes(name: string, rows: number, cols: number): string[] {
  if (name.includes("Token ID") || name === "待反馈 ID") return ["b", "q"];
  if (name === "短卷积末步 QKV") return ["末查询位置", "QKV 拼接通道"];
  if (name === "衰减 α / 写入 β") return ["末查询位置", "α / β 标量门"];
  if (["SiLU 门控", "上投影", "门控乘积"].includes(name))
    return ["q", "教学中间通道 I=6"];
  if (["Q", "旋转 Q", "上下文"].includes(name)) return ["q", "教学 nQ·d=4"];
  if (["K", "旋转 K", "V"].includes(name)) return ["q", "教学 nKV·d=2"];
  if (
    [
      "路由加权和",
      "共享专家",
      "末位置隐藏向量",
      "Final Norm",
      "隐藏输入",
    ].includes(name)
  )
    return ["末查询位置", "H=4"];
  if (
    [
      "Embedding",
      "输入 x",
      "归一化 x̂",
      "输出投影",
      "残差输出",
      "MoE 输出",
      "FFN 输出",
      "层输出",
    ].includes(name)
  )
    return ["q", "H=4"];
  if (name.startsWith("递归状态")) return ["key 通道", "value 通道"];
  if (name.startsWith("KV：")) return ["缓存位置 t", "KV 拼接通道"];
  if (name.startsWith("注意力")) return ["Q 头（仅末查询）", "历史位置 t"];
  if (name.includes("专家 [ID")) return ["选中专家 k", "ID / 权重"];
  if (name === "选中专家输出") return ["选中专家 k", "H=4"];
  if (name.includes("路由概率")) return ["末查询位置", "教学专家 E=3"];
  if (name.includes("批次")) return ["请求", "长度 / 阶段标记"];
  if (name === "卷积历史") return ["有限历史", "QKV 通道"];
  if (
    name.includes("Logits") ||
    name.includes("Softmax") ||
    name.includes("过滤概率")
  )
    return ["末查询位置", "候选词 V=6"];
  return [
    rows === 1 ? "展示行（见数值说明）" : "查询位置 q",
    cols === 4 ? "H=4 / 拼接通道" : "通道 / 记录字段",
  ];
}

const engineeringIds = new Set([
  "kv-cache",
  "paged-attention",
  "prefix-caching",
  "batching",
  "speculation",
  "chunked-prefill",
  "cuda-graphs",
  "metrics",
  "frameworks",
]);
// Axes describe the operator's semantic indices, never guessed from equal widths.
function mechanismAxes(id: string, name: string, rank: 1 | 2): string[] {
  if (rank === 2) {
    switch (id) {
      case "embedding":
        return ["V", "H"];
      case "qkv":
        return ["H", "d"];
      case "rope":
        return ["旋转输出通道", "旋转输入通道"];
      case "attention":
        if (name === "Kᵀ") return ["d", "t"];
        if (name === "Q" || name.startsWith("Q ·") || name === "当前查询 O")
          return ["q", "d"];
        if (name === "K" || name.startsWith("K ·") || name === "V")
          return ["t", "d"];
        return ["q", "t"];
      case "attention-output":
        return name.includes("W") ? ["nQ·d", "H"] : ["nQ", "d"];
      case "ffn":
        return name.includes("down") ? ["I", "H"] : ["H", "I"];
      case "gqa":
        return ["nQ", "t"];
      case "gated-deltanet":
        return ["dv", "dk"];
      case "moe":
        return ["k", "H"];
      case "prefill":
        return /因果|可见性/.test(name) ? ["q", "t"] : ["q", "H"];
      case "decode":
        if (name.includes("概率")) return ["q", "t"];
        return [
          name.includes("缓存") || name.startsWith("历史") ? "t" : "q",
          "d",
        ];
      case "lm-head":
        return ["H", "V"];
      case "kv-cache":
        return ["已处理位置 t", "d"];
      case "quantization":
        return ["原始 / 重建", "权重样本"];
      case "flash-attention":
        return ["tile", "t_tile"];
      case "parallelism":
        return name === "rank 部分结果"
          ? ["rank", "F"]
          : [
              name.includes("局部") || name.includes("本 rank")
                ? "H_local"
                : "H",
              "F",
            ];
      default:
        return ["行", "列"];
    }
  }
  switch (id) {
    case "embedding":
      return [name.includes("ID") ? "q" : "H"];
    case "rmsnorm":
      return [/均方|分母|^r$/.test(name) ? "标量" : "H"];
    case "qkv":
      return [
        name.includes("输入") || name === "x" || name.includes("贡献")
          ? "H"
          : name.includes("拼接")
            ? "Q / K / V 拼接通道"
            : "d",
      ];
    case "rope":
      return [/位置|频率|角度/.test(name) ? "标量" : "通道对"];
    case "attention":
      return [/概率|权重/.test(name) ? "t" : "d"];
    case "softmax":
      return [name.includes("分母") ? "标量" : "V"];
    case "attention-output":
      return [
        name.includes("拼接")
          ? "nQ·d"
          : name.includes("残差") || /投影后|投影结果/.test(name)
            ? "H"
            : "d",
      ];
    case "residual":
      return ["H"];
    case "ffn":
      return [
        name === "x" || name.includes("输入 x") || name.includes("FFN 输出")
          ? "H"
          : "I",
      ];
    case "gated-deltanet":
      return [
        name.includes("缓冲")
          ? "有限历史"
          : name === "k" || name === "k / q"
            ? "dk"
            : "dv",
      ];
    case "moe":
      return [
        /logits|概率|专家选择/.test(name)
          ? "E"
          : /权重|ID/.test(name)
            ? "k"
            : "H",
      ];
    case "decode":
      return [
        name.includes("注意力")
          ? "t"
          : name.includes("Q") || name.includes("聚合")
            ? "d"
            : "H",
      ];
    case "sampling":
      return ["V"];
    case "lm-head":
      return [name.includes("隐藏") || /位置 h|贡献/.test(name) ? "H" : "V"];
    case "quantization":
      return [name.includes("scale") ? "标量" : "权重样本"];
    case "flash-attention":
      if (name.includes("分母")) return ["标量"];
      if (
        name.includes("输出") ||
        name.includes("结果") ||
        name.includes("对照") ||
        name.includes("内容累积")
      )
        return ["dv"];
      return [name.includes("tile") ? "t_tile" : "t"];
    case "parallelism":
      return [
        name.includes("输入") || name === "全局 x"
          ? "H"
          : name === "局部 x"
            ? "H_local"
            : "F",
      ];
    default:
      return ["记录项"];
  }
}
function isPanelMetadata(id: string, name: string): boolean {
  return (
    id === "gqa" ||
    (id === "flash-attention" && /m.*l.*o/.test(name)) ||
    (id === "quantization" && (name.includes("scale") || name.includes("边界")))
  );
}
function panelObject(panel: MechanismPanel, id: string): ShapeObject {
  const isMetadata =
    (engineeringIds.has(id) && !panel.matrix) ||
    isPanelMetadata(id, panel.label) ||
    panel.kind === "rotation";
  if (panel.matrix)
    return tensor(
      panel.label,
      mechanismAxes(id, panel.label, 2),
      [panel.matrix.length, panel.matrix[0]?.length ?? 0],
      panel.note,
    );
  if (panel.values)
    return isMetadata
      ? {
          ...structure(
            panel.label,
            panel.values.length,
            panel.kind === "rotation"
              ? "绘图坐标 x、y 与角度三个字段；实际旋转向量只有两个通道。"
              : "统计字段 / 编号记录，不是神经网络激活",
          ),
          dimensions: [panel.values.length],
        }
      : tensor(
          panel.label,
          mechanismAxes(id, panel.label, 1),
          [panel.values.length],
          panel.note,
        );
  return structure(
    panel.label,
    panel.items?.length ?? 0,
    panel.note ?? "Token、请求、图节点或缓存映射项；不视为密集矩阵。",
  );
}
const mechanismRules: Record<string, string> = {
  tokenization:
    "合并相邻片段会减少序列长度，最后每个片段查到一个整数 ID；没有隐藏通道轴。",
  embedding:
    "词表索引选择一行，Token 数不变；输出增加隐藏通道轴，不对词表求和。",
  rmsnorm: "平方逐元素保持形状，均值只归约通道轴，标量分母广播回全部通道。",
  qkv: "相同输入乘三份独立矩阵，分别收缩输入通道；投影通道由每份权重的列数决定。",
  rope: "两通道构成一个坐标对；位置只改变 2×2 旋转矩阵，旋转输出维度不变。",
  attention:
    "QKᵀ 收缩每头通道形成查询×键位置；Softmax 保留形状；乘 V 时收缩键位置。",
  softmax:
    "逐元素指数保持词表轴，分母求和归约为标量；广播除法恢复同长概率向量。",
  "attention-output":
    "沿通道拼接头，不增加 Token；输出投影收缩拼接通道回到残差宽度。",
  residual: "两条同形状分支逐元素相加，没有收缩、拼接或广播不同宽度分支。",
  ffn: "上投影和门投影都从隐藏通道扩展到中间通道，逐元素门乘保持宽度，下投影恢复隐藏宽度。",
  gqa: "查询头数固定，共享 KV 头须整除查询头；缓存按 KV 头存储，读取结果仍按 Q 头输出。",
  "gated-deltanet":
    "本原理动图采用 value×key 状态：S·k 收缩 key 轴，误差 e·kᵀ 是外积；S·q 输出 value 通道。",
  moe: "路由打分沿专家轴选 top-k，选中权重重新归一化；对 k 个专家输出加权求和恢复隐藏宽度。",
  prefill:
    "本轮查询位置等于已知输入数，因果遮罩是位置×位置；末位置用于预测首个生成 Token。",
  decode:
    "本轮只处理一个新查询；历史加新位置构成 KV 长度；新采样 Token 下一轮才进入缓存。",
  sampling:
    "筛选只把未选词概率置零并重新归一，词表向量长度不变；最终离散采样输出一个 ID。",
  "lm-head":
    "末位置隐藏向量乘隐藏×词表矩阵，收缩隐藏轴，输出每个词表候选的 logit。",
  "kv-cache":
    "缓存条目对应已处理位置的 K/V；采样文本不直接成为 KV，完成请求时释放所属块。",
  "paged-attention":
    "逻辑位置通过块表映射到非连续物理块；块数按每请求长度向上取整，不是注意力矩阵形状。",
  "prefix-caching":
    "匹配的是同模型同条件前缀对应的缓存位置；只复用已计算前缀，未命中后缀仍要执行。",
  batching:
    "等待队列与活动槽位按请求记录变化；这一轮执行的是每个活动请求的一个新位置，缓存归属保持独立。",
  quantization:
    "逐元素编码与重建不改变权重逻辑形状；位宽改变存储字节数，scale 为共享标量元数据。",
  speculation:
    "候选序列先由目标模型验证；只保留第一次拒绝前的接受前缀并追加修正或 bonus Token。",
  "flash-attention":
    "按历史 tile 分块读取，在线 m/l 标量与内容累积保持固定宽度；最终 o/l 与整行 Softmax 相同。",
  parallelism:
    "输入通道和权重行同步切分，各 rank 输出同宽部分点积；sum 归约恢复完整结果。",
  "chunked-prefill":
    "每轮预算是当前 prefill 块位置数加 decode 的一个位置，尾块按剩余输入缩短。",
  "cuda-graphs":
    "图对象保存 kernel 节点、依赖、形状与缓冲区地址契约；重放不省略网络数学计算。",
  metrics:
    "计时对象是事件记录与标量统计；TTFT、ITL 和吞吐使用各自时间边界，不定义神经张量轴。",
  frameworks:
    "请求队列、调度预算、执行器与 kernel 分层管理数据；示意计算结果不代表具体框架的统一 tensor 布局。",
};

export function deriveMechanismShapes(
  id: string,
  frameIndex: number,
  params: Record<string, number> = {},
): ShapeLesson {
  const resolvedParams = Object.fromEntries(
    (mechanisms.find((m) => m.id === id)?.parameters ?? []).map((p) => [
      p.key,
      params[p.key] ?? p.initial,
    ]),
  );
  const frames = buildMechanismFrames(id, resolvedParams);
  const current = frames[frameIndex];
  if (!current) throw Error("形状讲解需要有效的机制与当前关键帧");
  const objects = current.panels.map((p) => panelObject(p, id));
  const rule = mechanismRules[id];
  if (!rule) throw Error("未定义该原理的形状推导");
  const numeric = current.panels.flatMap((p) =>
    p.matrix
      ? [p.matrix.length, p.matrix[0]?.length ?? 0]
      : p.values
        ? [p.values.length]
        : [],
  );
  const steps = mechanismSteps(
    id,
    frameIndex,
    current,
    frames,
    resolvedParams,
    objects,
    rule,
  );
  const notes = [
    "此处形状属于当前小数组 / 机制 fixture，每个原理的维度可能不同；流程工作台的 H=4 不能套到全部原理。",
    "矩阵乘统一写 [输入通道, 输出通道] 的数学权重；向量省略长度为 1 的批与位置轴。",
    "当前读数的项数统计展示记录；‘暂无’、‘队列空’等文字是占位说明，实际等待 / 分配计数见运算读数。",
  ];
  if (engineeringIds.has(id))
    notes.push(
      "请求、队列、块表和执行图是结构对象；项数是结构长度，不是隐藏维度。统计读数不代表 GPU 实测。",
    );
  if (id === "gated-deltanet")
    notes.push(
      "原理动图使用 value×key；流程轨迹与模型对照使用 key×value。两种约定互为转置，读取方向随之改变。",
    );
  if (id === "moe")
    notes.push(
      "该 fixture 使用固定专家输出与固定共享向量，并未计算专家 FFN 或共享门。模型对照才显示配置的 FFN 与标量共享门。",
    );
  return {
    stage: current.stage,
    scope: engineeringIds.has(id)
      ? "当前工程对象 / 机制 fixture"
      : "当前原理小数组 / 精确形状",
    steps,
    notes,
    currentObjects: objects,
    queryLength: 0,
    keyLength: 0,
    packedTokens: 0,
    batchRequests: [],
    axes: [
      { symbol: "q / t", meaning: "查询位置 / 被读取的历史位置" },
      { symbol: "H / I", meaning: "隐藏通道 / FFN 中间通道" },
      { symbol: "nQ / nKV", meaning: "查询头 / KV 头" },
      {
        symbol: "E / k / Nₑ",
        meaning: "专家总数 / 每 Token 选中专家 / 专家实际接收 Token",
      },
      {
        symbol: "V / 标量 / rank / F / t_tile",
        meaning:
          "词表候选 / 单个归约值 / 设备编号 / 投影输出通道 / tile 内历史位置",
      },
      {
        symbol: "行 / 列",
        meaning: `直接由当前数据数组测得${numeric.length ? `（当前维度读数 ${numeric.join(" / ")}）` : "；结构对象另列项数"}`,
      },
    ],
  };
}
function mechanismSteps(
  id: string,
  i: number,
  current: MechanismFrame,
  frames: MechanismFrame[],
  p: Record<string, number>,
  objects: ShapeObject[],
  rule: string,
): ShapeStep[] {
  const vec = (name: string, n: number | string) =>
    isPanelMetadata(id, name) && id === "flash-attention"
      ? {
          ...structure(
            name,
            Number(n),
            "在线归约的 m、l 与 o 标量记录；o 的 value 宽度为 1。",
          ),
          dimensions: [n],
        }
      : tensor(name, mechanismAxes(id, name, 1), [n]);
  const mat = (name: string, r: number | string, c: number | string) =>
    tensor(name, mechanismAxes(id, name, 2), [r, c]);
  const make = (
    inputs: ShapeObject[],
    outputs: ShapeObject[],
    formula: string,
    substitution: string,
    extra = "",
  ) => [
    step(
      current.stage,
      inputs,
      outputs,
      formula,
      `${rule} ${extra}`,
      substitution,
    ),
  ];
  const previous = frames[Math.max(0, i - 1)].panels.map((panel) =>
    panelObject(panel, id),
  );
  switch (id) {
    case "tokenization": {
      const out =
        current.panels.find((p) => p.kind === "tokens")?.items?.length ??
        current.output?.length ??
        5;
      const before =
        i === 0
          ? 5
          : (frames[i - 1].panels.find((p) => p.kind === "tokens")?.items
              ?.length ?? 5);
      return make(
        [structure("输入片段", before)],
        [
          i === 4
            ? tensor("Token ID", ["Token"], [out])
            : structure("当前片段", out),
        ],
        current.formula,
        `${before} 个片段 → ${out} ${i === 4 ? "个 ID" : "个片段"}；合并不添加隐藏通道`,
      );
    }
    case "embedding":
      return make(
        [tensor("ID", ["位置"], [1]), mat("E", 4, 3)],
        [tensor("查表输出", ["位置", "H"], [1, 3])],
        "E[2,:] → x",
        "ID=2 选择 4×3 表的第 2 行 → 1×3；读数向量显示为 [3]",
      );
    case "rmsnorm":
      return i === 1
        ? make(
            [vec("x", 4)],
            [vec("x²", 4), vec("均方", 1)],
            "meanSquare=Σₕ xₕ²/4",
            "[4] → 平方 [4] → 通道求和得到 [1]",
          )
        : i === 2
          ? make(
              [vec("均方", 1)],
              [vec("r", 1)],
              current.formula,
              "[1] + ε → sqrt → [1]",
            )
          : make(
              [vec("x", 4), vec("γ", 4)],
              [vec(i === 0 ? "原输入" : "归一化 y", 4)],
              current.formula,
              "4 个隐藏通道，标量分母广播到 4 通道",
            );
    case "qkv":
      return i === 1
        ? make(
            [vec("x", 2), mat("W_Q 第一列", 2, 1)],
            [vec("逐项贡献", 2), vec("Q₀", 1)],
            current.formula,
            "[2]·[2] = 1×0 + 2×1 = 2（一个输出通道）",
          )
        : make(
            [vec("x", 2), mat("W_Q / W_K / W_V", 2, 2)],
            [vec("Q", 2), vec("K", 2), vec("V", 2)],
            current.formula,
            "每份 [2] × [2,2] → [2]；末帧拼接读数是 2+2+2=6 项，不是增加 6 个 Token",
          );
    case "rope":
      return i === 1
        ? make(
            [vec("位置 p", 1), vec("频率 θ", 1)],
            [mat("R(φ)", 2, 2)],
            current.formula,
            `p=${p.position ?? 2}，φ=${p.position ?? 2}×π/8；旋转矩阵 [2,2]`,
          )
        : make(
            [mat("R(φ)", 2, 2), vec("通道对 x", 2)],
            [vec("旋转通道对", 2)],
            "x′=R(φ)x",
            "[2,2] × [2] → [2]；绘图三字段不等于向量三个通道",
          );
    case "attention":
      return i === 3
        ? make(
            [mat("当前查询概率", 1, 3), mat("V", 3, 2)],
            [mat("当前查询 O", 1, 2)],
            current.formula,
            `选中查询 i=${p.row ?? 2}：[1,3] × [3,2] → [1,2]`,
          )
        : make(
            [
              mat(i === 0 ? "Q" : "分数", 3, i === 0 ? 2 : 3),
              mat(i === 0 ? "Kᵀ" : "因果 mask", i === 0 ? 2 : 3, 3),
            ],
            [mat(i === 0 ? "分数" : "遮罩分数 / 概率", 3, 3)],
            current.formula,
            i === 0
              ? "[3,2] × [2,3] → [3,3]，收缩每头通道 2"
              : "3×3 的查询×历史轴保持；按每行可见历史做 Softmax",
          );
    case "softmax":
      return make(
        [vec("logits", 4)],
        [vec(i === 2 ? "概率" : "分数 / 指数权重", 4), vec("归约分母", 1)],
        current.formula,
        `V=4；${p.temperature === 0 ? "T=0 使用 argmax" : "分母归约 4 个权重为 1 项"}，概率轴仍有 4 项`,
      );
    case "attention-output":
      return i === 0
        ? make(
            [vec("头 0", 2), vec("头 1", 2)],
            [mat("头输出集合", 2, 2)],
            current.formula,
            "2 个头 × 每头 2 通道，保持各头独立",
          )
        : i === 1
          ? make(
              [vec("头 0", 2), vec("头 1", 2)],
              [vec("拼接", 4)],
              current.formula,
              "[2] concat [2] → [4]；只有一个查询位置",
            )
          : make(
              [vec("拼接", 4), mat("W_o", 4, 2)],
              [vec("残差宽度输出", 2)],
              current.formula,
              "[4] × [4,2] → [2]；本例残差宽度为 2",
            );
    case "residual":
      return make(
        [vec("x", 4), vec("Δ", 4)],
        [vec("y", 4)],
        current.formula,
        `[4]+[4]→[4]；分支系数 ${p.branch ?? 1} 只改变值`,
      );
    case "ffn":
      return i === 3
        ? make(
            [vec("门控中间值", 4), mat("W_down", 4, 2)],
            [vec("FFN 输出", 2)],
            current.formula,
            "[4] × [4,2] → [2]，收缩 I=4",
          )
        : i === 2
          ? make(
              [vec("gate", 4), vec("up", 4)],
              [vec("SiLU(gate)⊙up", 4)],
              current.formula,
              "[4] ⊙ [4] → [4]，没有求和",
            )
          : make(
              [vec("x", 2), mat("W_up / W_gate", 2, 4)],
              [vec("up", 4), vec("gate", 4)],
              current.formula,
              "H=2，I=4：[2] × [2,4] → [4]",
            );
    case "gqa": {
      const kv = p.kvHeads ?? 2;
      return make(
        [
          tensor("Q 头", ["nQ", "q", "d"], [4, 1, "d"]),
          tensor("KV 头", ["nKV", "t", "d"], [kv, 2, "d"]),
        ],
        [
          mat("示意读取权重", 4, 2),
          tensor("各 Q 头输出", ["nQ", "q", "d"], [4, 1, "d"]),
        ],
        current.formula,
        `4/${kv}=${4 / kv} 个 Q 共用一个 KV；缓存比例 ${kv}/4=${kv / 4}`,
        "本例只有头映射和 4×2 权重 fixture，未提供数值 Q/K/V；d 保持符号，不能从头数猜测头维度。",
      );
    }
    case "gated-deltanet":
      return make(
        [mat("S（value×key）", 2, 2), vec("k / q", 2), vec("v / e", 2)],
        [mat("更新 S", 2, 2), vec("读取 S·q", 2)],
        current.formula,
        `S[2,2] × k[2] → 预测 [2]；e[2] × kᵀ[2] → [2,2]；α=${p.alpha ?? 0.8}，β=0.7`,
      );
    case "moe":
      return i < 2
        ? make(
            [vec("路由 logits", 8)],
            [vec("完整概率", 8), vec("top-2 权重 / ID", 2)],
            current.formula,
            "一个 Token 对 E=8 打分，选择 k=2；选中权重 [2]，完整概率保持 [8]",
          )
        : make(
            [
              mat("选中专家输出", 2, 2),
              vec("归一权重", 2),
              vec("固定共享向量", 2),
            ],
            [vec("融合 y", 2)],
            current.formula,
            "k=2 × H=2，沿 k 加权求和 → H=2；加共享 [2]",
            "本 fixture 专家输出直接由选中 ID 给定，并未模拟真实专家中间宽度。",
          );
    case "prefill": {
      const q = p.tokens ?? 4;
      return i === 3
        ? make(
            [mat("所有位置隐藏", q, 2)],
            [mat("末位置隐藏", 1, 2), structure("第一个采样 Token", 1)],
            current.formula,
            `从 [${q},2] 选择最后一行 → [1,2]；新输出尚未进 KV`,
          )
        : make(
            [mat("教学隐藏", q, 2)],
            [
              mat("因果可见性", q, q),
              structure("已处理 KV 位置", i < 2 ? 0 : q),
            ],
            current.formula,
            `q=${q}，Prefill 无历史时 t=${q}；可见性 [${q},${q}]，每位置隐藏宽度 2`,
          );
    }
    case "decode":
      return i === 0
        ? make(
            [structure("历史 KV 位置", 3), structure("新 Token 输入", 1)],
            [structure("待计算查询", 1)],
            current.formula,
            "历史 t_old=3；新查询 q=1，待追加缓存",
          )
        : i === 3
          ? make(
              [vec("末位置隐藏", 2)],
              [structure("新采样 Token", 1), structure("已处理 KV 位置", 4)],
              current.formula,
              "生成 g1 一个 Token；KV 仍有 4 位置，g1 下一轮再写入",
            )
          : make(
              [mat("新 Q", 1, 2), mat("K / V 缓存", 4, 2)],
              [mat("当前查询概率", 1, 4), mat("O", 1, 2)],
              current.formula,
              "q=1，t=3+1=4；[1,2]×[2,4]→[1,4]；[1,4]×[4,2]→[1,2]",
            );
    case "sampling":
      return make(
        [vec("词表分布", 4)],
        [i < 2 ? vec("过滤概率", 4) : structure("选中 Token ID", 1)],
        current.formula,
        `V=4，top-k=${p.topK ?? 3} 保留非零项但向量长度仍为 4；固定 u=0.42 最终选 1 个 ID`,
      );
    case "lm-head":
      return make(
        [vec("末位置 h", 2), mat("W_vocab", 2, 4)],
        [vec(i === 3 ? "概率" : "logits", 4)],
        current.formula,
        "H=2，V=4：[2] × [2,4] → [4]；对每个词表候选收缩 2 个隐藏通道",
      );
    case "quantization":
      return make(
        [vec("原始权重", 6), vec("共享 scale", 1)],
        [vec(i === 2 ? "整数码 q" : "重建 / 处理权重", 6)],
        current.formula,
        `6 个权重保持 [6]；位宽 b=${p.bits ?? 4}，整数 payload=ceil(6×b/8)=${Math.ceil((6 * (p.bits ?? 4)) / 8)} 字节，不含 scale`,
      );
    case "flash-attention":
      return i === 0
        ? make(
            [vec("分数行", 4), vec("V", 4)],
            [mat("分数 tiles", 2, 2)],
            current.formula,
            "1 查询、t=4、教学 value 宽度 1；按历史分成 2 个 tile，每个 2 项",
          )
        : i === 3
          ? make(
              [vec("内容累积 o", 1), vec("分母 l", 1)],
              [vec("输出", 1)],
              current.formula,
              "所有 4 个历史位置归约完成；[1]/[1]→[1]",
            )
          : make(
              [vec("分数 tile", 2), vec("V tile", 2), vec("m/l/o 元数据", 3)],
              [vec("更新 m/l/o", 3)],
              current.formula,
              "tile 长度 2；每 tile 将历史轴归约为 m、l、o 三个标量；内容宽度仍为 1",
            );
    case "parallelism": {
      const ranks = p.devices ?? 2;
      return i === 0
        ? make(
            [vec("全局 x", 4), mat("W", 4, 2)],
            [
              tensor("本 rank x", ["H_local"], [4 / ranks]),
              mat("本 rank W", 4 / ranks, 2),
            ],
            current.formula,
            `输入 H=4，rank=${ranks}，本地 H=${4 / ranks}`,
          )
        : i === 1
          ? make(
              [vec("局部 x", 4 / ranks), mat("局部 W", 4 / ranks, 2)],
              [vec("每 rank 部分输出", 2)],
              current.formula,
              `[${4 / ranks}] × [${4 / ranks},2] → [2]`,
            )
          : make(
              [mat("rank 部分结果", ranks, 2)],
              [vec("全局输出", 2)],
              current.formula,
              `${ranks} 份 [2] 沿 rank 求和 → [2]，输出没有拼接成 ${2 * ranks} 通道`,
            );
    }
    case "paged-attention": {
      const size = p.blockSize ?? 4,
        a = Math.ceil(5 / size),
        b = Math.ceil(2 / size);
      return make(
        [structure("A/B 逻辑位置", 7)],
        [
          structure("A 页表", a),
          structure("B 页表", b),
          structure("分配物理块", a + b),
        ],
        current.formula,
        `ceil(5/${size})+ceil(2/${size})=${a}+${b}=${a + b} 块；容量 ${(a + b) * size}，尾部空位 ${(a + b) * size - 7}`,
      );
    }
    case "kv-cache":
      return make(
        [
          structure("本轮已处理输入", i === 0 ? 3 : i === 3 ? 1 : 0),
          structure("请求缓存归属", 1),
        ],
        [structure("已缓存位置", current.readouts.cachedTokens)],
        current.formula,
        `当前已缓存位置=${current.readouts.cachedTokens}；${i === 4 ? "释放 4 个位置，分配块回收" : "每位置一份 K 与一份 V，不含尚未计算的采样 Token"}`,
      );
    case "chunked-prefill": {
      const chunk = p.chunk ?? 2,
        rounds = Math.ceil(6 / chunk),
        working = i > 0 && i <= rounds;
      const n = working ? Math.min(chunk, 6 - (i - 1) * chunk) : 0;
      return make(
        [
          structure("A 尚未处理的输入", working ? 6 - (i - 1) * chunk : 6),
          structure("B decode 查询", 1),
        ],
        [
          structure(
            working ? "本轮新查询预算" : "调度 / 输出对象",
            working ? n + 1 : 2,
          ),
        ],
        current.formula,
        working
          ? `A 当前块 min(${chunk},剩余)=${n}；加 B 1 → N=${n + 1}；不是固定每轮 ${chunk + 1}`
          : `A 共 6 输入，每块最多 ${chunk}，共 ${rounds} 轮；A 完成输入才采样`,
      );
    }
    default: {
      const counts = objects
        .map((o) => `${o.name} ${formatShape(o)}`)
        .join("；");
      return make(
        previous.length ? previous : [structure("当前输入对象", 1)],
        objects,
        current.formula,
        `${counts}${
          Object.keys(current.readouts).length
            ? `；读数 ${Object.entries(current.readouts)
                .map(
                  ([k, v]) => `${k}=${Number.isInteger(v) ? v : v.toFixed(3)}`,
                )
                .join("，")}`
            : ""
        }`,
        "输入与输出列表是此关键帧涉及的结构 / 读数快照，图节点条数不表示 GPU 时间或张量维度。",
      );
    }
  }
}
