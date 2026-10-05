import type { SourceRef } from "../types";
export type FrameworkModule = {
  id: string;
  name: string;
  responsibility: string;
  inputs: string[];
  outputs: string[];
  relatedTopics: string[];
  sources: SourceRef[];
  snippet: { kind: "source" | "pseudocode"; code: string; explanation: string };
};
export type FrameworkDefinition = {
  id: "vllm" | "sglang";
  release: string;
  commit: string;
  modules: FrameworkModule[];
  edges: {
    from: string;
    to: string;
    kind: "call" | "message" | "data";
    label: string;
  }[];
};
export const frameworks: FrameworkDefinition[] = [
  {
    id: "vllm",
    release: "v0.30.0",
    commit: "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
    modules: [
      {
        id: "api",
        name: "OpenAI API Router",
        responsibility:
          "解析协议请求并调用 Chat Serving，返回完整响应或 SSE。该快照中旧 api_server.py 已是兼容重导出，实际路由位于 chat_completion/api_router.py。",
        inputs: ["HTTP / ChatCompletionRequest"],
        outputs: ["ChatCompletionResponse / SSE"],
        relatedTopics: ["frameworks", "tokenization"],
        sources: [
          {
            id: "vllm-api",
            title:
              "vllm/entrypoints/openai/chat_completion/api_router.py · create_chat_completion",
            url: "https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/openai/chat_completion/api_router.py#L58-L62",
            checkedAt: "2026-10-04",
            revision: "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            path: "vllm/entrypoints/openai/chat_completion/api_router.py",
            symbol: "create_chat_completion",
            startLine: 58,
          },
          {
            id: "legacy",
            title: "旧 api_server 兼容入口",
            url: "https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/openai/api_server.py#L24-L30",
            checkedAt: "2026-10-04",
            revision: "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            path: "vllm/entrypoints/openai/api_server.py",
            symbol: "warnings.warn",
            startLine: 24,
          },
        ],
        snippet: {
          kind: "source",
          code: 'handler = chat(raw_request)\nif handler is None:\n    raise NotImplementedError("The model does not support Chat Completions API")\n\ngenerator = await handler.create_chat_completion(request, raw_request)',
          explanation:
            "从应用状态取 Serving 处理器，再交给协议层；图中省略 Serving 和模板渲染等中间层。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "client",
        name: "AsyncLLM",
        responsibility:
          "管理异步请求、输出处理与客户端队列，把 EngineCoreRequest 送进独立 EngineCore；流式输出从队列返回协议层。",
        inputs: ["提示与采样参数", "EngineCoreOutputs"],
        outputs: ["EngineCoreRequest", "RequestOutput"],
        relatedTopics: ["tokenization", "sampling"],
        sources: [
          {
            id: "vllm-client",
            title: "vllm/v1/engine/async_llm.py · AsyncLLM._add_request",
            url: "https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/async_llm.py#L520-L524",
            checkedAt: "2026-10-04",
            revision: "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            path: "vllm/v1/engine/async_llm.py",
            symbol: "AsyncLLM._add_request",
            startLine: 520,
          },
        ],
        snippet: {
          kind: "source",
          code: "# Register locally before the first await so concurrent tasks see this request.\nself.output_processor.add_request(request, prompt, parent_req, index, queue)\n\n# Add the EngineCoreRequest to EngineCore (separate process).\nawait self.engine_core.add_request_async(request)",
          explanation:
            "本地先注册输出状态，再 await 向 EngineCore 添加请求。源码明确说明 EngineCore 是独立进程。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "engine",
        name: "EngineCore",
        responsibility:
          "组织一个推理迭代：调度、执行、采样与结果更新。Executor 抽象承接不同部署方式，不应把 EngineCore 直接画成 GPU 内核。",
        inputs: ["EngineCoreRequest"],
        outputs: ["EngineCoreOutputs"],
        relatedTopics: ["prefill", "decode", "batching"],
        sources: [
          {
            id: "vllm-engine",
            title: "vllm/v1/engine/core.py · EngineCore.step",
            url: "https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core.py#L598-L601",
            checkedAt: "2026-10-04",
            revision: "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            path: "vllm/v1/engine/core.py",
            symbol: "EngineCore.step",
            startLine: 598,
          },
        ],
        snippet: {
          kind: "source",
          code: "if not self.scheduler.has_requests():\n    return {}, False\nscheduler_output = self.scheduler.schedule(self._should_throttle_prefills())\nfuture = self.model_executor.execute_model(scheduler_output, non_block=True)",
          explanation:
            "无请求时跳过；有请求时先调度，再让 Executor 执行。图中 Runner 边经 Executor 中介，通信方式取决于部署。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "scheduler",
        name: "Scheduler",
        responsibility:
          "在 Token 预算内让 num_computed_tokens 追赶待计算长度，涵盖 Decode、chunked prefill、前缀命中和投机 Token。",
        inputs: ["等待与运行请求", "Token / cache 预算"],
        outputs: ["SchedulerOutput"],
        relatedTopics: ["batching", "chunked-prefill", "speculation"],
        sources: [
          {
            id: "vllm-scheduler",
            title: "vllm/v1/core/sched/scheduler.py · Scheduler.schedule",
            url: "https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/scheduler.py#L579-L583",
            checkedAt: "2026-10-04",
            revision: "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            path: "vllm/v1/core/sched/scheduler.py",
            symbol: "Scheduler.schedule",
            startLine: 579,
          },
        ],
        snippet: {
          kind: "source",
          code: "req_to_new_blocks: dict[str, KVCacheBlocks] = {}\nnum_scheduled_tokens: dict[str, int] = {}\ntoken_budget = self.max_num_scheduled_tokens\nspec = self.vllm_config.speculative_config\ndraft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0",
          explanation:
            "预算以本轮计划 Token 数计，不是固定请求数量。本网站连续批实验隔离了补位规则。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "cache",
        name: "KVCacheManager",
        responsibility:
          "协调请求的缓存组、已计算块与新增槽位，考虑前缀命中及 lookahead。资源不足可返回 None，由调度器处理。",
        inputs: ["Request", "新增 Token 数"],
        outputs: ["KVCacheBlocks / None"],
        relatedTopics: ["kv-cache", "paged-attention"],
        sources: [
          {
            id: "vllm-cache",
            title:
              "vllm/v1/core/kv_cache_manager.py · KVCacheManager.allocate_slots",
            url: "https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/kv_cache_manager.py#L522-L524",
            checkedAt: "2026-10-04",
            revision: "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            path: "vllm/v1/core/kv_cache_manager.py",
            symbol: "KVCacheManager.allocate_slots",
            startLine: 522,
          },
        ],
        snippet: {
          kind: "source",
          code: "required_blocks = num_blocks_to_allocate + watermark_blocks\nif required_blocks > self.block_pool.get_num_free_blocks():\n    return None",
          explanation:
            "检查所需块与空闲块；不足时返回 None。真实混合缓存通过 Coordinator 按缓存组管理。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "blocks",
        name: "BlockPool",
        responsibility:
          "维护空闲队列、块引用与块哈希索引，支持分配、释放和前缀缓存查找；不是 Token 前缀树。",
        inputs: ["块请求与 block hash"],
        outputs: ["物理 KVCacheBlock"],
        relatedTopics: ["paged-attention", "prefix-caching"],
        sources: [
          {
            id: "vllm-blocks",
            title: "vllm/v1/core/block_pool.py · BlockPool",
            url: "https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/block_pool.py#L143-L149",
            checkedAt: "2026-10-04",
            revision: "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            path: "vllm/v1/core/block_pool.py",
            symbol: "BlockPool",
            startLine: 143,
          },
        ],
        snippet: {
          kind: "source",
          code: 'class BlockPool:\n    """BlockPool that manages KVCacheBlocks.\n    It provides methods to allocate, free and cache the kv cache blocks. The\n    free_block_queue stores the free blocks in eviction order to enable\n    allocation, free, and cache eviction. The cached_block_hash_to_block\n    maps between block hash and cached block to support finding cached blocks\n    by their block hash.',
          explanation:
            "注释直接说明空闲队列与 hash→block 映射。注意完整块粒度和引用生命周期。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "runner",
        name: "GPUModelRunner",
        responsibility:
          "把 SchedulerOutput 转成 GPU 输入、缓存索引和执行状态，执行模型并衔接采样；涉及编译、CUDA Graph 与注意力后端。",
        inputs: ["SchedulerOutput", "KV 布局"],
        outputs: ["ModelRunnerOutput / 中间状态"],
        relatedTopics: ["cuda-graphs", "flash-attention", "quantization"],
        sources: [
          {
            id: "vllm-runner",
            title:
              "vllm/v1/worker/gpu_model_runner.py · GPUModelRunner.execute_model",
            url: "https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_model_runner.py#L4187-L4191",
            checkedAt: "2026-10-04",
            revision: "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            path: "vllm/v1/worker/gpu_model_runner.py",
            symbol: "GPUModelRunner.execute_model",
            startLine: 4187,
          },
        ],
        snippet: {
          kind: "source",
          code: 'def execute_model(\n    self,\n    scheduler_output: "SchedulerOutput",\n    intermediate_tensors: IntermediateTensors | None = None,\n) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:',
          explanation:
            "执行器的函数契约允许多类返回值，不能认为每轮都直接返回采样结果。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "model",
        name: "Qwen 混合骨干",
        responsibility:
          "Qwen3_5DecoderLayer 根据 layer_type 选择 Gated DeltaNet 或全注意力，根据 model_type 选择 Dense FFN 或 MoE。两个指定 Qwen 配置的架构类仍名为 Qwen3_5…，此路径因此相关。",
        inputs: ["hidden_states", "位置与缓存状态"],
        outputs: ["更新 hidden_states / logits"],
        relatedTopics: ["gated-deltanet", "moe", "ffn"],
        sources: [
          {
            id: "vllm-model",
            title:
              "vllm/model_executor/models/qwen3_5.py · Qwen3_5DecoderLayer",
            url: "https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/qwen3_5.py#L135-L137",
            checkedAt: "2026-10-04",
            revision: "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            path: "vllm/model_executor/models/qwen3_5.py",
            symbol: "Qwen3_5DecoderLayer",
            startLine: 135,
          },
        ],
        snippet: {
          kind: "source",
          code: 'self.layer_type = layer_type\nself.layer_idx = extract_layer_index(prefix)\nis_moe_layer = config.model_type == "qwen3_5_moe_text"',
          explanation:
            "注意力层类型与 FFN 是否 MoE 是两个独立判定；下方结构页显示官方配置。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
    ],
    edges: [
      {
        to: "client",
        kind: "call",
        label: "经 Serving / 渲染层调用",
        from: "api",
      },
      {
        to: "engine",
        kind: "message",
        label: "EngineCoreRequest / IPC",
        from: "client",
      },
      {
        to: "scheduler",
        kind: "call",
        label: "schedule()",
        from: "engine",
      },
      {
        to: "cache",
        kind: "call",
        label: "allocate_slots()",
        from: "scheduler",
      },
      {
        to: "blocks",
        kind: "call",
        label: "分配 / 释放物理块",
        from: "cache",
      },
      {
        to: "runner",
        kind: "message",
        label: "经 Executor 分发 · 部署相关",
        from: "engine",
      },
      {
        to: "runner",
        kind: "data",
        label: "缓存组 / 块映射 · 折叠视图",
        from: "cache",
      },
      {
        to: "model",
        kind: "call",
        label: "模型 forward",
        from: "runner",
      },
      {
        to: "client",
        kind: "message",
        label: "EngineCoreOutputs / IPC",
        from: "engine",
      },
    ],
  },
  {
    id: "sglang",
    release: "v0.5.21",
    commit: "e00930c5489053f26d86b179cee0d087f846acbb",
    modules: [
      {
        id: "api",
        name: "HTTP Server",
        responsibility:
          "接收 /generate 等 HTTP 请求，把生成交给 TokenizerManager；流式路径以 SSE 逐块返回。此图以常规文本生成与单 HTTP worker 为导览范围。",
        inputs: ["GenerateReqInput / HTTP"],
        outputs: ["JSON / SSE"],
        relatedTopics: ["frameworks", "tokenization"],
        sources: [
          {
            id: "sglang-api",
            title:
              "python/sglang/srt/entrypoints/http_server.py · generate_request",
            url: "https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/entrypoints/http_server.py#L923-L926",
            checkedAt: "2026-10-04",
            revision: "e00930c5489053f26d86b179cee0d087f846acbb",
            path: "python/sglang/srt/entrypoints/http_server.py",
            symbol: "generate_request",
            startLine: 923,
          },
        ],
        snippet: {
          kind: "source",
          code: 'async for out in _global_state.tokenizer_manager.generate_request(\n    obj, request\n):\n    yield b"data: " + dumps_json(out) + b"\\n\\n"',
          explanation:
            "异步迭代 TokenizerManager 的输出并包装成 SSE。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "tokenizer",
        name: "TokenizerManager",
        responsibility:
          "规范化请求、执行分词与输入处理，向 Scheduler 发送对象，跟踪每个请求的输出并回送 HTTP。",
        inputs: ["文本 / 输入参数", "BatchStrOutput"],
        outputs: ["TokenizedGenerateReqInput", "生成响应"],
        relatedTopics: ["tokenization", "sampling"],
        sources: [
          {
            id: "sglang-tokenizer",
            title:
              "python/sglang/srt/managers/tokenizer_manager.py · TokenizerManager._dispatch_to_scheduler",
            url: "https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/managers/tokenizer_manager.py#L648-L651",
            checkedAt: "2026-10-04",
            revision: "e00930c5489053f26d86b179cee0d087f846acbb",
            path: "python/sglang/srt/managers/tokenizer_manager.py",
            symbol: "TokenizerManager._dispatch_to_scheduler",
            startLine: 648,
          },
        ],
        snippet: {
          kind: "source",
          code: "def _dispatch_to_scheduler(self, obj: Any) -> None:\n    if self.tokenizer_ipc_name is not None:\n        stamp_http_worker_ipc(obj, self.tokenizer_ipc_name)\n    sock_send(self.send_to_scheduler, obj)",
          explanation:
            "经 socket 发送到 Scheduler，属于进程消息，不是普通对象方法调用。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "scheduler",
        name: "Scheduler",
        responsibility:
          "接收请求、选择下一批并执行、处理结果；可使用重叠调度等其他循环。这里链接 normal loop，不冒充覆盖全部路径。",
        inputs: ["TokenizedGenerateReqInput", "运行批状态"],
        outputs: ["ScheduleBatch / 生成 Token"],
        relatedTopics: ["batching", "prefill", "decode"],
        sources: [
          {
            id: "sglang-scheduler",
            title:
              "python/sglang/srt/managers/scheduler.py · Scheduler.event_loop_normal",
            url: "https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/managers/scheduler.py#L1918-L1923",
            checkedAt: "2026-10-04",
            revision: "e00930c5489053f26d86b179cee0d087f846acbb",
            path: "python/sglang/srt/managers/scheduler.py",
            symbol: "Scheduler.event_loop_normal",
            startLine: 1918,
          },
        ],
        snippet: {
          kind: "source",
          code: "# Get the next batch to run\nplan = self.get_next_batch_to_run(\n    running_batch=self.running_batch, last_batch=self.last_batch\n)\nself.running_batch = plan.running_batch\nbatch = plan.batch_to_run",
          explanation:
            "下一批计划还更新 running_batch；真实实现的调度状态比固定槽位实验更丰富。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "cache",
        name: "RadixCache",
        responsibility:
          "用 Radix 压缩前缀结构匹配最长缓存前缀，支持插入、锁引用和驱逐。Token 序列之外的 extra_key 分隔命名空间；不同缓存后端还需不同状态策略。",
        inputs: ["RadixKey / extra_key"],
        outputs: ["匹配的缓存位置与节点"],
        relatedTopics: ["prefix-caching", "paged-attention", "gated-deltanet"],
        sources: [
          {
            id: "sglang-cache",
            title:
              "python/sglang/srt/mem_cache/radix_cache.py · RadixCache.match_prefix",
            url: "https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/mem_cache/radix_cache.py#L402-L406",
            checkedAt: "2026-10-04",
            revision: "e00930c5489053f26d86b179cee0d087f846acbb",
            path: "python/sglang/srt/mem_cache/radix_cache.py",
            symbol: "RadixCache.match_prefix",
            startLine: 402,
          },
        ],
        snippet: {
          kind: "source",
          code: "value, last_node = self._match_prefix_helper(self.root_node, key)\nif value:\n    value = torch.cat(value)\nelse:\n    value = self._empty_match_result.device_indices",
          explanation:
            "匹配 helper 返回位置列表与终止节点，有匹配时拼接位置；并不是 vLLM 的块哈希索引。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "runner",
        name: "ModelRunner",
        responsibility:
          "从 ForwardBatch 执行模型，衔接注意力后端、图重放与输出。Scheduler 经 TpModelWorker 等中间层到达 Runner，图中作职责层面的折叠。",
        inputs: ["ForwardBatch / 状态"],
        outputs: ["ModelRunnerOutput"],
        relatedTopics: ["cuda-graphs", "attention", "quantization"],
        sources: [
          {
            id: "sglang-runner",
            title:
              "python/sglang/srt/model_executor/model_runner.py · ModelRunner.forward",
            url: "https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/model_executor/model_runner.py#L1753-L1758",
            checkedAt: "2026-10-04",
            revision: "e00930c5489053f26d86b179cee0d087f846acbb",
            path: "python/sglang/srt/model_executor/model_runner.py",
            symbol: "ModelRunner.forward",
            startLine: 1753,
          },
        ],
        snippet: {
          kind: "source",
          code: "output = self._forward_raw(\n    forward_batch,\n    pp_proxy_tensors,\n    reinit_attn_backend,\n    split_forward_count,\n)",
          explanation:
            "真正前向下沉到 _forward_raw，周围还有 profiling、专家分布与重平衡。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "model",
        name: "Qwen 混合骨干",
        responsibility:
          "从 config.layers_block_type 选择 Attention 或 LinearDecoderLayer。官方配置 architectures 为 Qwen3_5ForConditionalGeneration 或 MoE 变体，类名与模型发布名不同。",
        inputs: ["hidden_states", "ForwardBatch"],
        outputs: ["logits / 更新状态"],
        relatedTopics: ["gated-deltanet", "moe", "ffn"],
        sources: [
          {
            id: "sglang-model",
            title:
              "python/sglang/srt/models/qwen3_5.py · Qwen3_5ForCausalLM.get_layer",
            url: "https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/models/qwen3_5.py#L1674-L1677",
            checkedAt: "2026-10-04",
            revision: "e00930c5489053f26d86b179cee0d087f846acbb",
            path: "python/sglang/srt/models/qwen3_5.py",
            symbol: "Qwen3_5ForCausalLM.get_layer",
            startLine: 1674,
          },
        ],
        snippet: {
          kind: "source",
          code: "# Decoder layers\ndef get_layer(idx: int, prefix: str):\n    layer_type = config.layers_block_type[idx]\n    layer_class = self.decoder_layer_types[layer_type]",
          explanation:
            "构造层时按类型选不同实现；不把所有层都解释成 softmax Attention。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
      {
        id: "detokenizer",
        name: "DetokenizerManager",
        responsibility:
          "接收 Token ID输出、维护增量解码状态，转成字符串并回送 Tokenizer侧；消息路径会随多 HTTP worker 配置改变。",
        inputs: ["BatchTokenIDOutput"],
        outputs: ["BatchStrOutput"],
        relatedTopics: ["tokenization", "decode"],
        sources: [
          {
            id: "sglang-detokenizer",
            title:
              "python/sglang/srt/managers/detokenizer_manager.py · DetokenizerManager.handle_batch_token_id_out",
            url: "https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/managers/detokenizer_manager.py#L454-L458",
            checkedAt: "2026-10-04",
            revision: "e00930c5489053f26d86b179cee0d087f846acbb",
            path: "python/sglang/srt/managers/detokenizer_manager.py",
            symbol: "DetokenizerManager.handle_batch_token_id_out",
            startLine: 454,
          },
        ],
        snippet: {
          kind: "source",
          code: "output_strs = (\n    self._decode_batch_token_id_output(recv_obj)\n    if len(recv_obj.rids) > 0\n    else []\n)",
          explanation:
            "空闲批返回空字符串列表；有请求时执行批量反分词，不是简单拼接词表片段。 片段保留源码语句，仅去掉共同缩进；并非完整函数。",
        },
      },
    ],
    edges: [
      {
        to: "tokenizer",
        kind: "call",
        label: "generate_request()",
        from: "api",
      },
      {
        to: "scheduler",
        kind: "message",
        label: "TokenizedGenerateReqInput / socket",
        from: "tokenizer",
      },
      {
        to: "cache",
        kind: "call",
        label: "前缀匹配 / 缓存管理",
        from: "scheduler",
      },
      {
        to: "runner",
        kind: "call",
        label: "经 TpModelWorker 等中介",
        from: "scheduler",
      },
      {
        to: "runner",
        kind: "data",
        label: "缓存位置 / 状态 · 折叠视图",
        from: "cache",
      },
      {
        to: "model",
        kind: "call",
        label: "模型 forward",
        from: "runner",
      },
      {
        to: "detokenizer",
        kind: "message",
        label: "BatchTokenIDOutput / IPC",
        from: "scheduler",
      },
      {
        to: "tokenizer",
        kind: "message",
        label: "BatchStrOutput / IPC",
        from: "detokenizer",
      },
    ],
  },
];
