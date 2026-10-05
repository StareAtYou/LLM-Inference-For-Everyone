# 推理图谱第一版 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个可运行、视觉完成度高、无小练习的中文大模型推理学习网站，包含两个 Qwen 模型的结构示例、五个机制实验和 SGLang/vLLM 源码导览。

**Architecture:** React 页面读取结构化学习内容和固定版本模型配置；纯 TypeScript 模拟函数产生状态，SVG/HTML 组件显示同一份状态。路由按页加载，本地保存浏览与收藏，所有教学模拟/理论估算带明确来源标签。为后续实测保留观测值来源与单位字段，不构建 GPU 后端。

**Tech Stack:** React、TypeScript、Vite、React Router、CSS、SVG；Vitest 验证模拟与解析逻辑，Playwright 验证主要流程和截图。当前 Node v22.22.0、npm 10.9.4；安装时选择兼容的稳定依赖并提交 package-lock.json。

**Spec:** `docs/superpowers/specs/2026-10-04-inference-atlas-design.md`（用户已批准，包含随后指定的 Qwen 模型）。

## Global Constraints

- 第一版为 A：浏览器内的学习与机制模拟；以后逐步扩展到 B：真实模型服务与 GPU 实验。
- 不包含小练习、测验、评分或答题任务。
- 中文讲解为主，保留英文术语和代码；浏览记录只代表访问过，不声称掌握程度。
- 视觉方向为 A：技术出版物 × 交互实验室，浅色阅读界面配深色计算演示画布。
- 真实预设的层数、维度与 head 配置只读；修改结构参数会进入“自定义教学配置”。
- Qwen3.8-27B：revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；Qwen3.6-35B-A3B：revision `995ad96eacd98c81ed38be0c5b274b04031597b0`。
- vLLM v0.30.0 commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；SGLang v0.5.21 commit `e00930c5489053f26d86b179cee0d087f846acbb`。
- 不下载模型权重、不要求 API Key；固定 Token、激活和输出示例标记为教学数据。
- 缓存显示全注意力 KV 分项；线性层状态单独解释，不把分项写成完整运行时显存。
- 六条主要路由 `/`、`/learn`、`/pipeline`、`/models`、`/lab/:experiment`、`/frameworks/:framework`；专题 `/learn/:topic`。
- 深度参数 `depth=beginner|advanced|expert`；框架模块参数 `module=模块ID`；分享参数采用白名单校验。
- 至少检查 390px、768px、1440px 三种宽度，并补查 360px、1920px 边界。
- 尊重 `prefers-reduced-motion`；离屏和后台页面暂停持续动画；当前交付为本地可运行项目与预览。

## Review Focus

1. URL 有重复、非法或极大参数：得到有限、合法的配置或带提示的默认值，不崩溃、不静默伪造合法模型。
2. localStorage 禁用、损坏或包含旧数据：保留当前会话的导航、收藏与深度选择，合理恢复默认值。
3. 播放期间切换模型、实验或路由：旧定时器停止，状态重置，返回后没有多个时钟叠加。
4. 低温采样、空输入、零权重和资源不足：确定性结果或可解释错误，数字无 NaN/Infinity，不出现非法分配。
5. 手机、键盘和减弱动效：核心控件可操作，图形有文字说明，页面无横向溢出且导航不会锁住焦点。

## 文件与接口地图

新项目，当前只有 README 和设计资料。所有路径相对于仓库根；每个页面/实验独立文件，避免单个 App 包含所有内容与算法。

| 文件 | 责任 |
| --- | --- |
| `package.json`, `package-lock.json`, `index.html`, `tsconfig.json`, `vite.config.ts`, `.gitignore` | 本地启动、构建、类型检查、测试与依赖锁定 |
| `src/main.tsx`, `src/App.tsx`, `src/types.ts` | 应用入口、路由、共享数据契约 |
| `src/components/SiteShell.tsx`, `DepthSwitch.tsx`, `SourceNote.tsx`, `PlaybackControls.tsx`, `ParameterControl.tsx` | 导航、层级、来源、播放与带标签参数控件 |
| `src/styles/tokens.css`, `global.css`, `pages.css`, `visualizations.css` | 字体/色彩/间距、整体排版、页面和图形样式 |
| `src/content/topics.ts`, `models.ts`, `scenarios.ts`, `frameworks.ts` | 专题、真实模型预设、教学场景、固定源码导览 |
| `src/lib/preferences.ts`, `urlState.ts`, `search.ts` | 安全存储、分享解析、术语检索 |
| `src/hooks/usePreferences.ts`, `usePlayback.ts` | 跨页本地状态、统一播放时钟 |
| `src/simulation/attention.ts`, `sampling.ts`, `pipeline.ts`, `memory.ts`, `paging.ts`, `prefix.ts`, `batching.ts`, `quantization.ts`, `speculation.ts`, `experts.ts` | 各自独立的计算逻辑 |
| `src/pages/Home.tsx`, `Learn.tsx`, `Topic.tsx`, `Pipeline.tsx`, `Models.tsx`, `Lab.tsx`, `Frameworks.tsx`, `NotFound.tsx` | 页面组织，按页延迟加载 |
| `src/visualizations/RequestFlow.tsx`, `AttentionMatrix.tsx`, `ModelGraph.tsx`, `ExpertRouter.tsx`, `CacheBlocks.tsx`, `PrefixTree.tsx`, `BatchTimeline.tsx`, `QuantizationPlot.tsx`, `SpeculationTrace.tsx`, `FrameworkGraph.tsx` | 消费状态的图形，图例和可访问说明 |
| `src/experiments/KvCache.tsx`, `PagedAttention.tsx`, `ContinuousBatching.tsx`, `Quantization.tsx`, `SpeculativeDecoding.tsx` | 五个可操作实验与参数面板 |
| `tests/unit/*.test.ts`, `tests/e2e/*.spec.ts`, `playwright.config.ts` | 算法/恢复逻辑测试、用户流程与视觉检查 |
| `docs/quality/v1-review.md`, `docs/quality/screenshots/`, `README.md` | 自检证据、截图、使用说明与明确限制 |

共享类型在 `src/types.ts` 定义：`Depth = 'beginner' | 'advanced' | 'expert'`；`ModelId = 'qwen38-dense' | 'qwen36-moe' | 'teaching'`；`SourceRef = {id, title, url, checkedAt, revision?, path?, symbol?, startLine?}`；`ModelPreset = {id: ModelId, name, family: 'dense'|'moe', hiddenSize, headDim, qHeads, kvHeads, layerTypes: ('linear_attention'|'full_attention')[], ffnSize?, expertCount?, activeExperts?, sharedExperts?, linearKeyHeads, linearValueHeads, linearHeadDim, sources: SourceRef[]}`；`Topic = {id, title, englishTerms: string[], area, prerequisites: string[], levels: Record<Depth, {summary, explanation, details: string[]}>, links: {label, to}[], sources: SourceRef[]}`；`Preferences = {version: 1, depth: Depth, bookmarks: string[], recent: string[]}`；`Observation = {label, value, unit, source: 'simulation'|'estimate'|'measured', assumptions: string[]}`。这些对象中未另标的文本字段为string；维度、数量、行号和Observation.value为number。

数值参数用 number，参数合法性在模拟入口校验。错误返回可辨识的错误信息，由页面捕获显示；不吞掉算法错误。下面各任务定义新增返回类型，和函数放在同一模拟模块并 export。

## Task 1：可运行的设计系统、导航与首页

**Files:** 创建工程配置、`src/main.tsx`、`src/App.tsx`、`src/types.ts`、`src/components/SiteShell.tsx`、四个样式文件、`src/pages/Home.tsx`、`NotFound.tsx`、`src/visualizations/RequestFlow.tsx`、`playwright.config.ts`、`tests/e2e/navigation.spec.ts`。

**Interfaces:** 产出 `SiteShell({children: ReactNode})`、`RequestFlow({stage?: number, compact?: boolean})` 和 `App()`；路由按文件懒加载，尚未实现的页面在本任务验证时不作为已交付功能宣传。

- [ ] 建立 npm 启动/类型检查/构建/单元测试/E2E 脚本及兼容依赖锁；保留现有 README/设计资料。创建真实路由外壳，用 `npm run build` 确认工具链可运行。
- [ ] 在 `navigation.spec.ts` 写失败用例 `home_navigation_keyboard_mobile`：首页有定位标题和开始探索链接；390px 下导航可展开、Escape 关闭；Tab 焦点可见；未知路由有返回首页链接。
- [ ] `npx playwright test tests/e2e/navigation.spec.ts`，确认失败来自尚未实现行为。
- [ ] 实现暖白纸面、近黑正文、紫色重点、原创推理 SVG、大字排版、三条路径和章节布局。配置系统字体/稳定回退，首页图形满足减弱动效，不放虚构仪表数字。
- [ ] 同一 E2E 用例与 `npm run typecheck && npm run build` 通过；在1440px、390px截屏查看，提交 `feat: establish inference atlas shell and home`。

## Task 2：学习目录、实际模型配置、本地状态与可分享导航

**Files:** 创建 `src/content/topics.ts`、`models.ts`、`src/lib/preferences.ts`、`urlState.ts`、`search.ts`、`src/hooks/usePreferences.ts`、`src/components/DepthSwitch.tsx`、`SourceNote.tsx`、`src/pages/Learn.tsx`、`Topic.tsx`、`tests/unit/catalog.test.ts`、`preferences.test.ts`、`urlState.test.ts`、`tests/e2e/learning.spec.ts`；修改 `App.tsx`、`Home.tsx`。

**Interfaces:** `models: ModelPreset[]`、`topics: Topic[]`；`searchTopics(items: Topic[], query: string, depth: Depth, area?: string): Topic[]`；`readPreferences(storage: Pick<Storage,'getItem'>): Preferences`；`writePreferences(storage: Pick<Storage,'setItem'>, value: Preferences): boolean`；`parseRouteState(params: URLSearchParams): {depth: Depth, modelId: ModelId, warning?: string}`；`usePreferences(): {preferences: Preferences, setDepth(depth: Depth): void, toggleBookmark(id: string): void, recordVisit(id: string): void}`。

- [ ] 写失败测试：Qwen Dense `layerTypes.length=64`、full=16、hidden=5120、headDim=256；MoE长度40、full=10、experts=256、active=8、shared=1。损坏 JSON/抛错 storage 返回默认值；重复/非法模型和深度 URL 有明确回退。目录无重复 ID、不存在的前置知识或空层级。

  核心断言示例：`expect(models.find(m => m.id === 'qwen38-dense')?.layerTypes.filter(t => t === 'full_attention')).toHaveLength(16)`；`expect(readPreferences({getItem: () => '{'}).bookmarks).toEqual([])`；`expect(parseRouteState(new URLSearchParams('model=unknown')).warning).toBeTruthy()`。URL参数名称统一为`model`，值使用ModelId。
- [ ] `npx vitest run tests/unit/catalog.test.ts tests/unit/preferences.test.ts tests/unit/urlState.test.ts`，确认测试对缺失行为失败。
- [ ] 从 `docs/references/models/` 提取实际配置与来源。建立至少20个有实质三级解释的专题，覆盖规格第3节全部概念：Token/Embedding、Attention/RoPE/FFN、GQA/Gated DeltaNet/MoE、Prefill/Decode/采样、KV/分页/前缀、连续批处理/量化/投机，以及FlashAttention/并行/chunked prefill/融合/CUDA Graph/性能指标概览；允许多个相关概念共用一专题。
- [ ] 实现目录搜索、深度/领域筛选、知识依赖、收藏/最近访问与详情页。学习资料不以只有目录或空标题的页冒充交付；保存失败仍在内存中维持当前会话。分享参数用白名单，未知专题显示可恢复空状态。
- [ ] 单元测试通过；E2E `learning_navigation_and_recovery` 验证深度文本切换、英文术语搜索、刷新后收藏、存储禁用时继续阅读、直接打开专题链接。构建通过，提交 `feat: add learning atlas and grounded Qwen presets`。

## Task 3：推理流程、采样与统一播放

**Files:** 创建 `src/content/scenarios.ts`、`src/simulation/attention.ts`、`sampling.ts`、`pipeline.ts`、`src/hooks/usePlayback.ts`、`src/components/PlaybackControls.tsx`、`ParameterControl.tsx`、`src/pages/Pipeline.tsx`、`src/visualizations/AttentionMatrix.tsx`、`tests/unit/inference.test.ts`、`tests/e2e/pipeline.spec.ts`；更新 `RequestFlow.tsx`。

**Interfaces:** `softmax(logits: number[], temperature: number): number[]`（temperature=0 为确定性argmax）；`filterDistribution(probs: number[], topK: number, topP: number): number[]`；`causalAttention(scores: number[][]): number[][]`；`buildPipelineFrames(modelId: ModelId, scenarioId: string, sampling: {temperature: number, topK: number, topP: number}): PipelineFrame[]`，`PipelineFrame={stage, inputTokens: string[], outputTokens: string[], cacheTokens, selectedLayer?, description, observations: Observation[]}`；`usePlayback({frameCount, intervalMs, resetKey}: {frameCount: number, intervalMs: number, resetKey: string}): {index, playing, play(): void, pause(): void, step(): void, reset(): void, seek(index: number): void}`。

- [ ] 写失败测试 `causal_mask_never_sees_future`、`sampling_normalizes_filtered_distribution`、`zero_temperature_returns_argmax`：矩阵上三角为0、每行和为1；top-k=1只剩最大项；极大 logits 不溢出；空 logits 返回可解释错误。`first_token_comes_from_prefill` 验证教学帧首输出来自Prefill。

  核心断言：`expect(causalAttention([[1, 2], [3, 4]])[0]).toEqual([1, 0])`；`expect(softmax([1, 3, 2], 0)).toEqual([0, 1, 0])`；`expect(filterDistribution([0.1, 0.7, 0.2], 1, 1)).toEqual([0, 1, 0])`。
- [ ] `npx vitest run tests/unit/inference.test.ts` 确认失败。
- [ ] 实现纯模拟和至少三个固定场景；输入Token/ID与概率明确为教学数据。两个Qwen预设决定真实拓扑和形状，数值激活/输出不冒充真实推理。实现输入→分词→Embedding→Prefill首输出→Decode循环→停止的帧，配原因解释和来源标签。
- [ ] 实现播放/暂停/单步/跳转/重置；切模型、场景与采样参数重置时钟；document.hidden和画布离屏时暂停；减弱动效默认静态且仍可单步。
- [ ] 单元通过；E2E `pipeline_clock_is_single_and_resettable`：暂停后Token不增长、重置恢复初态、切模型不残留旧状态、离开页面时钟停止。类型/构建通过，提交 `feat: visualize prefill decode and sampling`。

## Task 4：真实混合网络与 Dense/MoE 结构探索

**Files:** 创建 `src/pages/Models.tsx`、`src/visualizations/ModelGraph.tsx`、`ExpertRouter.tsx`、`src/simulation/experts.ts`、`tests/unit/models.test.ts`、`tests/e2e/models.spec.ts`；消费 task2 的模型目录和task3的Attention矩阵。

**Interfaces:** `routeExperts(logits: number[], activeCount: number): {id: number, weight: number}[]`（top-k 后重新归一化，稳定处理同值）；`ModelGraph({model: ModelPreset, selectedLayer: number, onSelect(index: number): void})`；`ExpertRouter({routed: {id: number, weight: number}[], expertCount: number, sharedExperts: number})`。

- [ ] 写失败测试：MoE选出8个不同路由专家、权重和为1；共享专家另计；非法activeCount报错。结构每第四层为全注意力；Dense Q形状使用24×256，不用5120/24反推head维度。

  核心断言：`expect(routeExperts([1, 4, 2, 3], 2).map(x => x.id)).toEqual([1, 3])`；`expect(() => routeExperts([1], 2)).toThrow()`。
- [ ] `npx vitest run tests/unit/models.test.ts` 确认失败。
- [ ] 实现完整层序列与可展开Block；真实预设结构只读。层选择联动Gated DeltaNet固定状态/全注意力KV、Norm、门控、残差和Dense FFN/MoE；显示8 routed + 1 shared。真实模型Vision分支标为文本路径未启用，MTP说明关联投机专题。
- [ ] 添加缩小教学配置、RoPE二维旋转、GQA共享对比和Attention矩阵。改变结构进入自定义教学状态，形状与标注同步，Gated DeltaNet数值更新明确为示意。
- [ ] 单元通过；E2E `model_switch_updates_structure_and_experts` 检查64→40层、Dense→MoE、层类型和专家计数、回到真实预设后参数只读。构建通过，提交 `feat: explore Qwen hybrid dense and MoE networks`。

## Task 5：KV、分页和前缀复用实验

**Files:** 创建 `src/pages/Lab.tsx`、`src/simulation/memory.ts`、`paging.ts`、`prefix.ts`、`src/experiments/KvCache.tsx`、`PagedAttention.tsx`、`src/visualizations/CacheBlocks.tsx`、`PrefixTree.tsx`、`tests/unit/cache.test.ts`、`tests/e2e/cache.spec.ts`；扩展 `urlState.ts`。

**Interfaces:** `estimateFullAttentionKvBytes(model: ModelPreset, tokens: number, batch: number, bytesPerElement: number): number`；`createPagedState(capacity: number, blockSize: number): PagedState`；`stepPaged(state: PagedState, event: {type:'allocate', id:string, tokens:number}|{type:'release', id:string}): PagedState`；`PagedState={capacity, blockSize, requests: Record<string,{tokens:number, blocks:number[]}>, freeBlocks:number[]}`；`buildPrefixTree(sequences: string[][]): PrefixNode`，`PrefixNode={token:string, count:number, children:PrefixNode[]}`；在`urlState.ts`增加`parseExperimentParams(experiment: string, params: URLSearchParams): {values: Record<string, number|string>, warning?: string}`、`serializeExperimentParams(experiment: string, values: Record<string, number|string>): URLSearchParams`。实验组件零参数读取经过校验的URL/局部状态。

- [ ] 写失败测试：Dense在1024 tokens、batch1、2bytes时KV分项=67108864；MoE=20971520。blockSize4，长度5需要2块；释放后可复用；占用与空闲不重叠且覆盖capacity；不足时明确失败且输入状态未修改；相同前缀形成共享节点。

  核心断言：`expect(estimateFullAttentionKvBytes(models.find(m => m.id === 'qwen38-dense')!, 1024, 1, 2)).toBe(67108864)`；`expect(stepPaged(createPagedState(8, 4), {type:'allocate', id:'a', tokens:5}).requests.a.blocks).toHaveLength(2)`。
- [ ] `npx vitest run tests/unit/cache.test.ts` 确认失败。
- [ ] 实现公式、MiB/GiB、标准教学层数与Qwen全注意力分项模式，线性状态单独说明。分页显示逻辑/物理映射、尾块浪费、分配/释放与OOM反馈；前缀图对应vLLM块哈希和SGLang Radix概念。
- [ ] 实现可访问参数控件与配置分享；tokens限定1..262144、batch1..64、blockSize1..64、capacity1..128、bytes只接受0.5/1/2/4；重复参数或非法范围回退并提示。教学分页池与真实模型字节估算分开标注。
- [ ] 单元通过；E2E `cache_share_release_and_invalid_parameters` 检查链接恢复配置、篡改URL回退、分配/释放、来源标签和分项名称。构建通过，提交 `feat: add KV cache paging and prefix experiments`。

## Task 6：连续批处理实验

**Files:** 创建 `src/simulation/batching.ts`、`src/experiments/ContinuousBatching.tsx`、`src/visualizations/BatchTimeline.tsx`、`tests/unit/batching.test.ts`；修改 `Lab.tsx`、`scenarios.ts`、`urlState.ts`。

**Interfaces:** `simulateBatching(requests: {id:string, arrival:number, outputTokens:number}[], capacity:number, mode:'static'|'continuous'): BatchResult`；`BatchResult={frames:{tick:number, slots:(string|null)[]}[], completions:Record<string,number>, busySlots:number, totalSlots:number}`。固定成本模型每Decode步一模拟单位，Prefill成本公开说明并保持两个模式一致。

- [ ] 写失败测试：任何request不得在arrival之前进入槽位；每tick同request最多一槽；未完成输出不得标记完成；两个模式用相同请求；capacity=1且所有arrival=0时结果一致；空请求集返回有限的零占用结果。

  核心断言：`expect(simulateBatching([], 2, 'continuous').busySlots).toBe(0)`；对`{id:'late',arrival:3,outputTokens:2}`断言所有`tick<3`的frame.slots不包含`late`。静态/连续对比额外检验`completions`有且只有输入request ID。
- [ ] `npx vitest run tests/unit/batching.test.ts` 确认失败。
- [ ] 实现静态批等待整批完成、连续批在空位纳入等待请求的事件模拟；同一时间刻度的双时间线、利用率及完成顺序从BatchResult导出。提供不同到达和长短输出预设，不把模拟单位显示为GPU实测ms。
- [ ] 参数变化/重置接入task3播放控件；为无性能提升的场景展示解释，不强行给加速结论。
- [ ] 测试通过、浏览器确认时间线随单步同步变化、构建通过，提交 `feat: simulate static and continuous batching`。

## Task 7：量化实验

**Files:** 创建 `src/simulation/quantization.ts`、`src/experiments/Quantization.tsx`、`src/visualizations/QuantizationPlot.tsx`、`tests/unit/quantization.test.ts`；修改 `Lab.tsx`、`urlState.ts`。

**Interfaces:** `quantizeSymmetric(weights:number[], bits:4|8): QuantizedVector`；`QuantizedVector={scale:number, codes:number[], reconstructed:number[], meanSquaredError:number, payloadBytes:number}`；第一版使用对称范围 ±(2^(bits-1)-1)，说明保留一个码点的教学简化。

- [ ] 写失败测试：[-1,0,1]在INT4映射[-7,0,7]；scale=1/7；误差为0；全零输入不除零；空输入/非有限输入明确报错；payloadBytes=ceil(length×bits/8)，注明元数据未计入。

  核心断言：`expect(quantizeSymmetric([-1, 0, 1], 4).codes).toEqual([-7, 0, 7])`；`expect(quantizeSymmetric([0, 0], 4).meanSquaredError).toBe(0)`；`expect(quantizeSymmetric([-1, 0, 1], 4).payloadBytes).toBe(2)`。
- [ ] `npx vitest run tests/unit/quantization.test.ts` 确认失败。
- [ ] 实现原值→量化码→反量化值的联动图与误差；FP16作为存储基准，注明图中原值不承担完整IEEE FP16舍入模拟。使用固定/可选权重场景，INT4/INT8切换改变实际计算。
- [ ] 展示存储收益、误差、scale和假设；不由位宽推断端到端速度，不声称教学量化等同所有实际格式。
- [ ] 测试/类型/构建通过，浏览器检查负值、全零与不同位宽的图形，提交 `feat: visualize quantization and reconstruction error`。

## Task 8：投机解码实验

**Files:** 创建 `src/simulation/speculation.ts`、`src/experiments/SpeculativeDecoding.tsx`、`src/visualizations/SpeculationTrace.tsx`、`tests/unit/speculation.test.ts`；修改 `Lab.tsx`、`scenarios.ts`、`urlState.ts`。

**Interfaces:** `simulateSpeculation(targetTokens:string[], draftTokens:string[], draftLength:number, costs:{draft:number, verify:number}): SpeculationResult`；`SpeculationResult={rounds:{draft:string[], accepted:string[], rejected:string[], correction?:string}[], output:string[], baselineCost:number, speculativeCost:number}`。每轮按当前已接受目标前缀提取预设草稿；全接受时可由目标模型补一个Token，首次拒绝后丢弃余下草稿。

- [ ] 写失败测试：草稿[a,x,c]目标[a,b,c]只接受a、修正b，不能接受拒绝后的c；全匹配/全拒绝都产生完整且等于目标的greedy输出；高草稿成本可使speedup<1；成本非正或草稿长度0报错。

  核心断言：`const result = simulateSpeculation(['a','b','c'], ['a','x','c'], 3, {draft:1, verify:2})`；`expect(result.rounds[0].accepted).toEqual(['a'])`；`expect(result.rounds[0].correction).toBe('b')`；`expect(result.output).toEqual(['a','b','c'])`。
- [ ] `npx vitest run tests/unit/speculation.test.ts` 确认失败。
- [ ] 实现逐轮草稿/目标验证/接受/拒绝/修正，和普通Decode对比。提供全接受、部分接受、低接受率三类固定场景，接受率由轨迹导出而非用户直接指定伪造结果。
- [ ] 明确这是greedy教学验证，展示MTP/独立草稿的概念入口；不暗示已经执行两个Qwen的真实MTP或概率校正算法。
- [ ] 测试通过、浏览器确认首次拒绝后图形和输出一致、构建通过，提交 `feat: demonstrate greedy speculative decoding`。

## Task 9：SGLang/vLLM 源码与模块导览

**Files:** 创建 `src/content/frameworks.ts`、`src/pages/Frameworks.tsx`、`src/visualizations/FrameworkGraph.tsx`、`tests/unit/frameworks.test.ts`、`tests/e2e/frameworks.spec.ts`；修改topics关联和首页框架入口。

**Interfaces:** `FrameworkModule={id, name, responsibility, inputs:string[], outputs:string[], relatedTopics:string[], sources:SourceRef[], snippet:{kind:'source'|'pseudocode', code:string, explanation:string} }`；`FrameworkDefinition={id:'vllm'|'sglang', release, commit, modules:FrameworkModule[], edges:{from,to,kind:'call'|'message'|'data',label:string}[]}`；`frameworks: FrameworkDefinition[]`。

- [ ] 读取规格第4节的12个快照源码路径，以及两个框架对应Qwen3_5模型实现。核对模块职责/符号/调用或通信类型，记录短片段的准确行号、来源与许可证。固定commit链接和官方文档分开记录，不从main推断快照行为。
- [ ] 写失败测试：每框架至少6模块；图边和topic引用都存在；source片段有commit/path/符号；permalink含完整commit；两框架的缓存实现解释不混淆。`npx vitest run tests/unit/frameworks.test.ts` 确认对缺失数据失败。
- [ ] 实现请求→调度→执行/缓存→输出的图形、模块选择、输入输出与代码解说；清楚区分实际源码片段与教学伪代码、直接调用与进程通信。
- [ ] 加入Qwen文本骨干的实现关联：混合层类型、Dense FFN/MoE、缓存类别；源码文字以核对结果为准，不靠版本名猜测支持详情。切框架同步更新图、说明、链接和module查询参数，非法module可恢复。
- [ ] 单元通过；E2E `framework_modules_are_versioned_and_shareable` 验证框架切换、模块深链刷新、原理跳转和commit链接；构建通过，提交 `feat: connect inference concepts to framework source`。

## Task 10：全站视觉、交互、内容与响应式验收

**Files:** 创建 `tests/e2e/quality.spec.ts`、`docs/quality/v1-review.md`、`docs/quality/screenshots/`；修改发现问题所属的页面、样式或模拟文件。

**Interfaces:** 消费前九项已交付接口；质量报告记录检查的页面、状态、宽度、问题、修复与复查结果，不以“截图存在”替代审美审查。

- [ ] 补充失败/边界用例：直接加载与刷新六个主要页面和专题；暂停后切路由、极端有效参数、本地存储抛错、无搜索结果、减弱动效、键盘控制。检查无运行错误和NaN/Infinity；避免测试外部GitHub的瞬时可用性。
- [ ] 用Playwright在390/768/1440宽度捕获全部主要页面，补360/1920检查；批处理/框架/长代码和MoE图分别检查重要展开状态。用图像查看工具实际查看截图，不只读DOM。
- [ ] 按规格七项视觉/交互标准审查：字号/行长/留白、区域层级、深浅画布节奏、图例/颜色、动效、微交互、移动布局和原创性；逐项写具体发现。检查正文对比度、44px主要点击目标、200%文本放大与无页面横向溢出。
- [ ] 修复发现的具体问题并复查受影响页面；检查字体失败回退、离屏/后台暂停和按页加载。公开说明测试模拟与源码范围；全站检索移除假数字、无功能入口与练习/成绩设计。
- [ ] `npm run typecheck && npm run test:unit && npm run build && npm run test:e2e` 全部通过后提交 `fix: refine responsive visuals and interaction quality`；保留检查证据，不承诺已获得设计奖项。

## Task 11：可运行交付、文档与最终复核

**Files:** 更新 `README.md`、`docs/quality/v1-review.md`；必要时添加 `docs/architecture.md`。

**Interfaces:** 启动命令 `npm install`、`npm run dev`；生产构建 `npm run build`，预览 `npm run preview`；文档解释静态托管需SPA路由回退，未执行公开部署。

- [ ] README写清使用方式、目录、三个学习深度、两个Qwen的真实结构/教学数值边界、五个实验、源码快照、验证命令及B阶段路线。注明localStorage/来源单位/非实测指标，不复制大量外部文档。
- [ ] 最终核对规格覆盖：首页、地图、专题、流程、模型、五个实验、两框架、来源、收藏、深链、响应式、减弱动效和无练习；报告仍未交付的长期主题，无空白页面混入范围。
- [ ] 以生产预览验证资源与主要流程，查看git diff/状态，确认没有权重、临时凭据、缓存或无关文件进入提交。若新增修改影响验证，重新运行对应检查。
- [ ] 根据用户选定的执行方式完成独立最终代码审查；修复实际问题并复验，不进行未授权公开发布。提交 `docs: document inference atlas v1 delivery`。
- [ ] 打开本地预览供用户查看，最终给出启动/预览入口、已实现范围、验证证据与后续B的明确边界。

## 计划自审与执行交接

上述任务覆盖规格第1–8节。Review Focus 1/2由task2、5与10覆盖；3由task3、10覆盖；4由task3、5–8覆盖；5由task1、10覆盖。算法计算与显示共享同一状态，真实配置、教学数据和源码快照有独立来源标签。

建议在当前会话由主代理逐项实现（Native），因为页面、共享播放、内容目录和实验状态接口联系紧密，直接实现减少交接成本；最后做独立代码审查。若用户选择Subagent-driven，则按技能逐项委派并分别审查。实现前须由用户审阅本计划并选择执行方式。
