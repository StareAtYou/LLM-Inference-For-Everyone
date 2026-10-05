# 推理图谱 · Inference Atlas

面向入门、深入、精通三个阶段的中文大模型推理学习网站。第一版是可操作的浏览器教学工具：从 Token 与网络结构，到缓存、调度、量化和源码；没有小练习、测验或评分。

## 本地启动

建议 Node.js 22.12+；本次验证环境 Node 22.22.0 / npm 10.9.4。

```bash
npm install
npm run dev
```

终端显示开发入口，默认 http://127.0.0.1:5173。复现锁定依赖可使用 `npm ci`。

```bash
npm run build
npm run preview -- --port 4173
```

生产预览 http://127.0.0.1:4173。静态托管发布 `dist/`，需要将未知路由回退到 `index.html`，才能直接访问 `/learn/kv-cache` 等深链。GitHub Pages 构建使用 Hash 路由，不需要服务器配置回退。

## GitHub Pages 部署

仓库 Settings → Pages → Source 选择 **GitHub Actions**。公开仓库可使用免费 Pages；私有仓库需要支持 Pages 的套餐。推送到 `main` 或手动运行 `Deploy website to GitHub Pages` 工作流，会先检查脱敏、格式、单元测试与 Pages 路由，再发布 `dist/`。

部署后的项目地址为 https://stareatyou.github.io/LLM-Inference-For-Everyone/ ，具体页面使用 `#/pipeline?depth=beginner` 等路径，分享链接与刷新均可直接访问。是否已上线以 Actions 部署结果为准。

```bash
npm run test:pages
npm run build:pages
npm run preview -- --port 4173
```

Pages 构建的默认子路径为 `/LLM-Inference-For-Everyone/`，可通过 `PAGES_BASE_PATH` 覆盖；工作流使用当前项目子路径；仓库改名或使用自定义域名时，需要同步修改构建路径和 Pages 验证配置。本地默认开发仍使用常规路径。当前网站所有教学动画在浏览器运行；GitHub Pages 不运行 GPU 推理后端。

## 第一版范围

| 页面 | 内容与操作 |
| --- | --- |
| 首页 | 三条学习路径、请求流动图、模型、实验与源码入口 |
| 学习地图与专题 | 23 个主题及各自独立原理动图，三种解释深度、依赖提示、术语/源码模块检索、收藏和最近浏览 |
| 推理流程 | 单请求 / 连续批请求的完整计算轨迹；分段关键帧 / 连续时间轴双模式、可调速；旅程 / 逐层 / 逐算子视图，张量、缓存、输出与释放；29 个原理的双模式动图库 |
| 模型结构 | Qwen3.8-27B Dense、Qwen3.6-35B-A3B MoE 的完整层矩阵、混合注意力、FFN、专家路由、RoPE / GQA、递归/卷积状态与教学张量形状 |
| 五个实验 | 全注意力 KV 分项；分页/前缀缓存；连续批处理；INT4 / INT8 量化；贪心投机解码 |
| 框架源码 | vLLM 8 个模块、SGLang 7 个模块；固定版本关系图、职责/输入输出、短代码段、定位链接与原理关联 |

中文为主，保留英文术语。收藏、最近浏览和默认深度仅存本机 `localStorage`（键 `inference-atlas:preferences`）；禁用或损坏时仍可在当前会话使用，不向服务器上传。Ctrl/⌘K 打开检索，Esc 关闭手机菜单。支持减弱动效、键盘导航、手机/平板布局。

## 真实数据与教学数据

模型预设来自仓库保存的官方 `config.json`，来源与哈希见 [模型引用](docs/references/models/sources.json)。Dense / MoE 指 FFN 类型；两个预设均为混合注意力网络。真实结构参数只读，修改隐藏维度会进入标注的自定义教学模式。

- Qwen3.8-27B：revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。
- Qwen3.6-35B-A3B：revision `995ad96eacd98c81ed38be0c5b274b04031597b0`。
- vLLM v0.30.0：commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
- SGLang v0.5.21：commit `e00930c5489053f26d86b179cee0d087f846acbb`。

完整推理工作台使用固定教学 Token ID、确定性权重与 H=4 的小网络，逐层计算归一化、投影、注意力 / 线性状态、FFN / MoE、logits 与采样；数值、缓存和输出来自同一条不可变轨迹。MoE 主轨迹为 3 个路由专家选 2，再加门控共享专家。独立原理动图使用各自标注的小数组 / 工程事件示例；首页等概览仍使用说明性固定值。未运行真实 tokenizer 或 Qwen 权重。源码文件元数据、符号、行号与源文件 SHA-256 见 [框架引用](docs/references/frameworks/sources.json)。短代码段遵循 Apache-2.0，许可证随引用保存。图中的调用、消息、数据边是说明性的源码导览，源码概览中的中间层折叠已有标注；推理工作台另提供教学网络的完整逐层、逐算子轨迹。

缓存公式只估算 **Full Attention KV**，不包括线性层循环/卷积状态、模型权重、激活、元数据和其他显存开销。量化图为小数组对称权重量化；有效字节不含尺度与打包对齐。批处理时钟与投机解码成本为显式教学假设，不代表真实 GPU 性能。数字都有来源/单位/边界说明，当前没有实测指标。

## 验证与复查

```bash
npm run typecheck
npm run test:unit
npm run build
npm run test:e2e
npm run format:check
```

Playwright 在 macOS 默认使用已安装的 Chrome，其他平台默认 Chromium。没有浏览器时先运行 `npx playwright install chromium`，再设置 `PLAYWRIGHT_CHANNEL=chromium`。对已启动的生产预览运行测试：

```bash
PREVIEW_URL=http://127.0.0.1:4173 npm run test:e2e
PREVIEW_URL=http://127.0.0.1:4173 node scripts/capture-quality.mjs
```

连续动画衔接修正已通过 101 项单元测试（含发布脱敏检查）、39 项端到端测试、类型检查与生产构建。15 个页面/状态在 360、390、768、1440、1920px 共 75 次布局检查中无页面溢出或运行错误。原版复查见 [质量复查](docs/quality/v1-review.md)；工作台的 30 张截图见 [工作台复查](docs/quality/workbench-review.md)；双模式新增 15 张连续场景截图见 [连续动画复查](docs/quality/continuous-review.md)；持续数据对象、层间回路与 Token 反馈的修正见 [动画衔接复查](docs/quality/motion-review.md)。

## 目录与后续 B 阶段

[架构说明](docs/architecture.md) 描述内容、模拟、图形和状态的边界。`src/content/` 保存知识/模型/框架；`src/simulation/` 是纯计算；`src/visualizations/` 负责图形；`src/pages/` 与 `src/experiments/` 组织交互；`tests/` 验证行为；`docs/references/` 固定来源。

后续按依赖逐步扩展：

1. 接入服务端真实 Tokenizer、模型配置和受控推理接口；先明确硬件、模型 revision、运行参数及资源限额。
2. 增加 SGLang / vLLM 实际服务适配器，以请求 ID 关联排队、Prefill、Decode、KV 与输出事件。
3. 展示 TTFT、TPOT、吞吐、显存、缓存命中等实测观测值，同时保存环境与假设，区分机制模拟、理论估算和实测。
4. 在现有独立动图之上深化 FlashAttention、Chunked Prefill、Tensor / Pipeline / Expert Parallel、CUDA Graph 等机制，增加真实执行跟踪与部署场景。

以上是路线，尚未实现 GPU 推理、在线 benchmark、账号或服务端学习同步。现有并行与部署主题为概述；不会把长期“所有阶段/模块”的目标写成第一版已经全部交付。

## 发布前脱敏

运行 `npm run privacy:check` 检查受版本管理的工作区文件；发布前还可使用 `node scripts/privacy-check.mjs --ref HEAD --base origin/main` 检查待发布文件树与新增提交的元数据。扫描只报告文件/行号和类别，不输出命中的敏感值。环境变量、本机凭据、私钥与运行缓存由 `.gitignore` 排除。本次处理记录见 [发布脱敏记录](docs/quality/publication-privacy.md)。
