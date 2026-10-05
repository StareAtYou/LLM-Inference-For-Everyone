# 推理工作台 Implementation Plan

> **For agentic workers:** Use test-driven development for each owned component; the approved direct-implementation scope continues. Independent computation and mechanism views use dispatching-parallel-agents; integration and verification remain with the main implementer.

**Goal:** 单/批请求的完整推理数据流与所有已交付主题的独立原理动图。
**Architecture:** 纯 TypeScript trace engine → immutable event frames → SVG/HTML flow canvas and tensor inspector; separate mechanism registry/player embeds into topic and workbench pages.
**Tech Stack:** Existing React / TS / Vite / CSS / SVG / Vitest / Playwright, no new dependencies.
**Spec:** docs/superpowers/specs/2026-10-04-inference-workbench-design.md

## Global Constraints
- Reuse active worktree codex/inference-atlas-v1, no weights/API/GPU.
- Existing two real model presets readonly; tiny computed values label as teaching network.
- All 23 topic IDs plus internal normalization/projection/residual/output operations have dedicated animations.
- No quizzes; source/assumption/units stay visible; reduced motion, keyboard and mobile supported.

## Review Focus
1. Switching model/mode/params while playing resets frame and stops old clocks.
2. Batched requests never share layer state; sampling output joins cache only on next input pass.
3. Seek/replay does not mutate snapshots and completion releases allocated blocks.
4. No empty/NaN tensors or fake measured GPU timing; actual and miniature shape labels distinct.
5. Multiple players and mobile overlays have local clocks, no focus traps or overflow.

## Task 1 — Computation and immutable trace (independent)
Files: src/simulation/inferenceTrace.ts, tests/unit/inferenceTrace.test.ts.
Produces buildInferenceTrace(config), exported types TraceConfig/TraceFrame/TraceRequestState/TraceTensor. Exact interface agreed in agent brief. RED missing builder then GREEN mathematical checks, request lifecycle/cache independence/finite shapes/determinism; commit owned files only.

## Task 2 — Independent module/principle animations (independent)
Files: src/mechanisms/catalog.ts, MechanismPlayer.tsx, mechanisms.css; tests/unit/mechanisms.test.ts.
Produces mechanisms (id/title/subtitle/category), MechanismPlayer({id,compact?,onPlay?}); all topic IDs and six core operations. RED coverage/math then GREEN. Local clock/ref, no changes to shared hooks or pages; commit owned files only.

## Task 3 — Full journey UI and topic integration
Files: src/pages/Pipeline.tsx; src/visualizations/InferenceFlow.tsx, TensorInspector.tsx; src/styles/workbench.css; src/pages/Topic.tsx; tests/e2e/workbench.spec.ts, update pipeline test for expanded frame count.
RED no mode/node interaction then implement live route with request lanes, stage navigation, operator/layer/overview detail, tensors, caches, sampled output, release, mechanism panel. Independent players pause main playback on open. GREEN targeted E2E; keep aliases meaningful rather than old 10-frame fixture count.

## Task 4 — Verification and delivery
Run typecheck, all unit, build, full production E2E; screenshot single/batch/operators/mechanisms/topic at 390/768/1440 and boundaries360/1920, actual images viewed. Fix concrete problems with reproductions. Update README/quality notes; independent review of completed upgrade and fix important findings; local preview stays4173, keep branch/worktree.
