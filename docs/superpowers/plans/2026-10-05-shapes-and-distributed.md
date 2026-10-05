# Shapes and Distributed Inference Implementation Plan

> **For agentic workers:** Use focused parallel agents for independent shape, parallel and collective domains; coordinator integrates and runs whole-branch review. Test meaningful behavior before implementation.

**Goal:** Make tensor dimensions, six GPU partition strategies and nine communication operations understandable through derived shapes and continuous data animations.
**Architecture:** Extend existing trace/mechanism inspectors, add isolated parallel and collective explorers, integrate a lazy distributed page and learning topics. Shared useTimeline preserves current playback behavior.
**Tech Stack:** Existing React/TypeScript/Vite, SVG, Vitest and Playwright; no new dependencies.
**Spec:** docs/superpowers/specs/2026-10-05-shapes-and-distributed-design.md

## Global Constraints

Chinese UI, three depth levels, no exercises. Browser teaching examples only, source attribution and explicit limits. Preserve all current routing and timelines. Anonymous public Git identity; privacy check before publication.

## Review Focus

- Ragged batch and Decode query length must not be confused with historical KV length.
- GQA/KV and gated model projections must use correct configured widths.
- Non-divisible splits and invalid rank inputs must yield a clear explanation or supported selection.
- DP inference, Megatron SP, CP and EP group overlap must not teach training-only assumptions as universal inference facts.
- Paused continuous objects must freeze; mobile and text zoom must remain usable.

## Task 1: Shape explanations

Files: new src/education/*, tests/unit/shapeDerivations.test.ts; integrate src/visualizations/TensorInspector.tsx and src/mechanisms/MechanismPlayer.tsx.
Interfaces: shape presenter takes current TraceFrame/request/model; mechanism presenter takes mechanism id/current frame index and frames/params as needed.
- [x] Add failing tests for attention Prefill/Decode, GQA, FFN/MoE and all existing mechanism shape coverage.
- [x] Implement numeric/symbolic derivations with axes and current-stage highlighting.
- [x] Embed beside existing animated computation; run tests and typecheck.

## Task 2: Parallel strategies

Files: src/distributed/ParallelExplorer.tsx, parallel.ts, parallel.css; tests/unit/parallel.test.ts.
Interfaces: default ParallelExplorer accepts optional initialStrategy ('tp'|'dp'|'ep'|'pp'|'cp'|'sp'); isolated from collective implementation.
- [x] Test tensor partition/reconstruction, DP replica semantics, PP activation handoff, EP dispatch/merge, CP global attention and SP layout transitions.
- [x] Build six independently selectable animations using useTimeline and persistent data objects.
- [x] Expose input/output/local shape derivations, rank selection, parameters, official sources and boundaries.

## Task 3: Communication primitives

Files: src/distributed/CollectiveExplorer.tsx, collectives.ts, collectives.css; tests/unit/collectives.test.ts.
Interfaces: default CollectiveExplorer accepts optional initialOperation ('all-reduce'|'all-gather'|'reduce-scatter'|'all-to-all'|'broadcast'|'reduce'|'gather'|'scatter'|'send-recv'); isolated from parallel implementation.
- [x] Test mathematical operation results, ordering, root semantics, conservation and ReduceScatter+AllGather equivalence to AllReduce.
- [x] Build persistent moving buffer objects for nine operations; staged/continuous controls reuse useTimeline.
- [x] Show actual input/output buffers, shape changes, rank ownership and communication purpose.

## Task 4: Discoverability and integrated verification

Files: src/pages/Distributed.tsx, App.tsx, SiteShell.tsx, content/distributedTopics.ts, content/topics.ts, pages/Learn.tsx, pages/Topic.tsx; tests/e2e/distributed.spec.ts.
- [x] Add failing browser tests for navigation, topic embeds, each selector, shape updates, playback and mobile overflow.
- [x] Wire /distributed tabs, lazy topic embeds, navigation and related links; reference official sources.
- [x] Run full unit/E2E, Pages browser tests, formatting/build/privacy checks.
- [x] Independent review; fix substantive findings.
- [ ] Publish sanitized commits and verify online links.
