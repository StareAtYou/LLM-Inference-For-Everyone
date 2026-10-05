import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";
import MechanismPlayer from "../../src/mechanisms/MechanismPlayer";
import ContinuousMechanismScene, {
  continuousProgress,
} from "../../src/mechanisms/ContinuousMechanismScene";
import { buildMechanismFrames, mechanisms } from "../../src/mechanisms/catalog";

// A missing mode switch makes continuous playback unreachable while staged controls still work.
describe("principle playback mode contract", () => {
  test("starts staged and exposes both accessible playback modes", () => {
    const html = renderToStaticMarkup(
      createElement(MechanismPlayer, { id: "attention" }),
    );
    expect(html).toContain('data-mode="staged"');
    expect(html).toContain('aria-label="分段讲解原理" aria-pressed="true"');
    expect(html).toContain('aria-label="连续演示原理" aria-pressed="false"');
    expect(html).toContain('aria-label="原理动画进度"');
    expect(html).toContain('data-testid="mechanism-board"');
    expect(html).not.toContain('data-testid="continuous-mechanism-scene"');
  });
});

const scene = (id: string, position: number, params?: Record<string, number>) =>
  renderToStaticMarkup(
    createElement(ContinuousMechanismScene, {
      definition: mechanisms.find((m) => m.id === id)!,
      frames: buildMechanismFrames(id, params),
      position,
    }),
  );

describe("persistent continuous principle scenes", () => {
  test("whole mechanism percent does not reset at each exact keyframe", () => {
    expect(continuousProgress(0.75, 4)).toBe(0.25);
    expect(continuousProgress(1.5, 4)).toBe(0.5);
    expect(continuousProgress(3, 4)).toBe(1);
    expect(continuousProgress(-1, 4)).toBe(0);
  });
  test("shows every mechanism's full connected path with its real panel glyphs at start, between stages and end", () => {
    const signatures = new Set<string>();
    for (const definition of mechanisms) {
      const frames = buildMechanismFrames(definition.id);
      for (const position of [
        0,
        (frames.length - 1) / 2 + 0.125,
        frames.length - 1,
      ]) {
        const html = scene(definition.id, position);
        expect(html).toContain(`data-view="${definition.view}"`);
        expect(html.match(/data-stage-index=/g)).toHaveLength(frames.length);
        expect(html.match(/data-flow-edge=/g)).toHaveLength(frames.length - 1);
        for (const frame of frames) {
          expect(html).toContain(frame.stage);
          for (const panel of frame.panels)
            expect(html).toContain(`data-glyph-kind="${panel.kind}"`);
        }
        expect(html).not.toMatch(/(?:NaN|Infinity|undefined)/);
      }
      signatures.add(scene(definition.id, 0));
    }
    expect(signatures.size).toBe(29);
  });

  test("continuous motion advances on the persistent path while numeric results stay at exact keyframes", () => {
    const midway = scene("qkv", 1.5);
    const later = scene("qkv", 1.875);
    expect(midway).toContain('data-progress="0.5"');
    expect(later).toContain('data-progress="0.625"');
    expect(midway.match(/data-geometry="([^"]+)"/)?.[1]).not.toEqual(
      later.match(/data-geometry="([^"]+)"/)?.[1],
    );
    expect(scene("qkv", 4)).toContain('data-output="[2,0,1,-1,1,1]"');
    expect(midway.match(/data-values="([^"]*)"/g)).toEqual(
      later.match(/data-values="([^"]*)"/g),
    );
    expect(midway).toContain("数值保持关键帧的精确计算结果");
  });

  test("matrix shapes, mask and zero-valued operands retain their mathematical meaning", () => {
    const attention = scene("attention", 1.4, { row: 1 });
    expect(attention).toContain('data-rows="3" data-columns="2"');
    expect(attention).toContain('data-masked="true"');
    expect(scene("rmsnorm", 99, { amplitude: 0 })).toContain(
      'data-output="[0,0,0,0]"',
    );
  });

  test("rotation, expert routing and quantization have distinct data-bearing geometry", () => {
    expect(scene("rope", 1.5)).toContain('data-glyph-kind="rotation"');
    const experts = scene("moe", 2.5);
    expect(experts).toContain('data-glyph-kind="experts"');
    expect(experts).toContain('data-selected="true"');
    expect(scene("quantization", 3)).toContain('data-glyph-kind="plot"');
  });

  test("out-of-range progress is clamped without invalid geometry", () => {
    expect(scene("rmsnorm", -2)).toContain('data-progress="0"');
    expect(scene("rmsnorm", 999)).toContain('data-progress="1"');
    expect(scene("rmsnorm", NaN)).toContain('data-progress="0"');
  });
});
