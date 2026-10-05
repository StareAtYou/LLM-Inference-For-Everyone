import { describe, expect, test } from "vitest";
import {
  createTimeline,
  elapsedAtPosition,
  positionAtElapsed,
  TimelineClock,
} from "../../src/simulation/timeline";

describe("timeline trajectory", () => {
  test("maps unequal transition durations and exact boundaries", () => {
    const timeline = createTimeline({
      frameCount: 4,
      intervalMs: 100,
      durationsMs: [20, 40, 80, 999],
    });
    expect(timeline.durationMs).toBe(140);
    expect(positionAtElapsed(timeline, 10)).toBe(0.5);
    expect(positionAtElapsed(timeline, 20)).toBe(1);
    expect(positionAtElapsed(timeline, 40)).toBe(1.5);
    expect(positionAtElapsed(timeline, 100)).toBe(2.5);
    expect(positionAtElapsed(timeline, 140)).toBe(3);
    expect(positionAtElapsed(timeline, 1000)).toBe(3);
    expect(elapsedAtPosition(timeline, 2.5)).toBe(100);
    expect(elapsedAtPosition(timeline, -1)).toBe(0);
  });

  test("invalid durations and degenerate trajectories remain finite", () => {
    const timeline = createTimeline({
      frameCount: 4,
      intervalMs: 10,
      durationsMs: [0, Number.NaN, -5],
      keyframes: [3, 1, 1, -4, Number.NaN, 99],
    });
    expect(timeline.durationMs).toBe(30);
    expect(timeline.keyframes).toEqual([0, 1, 3]);
    for (const frameCount of [0, 1, Number.NaN]) {
      const empty = createTimeline({ frameCount, intervalMs: Number.NaN });
      expect(empty.durationMs).toBe(0);
      expect(positionAtElapsed(empty, Number.NaN)).toBe(0);
      expect(elapsedAtPosition(empty, Number.NaN)).toBe(0);
    }
  });
});

describe("shared playback clock", () => {
  test("continuous playback uses every small delta and stops at the final frame", () => {
    const clock = new TimelineClock({ frameCount: 3, intervalMs: 10 });
    clock.setMode("continuous", 0);
    clock.play(0);
    expect(clock.advance(2).position).toBeCloseTo(0.2);
    expect(clock.advance(5).position).toBeCloseTo(0.5);
    expect(clock.advance(10)).toMatchObject({ index: 1, fraction: 0 });
    expect(clock.advance(25)).toMatchObject({
      index: 2,
      fraction: 0,
      position: 2,
      playing: false,
      elapsedMs: 20,
    });
  });

  test("keyframes sample the same master trajectory without changing modes' position", () => {
    const clock = new TimelineClock({
      frameCount: 6,
      intervalMs: 10,
      keyframes: [0, 2, 5],
    });
    clock.setMode("continuous", 0);
    clock.play(0);
    clock.advance(7);
    expect(clock.setMode("staged", 7)).toMatchObject({
      position: 0.7,
      playing: false,
      mode: "staged",
    });
    expect(clock.setMode("continuous", 7).position).toBe(0.7);
    clock.setMode("staged", 7);
    expect(clock.step(7).position).toBe(2);
    clock.play(7);
    expect(clock.advance(16).position).toBe(2);
    expect(clock.advance(17)).toMatchObject({ position: 5, playing: false });
  });

  test("pause, resume, and speed changes rebase time without a jump", () => {
    const clock = new TimelineClock({ frameCount: 5, intervalMs: 100 });
    clock.setMode("continuous", 0);
    clock.play(0);
    expect(clock.advance(25).position).toBe(0.25);
    expect(clock.setSpeed(2, 25).position).toBe(0.25);
    expect(clock.advance(50).position).toBe(0.75);
    expect(clock.pause(50).position).toBe(0.75);
    expect(clock.advance(500).position).toBe(0.75);
    clock.play(500);
    expect(clock.advance(525).position).toBe(1.25);
  });

  test("staged speed changes preserve the current frame and pending interval", () => {
    const clock = new TimelineClock({ frameCount: 4, intervalMs: 20 });
    clock.play(0);
    expect(clock.advance(5).position).toBe(0);
    clock.setSpeed(3, 5);
    expect(clock.advance(9).position).toBe(0);
    expect(clock.advance(10).position).toBe(1);
  });

  test("changing staged duration rebases its pending interval without an immediate step", () => {
    const clock = new TimelineClock({ frameCount: 4, intervalMs: 100 });
    clock.play(0);
    clock.advance(80);
    clock.configure({ frameCount: 4, intervalMs: 50 }, 80);
    expect(clock.advance(80).position).toBe(0);
    expect(clock.advance(89).position).toBe(0);
    expect(clock.advance(90).position).toBe(1);
  });

  test("duration changes preserve fractional position while running", () => {
    const clock = new TimelineClock({ frameCount: 3, intervalMs: 10 });
    clock.setMode("continuous", 0);
    clock.play(0);
    clock.advance(5);
    expect(
      clock.configure(
        { frameCount: 3, intervalMs: 20, durationsMs: [20, 40] },
        5,
      ),
    ).toMatchObject({ position: 0.5, elapsedMs: 10, durationMs: 60 });
    expect(clock.advance(15).position).toBe(1);
    expect(clock.advance(25).position).toBe(1.25);
  });

  test("reverse seek pauses and subsequent playback begins at the sought position", () => {
    const clock = new TimelineClock({ frameCount: 4, intervalMs: 10 });
    clock.setMode("continuous", 0);
    clock.play(0);
    clock.advance(25);
    expect(clock.seek(0.25, 25)).toMatchObject({
      position: 0.25,
      playing: false,
    });
    clock.play(30);
    expect(clock.advance(35).position).toBe(0.75);
    expect(clock.seek(99, 35)).toMatchObject({ index: 3, fraction: 0 });
    expect(clock.play(40)).toMatchObject({ position: 0, playing: true });
    expect(clock.seek(-4, 40).position).toBe(0);
  });

  test("invalid speed cannot poison time and reset preserves playback preferences", () => {
    const clock = new TimelineClock({ frameCount: 3, intervalMs: 10 });
    expect(clock.setSpeed(100, 0).speed).toBe(16);
    expect(clock.setSpeed(0.01, 0).speed).toBe(0.25);
    expect(clock.setSpeed(Number.NaN, 0).speed).toBe(0.25);
    expect(clock.setSpeed(-2, 0).speed).toBe(0.25);
    clock.setMode("continuous", 0);
    clock.seek(1.5, 0);
    expect(clock.reset(0)).toMatchObject({
      position: 0,
      playing: false,
      speed: 0.25,
      mode: "continuous",
    });
  });
});
