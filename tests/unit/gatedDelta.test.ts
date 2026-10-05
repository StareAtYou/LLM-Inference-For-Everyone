import { test, expect } from "vitest";
import { buildDeltaTrace } from "../../src/simulation/gatedDelta";
test("delta_trace_updates_fixed_matrix_and_bounded_conv_state", () => {
  const frames = buildDeltaTrace(0.8, 0.6);
  expect(frames).toHaveLength(5);
  expect(frames[0].state).toEqual([
    [0, 0],
    [0, 0],
  ]);
  expect(frames[1].state).toEqual([
    [0.6, 0],
    [0.3, 0],
  ]);
  expect(frames[1].buffer).toEqual([0, 0, 1]);
  expect(frames[4].buffer).toEqual([-0.5, 0.8, 0.2]);
  for (const frame of frames) {
    expect(frame.state).toHaveLength(2);
    expect(frame.state.flat().every(Number.isFinite)).toBe(true);
    expect(frame.buffer).toHaveLength(3);
  }
});
test("zero_write_gate_retains_no_memory_and_decay_changes_later_state", () => {
  expect(buildDeltaTrace(0.8, 0).at(-1)!.state).toEqual([
    [0, 0],
    [0, 0],
  ]);
  expect(buildDeltaTrace(1, 0.6)[2].state).not.toEqual(
    buildDeltaTrace(0.2, 0.6)[2].state,
  );
  expect(() => buildDeltaTrace(NaN, 0.6)).toThrow();
});
