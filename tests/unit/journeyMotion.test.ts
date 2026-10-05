import { describe, expect, test } from "vitest";
import { buildInferenceTrace } from "../../src/simulation/inferenceTrace";
import {
  buildJourneyTracks,
  journeyPose,
} from "../../src/visualizations/journeyMotion";

const makeTrace = (mode: "single" | "batch") =>
  buildInferenceTrace({
    modelId: "qwen38-dense",
    scenarioId: "sky",
    mode,
    capacity: 2,
    temperature: 0,
    topK: 6,
    topP: 1,
    outputLimit: 3,
  });
const distance = (a: { x: number; y: number }, b: { x: number; y: number }) =>
  Math.hypot(a.x - b.x, a.y - b.y);

describe("unbroken inference transport", () => {
  for (const mode of ["single", "batch"] as const) {
    test(`${mode}: every operator, layer, request arrival and feedback boundary has matching position and velocity`, () => {
      const trace = makeTrace(mode);
      const tracks = buildJourneyTracks(trace);
      for (const track of tracks)
        for (let i = 1; i < trace.length - 1; i++) {
          const epsilon = 0.00001;
          const left = journeyPose(track, i - epsilon),
            at = journeyPose(track, i),
            right = journeyPose(track, i + epsilon);
          expect(distance(left, right)).toBeLessThan(0.03);
          const lv = {
            x: (at.x - left.x) / epsilon,
            y: (at.y - left.y) / epsilon,
          };
          const rv = {
            x: (right.x - at.x) / epsilon,
            y: (right.y - at.y) / epsilon,
          };
          expect(distance(lv, rv)).toBeLessThan(0.2);
        }
    });
  }
  test("the same request travels through the network and returns its generated token to the next input", () => {
    const trace = makeTrace("single"),
      [track] = buildJourneyTracks(trace);
    const emit = trace.find((f) => f.stage === "emit")!.index;
    const feedback = trace.find((f) => f.stage === "feedback")!.index;
    expect(journeyPose(track, emit).y).toBeGreaterThan(350);
    expect(journeyPose(track, (emit + feedback) / 2).y).toBeGreaterThan(350);
    expect(journeyPose(track, feedback).x).toBeLessThan(250);
    expect(
      distance(journeyPose(track, emit), journeyPose(track, feedback)),
    ).toBeGreaterThan(400);
    const embedding = trace.find(
      (f) => f.tick > 0 && f.stage === "embedding",
    )!.index;
    expect(journeyPose(track, embedding).x).toBeLessThan(300);
  });
  test("paths are finite and stay inside the canvas for both single and continuous batch traces", () => {
    for (const mode of ["single", "batch"] as const) {
      const trace = makeTrace(mode);
      for (const track of buildJourneyTracks(trace))
        for (let p = 0; p < trace.length - 1; p += 0.25) {
          const pose = journeyPose(track, p);
          expect(Number.isFinite(pose.x + pose.y)).toBe(true);
          expect(pose.x).toBeGreaterThan(0);
          expect(pose.x).toBeLessThan(940);
          expect(pose.y).toBeGreaterThan(0);
          expect(pose.y).toBeLessThan(520);
        }
    }
  });
});

test("parked batch requests retain their data representation while another request arrives or emits", () => {
  const trace = makeTrace("batch"),
    tracks = buildJourneyTracks(trace);
  for (const track of tracks)
    for (let i = 1; i < trace.length; i++) {
      if (!trace[i].requestIds.includes(track.id))
        expect(track.channelValues[i]).toEqual(track.channelValues[i - 1]);
    }
});

test("generated tokens stay with their own parked request until its next embedding", () => {
  const trace = makeTrace("batch");
  for (const track of buildJourneyTracks(trace)) {
    let token: string | null = null;
    trace.forEach((frame, i) => {
      if (frame.requestIds.includes(track.id) && frame.stage === "emit")
        token = frame.requests
          .find((r) => r.id === track.id)!
          .outputTokens.at(-1)!;
      if (frame.requestIds.includes(track.id) && frame.stage === "embedding")
        token = null;
      expect(track.carriedTokens[i]).toBe(token);
    });
  }
});
