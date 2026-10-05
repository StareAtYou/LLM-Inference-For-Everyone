export type TimelineMode = "staged" | "continuous";

export interface TimelineOptions {
  frameCount: number;
  intervalMs: number;
  durationsMs?: readonly number[];
  keyframes?: readonly number[];
}

export interface Timeline {
  frameCount: number;
  intervalMs: number;
  boundariesMs: readonly number[];
  durationMs: number;
  keyframes: readonly number[];
}

export interface TimelineSnapshot {
  index: number;
  fraction: number;
  position: number;
  playing: boolean;
  mode: TimelineMode;
  speed: number;
  elapsedMs: number;
  durationMs: number;
}

function positiveDuration(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) && value > 0
    ? value
    : fallback;
}

/** A master trajectory; keyframes are samples, never a second trajectory. */
export function createTimeline(options: TimelineOptions): Timeline {
  const frameCount = Number.isFinite(options.frameCount)
    ? Math.max(0, Math.trunc(options.frameCount))
    : 0;
  const intervalMs = positiveDuration(options.intervalMs, 1000);
  const boundariesMs = [0];
  for (let index = 0; index < frameCount - 1; index++) {
    boundariesMs.push(
      boundariesMs[index] +
        positiveDuration(options.durationsMs?.[index], intervalMs),
    );
  }
  const last = Math.max(0, frameCount - 1);
  const keyframes = options.keyframes
    ? [...new Set([0, last, ...options.keyframes.filter(Number.isFinite)])]
        .filter(
          (position) =>
            Number.isInteger(position) && position >= 0 && position <= last,
        )
        .sort((a, b) => a - b)
    : Array.from({ length: Math.max(1, frameCount) }, (_, index) => index);
  return {
    frameCount,
    intervalMs,
    boundariesMs,
    durationMs: boundariesMs.at(-1) ?? 0,
    keyframes,
  };
}

export function clampPosition(timeline: Timeline, position: number): number {
  const last = Math.max(0, timeline.frameCount - 1);
  return Number.isNaN(position) ? 0 : Math.max(0, Math.min(position, last));
}

export function elapsedAtPosition(timeline: Timeline, value: number): number {
  const position = clampPosition(timeline, value);
  const index = Math.floor(position);
  const start = timeline.boundariesMs[index] ?? 0;
  const end = timeline.boundariesMs[index + 1] ?? start;
  return start + (end - start) * (position - index);
}

export function positionAtElapsed(
  timeline: Timeline,
  elapsedMs: number,
): number {
  if (!(elapsedMs > 0) || timeline.durationMs === 0) return 0;
  if (elapsedMs >= timeline.durationMs)
    return Math.max(0, timeline.frameCount - 1);
  let low = 0;
  let high = timeline.boundariesMs.length - 1;
  while (low + 1 < high) {
    const middle = Math.floor((low + high) / 2);
    if (timeline.boundariesMs[middle] <= elapsedMs) low = middle;
    else high = middle;
  }
  const start = timeline.boundariesMs[low];
  const end = timeline.boundariesMs[low + 1];
  return low + (elapsedMs - start) / (end - start);
}

/** Timestamp-driven clock independent of rendering and browser schedulers. */
export class TimelineClock {
  private timeline: Timeline;
  private position = 0;
  private playing = false;
  private mode: TimelineMode = "staged";
  private speed = 1;
  private timestamp = 0;
  private stagedElapsedMs = 0;

  constructor(options: TimelineOptions) {
    this.timeline = createTimeline(options);
  }

  snapshot(): TimelineSnapshot {
    const index = Math.floor(this.position);
    return {
      index,
      fraction: this.position - index,
      position: this.position,
      playing: this.playing,
      mode: this.mode,
      speed: this.speed,
      elapsedMs: elapsedAtPosition(this.timeline, this.position),
      durationMs: this.timeline.durationMs,
    };
  }

  private nextKeyframe(): number {
    return (
      this.timeline.keyframes.find((position) => position > this.position) ??
      Math.max(0, this.timeline.frameCount - 1)
    );
  }

  advance(now: number): TimelineSnapshot {
    const delta = Number.isFinite(now) ? Math.max(0, now - this.timestamp) : 0;
    if (Number.isFinite(now)) this.timestamp = Math.max(this.timestamp, now);
    if (!this.playing) return this.snapshot();
    const scaledDelta = delta * this.speed;
    if (this.mode === "continuous") {
      this.position = positionAtElapsed(
        this.timeline,
        elapsedAtPosition(this.timeline, this.position) + scaledDelta,
      );
    } else {
      this.stagedElapsedMs += scaledDelta;
      while (this.stagedElapsedMs >= this.timeline.intervalMs && this.playing) {
        this.stagedElapsedMs -= this.timeline.intervalMs;
        this.position = this.nextKeyframe();
        this.stopAtEnd();
      }
    }
    this.stopAtEnd();
    return this.snapshot();
  }

  private stopAtEnd(): void {
    if (this.position >= Math.max(0, this.timeline.frameCount - 1)) {
      this.playing = false;
      this.stagedElapsedMs = 0;
    }
  }

  /** Remaining wall time until the next staged sample, including partial time. */
  nextStageDelayMs(): number {
    return (
      Math.max(0, this.timeline.intervalMs - this.stagedElapsedMs) / this.speed
    );
  }

  play(now: number): TimelineSnapshot {
    this.advance(now);
    if (this.position >= Math.max(0, this.timeline.frameCount - 1)) {
      this.position = 0;
      this.stagedElapsedMs = 0;
    }
    this.playing = this.timeline.frameCount > 1;
    return this.snapshot();
  }

  pause(now: number): TimelineSnapshot {
    this.advance(now);
    this.playing = false;
    return this.snapshot();
  }

  setMode(mode: TimelineMode, now: number): TimelineSnapshot {
    if (mode !== this.mode) {
      this.pause(now);
      this.mode = mode;
      this.stagedElapsedMs = 0;
    }
    return this.snapshot();
  }

  setSpeed(speed: number, now: number): TimelineSnapshot {
    this.advance(now);
    if (Number.isFinite(speed) && speed > 0) {
      this.speed = Math.max(0.25, Math.min(16, speed));
    }
    return this.snapshot();
  }

  seek(position: number, now: number): TimelineSnapshot {
    this.pause(now);
    this.position = clampPosition(this.timeline, position);
    this.stagedElapsedMs = 0;
    return this.snapshot();
  }

  step(now: number): TimelineSnapshot {
    this.pause(now);
    this.position =
      this.mode === "staged"
        ? this.nextKeyframe()
        : clampPosition(this.timeline, Math.floor(this.position) + 1);
    this.stagedElapsedMs = 0;
    return this.snapshot();
  }

  reset(now: number): TimelineSnapshot {
    this.pause(now);
    this.position = 0;
    this.stagedElapsedMs = 0;
    return this.snapshot();
  }

  configure(options: TimelineOptions, now: number): TimelineSnapshot {
    this.advance(now);
    const stagedProgress = this.stagedElapsedMs / this.timeline.intervalMs;
    this.timeline = createTimeline(options);
    this.stagedElapsedMs = stagedProgress * this.timeline.intervalMs;
    this.position = clampPosition(this.timeline, this.position);
    this.stopAtEnd();
    return this.snapshot();
  }
}
