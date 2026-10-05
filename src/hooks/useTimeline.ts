import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import {
  TimelineClock,
  type TimelineMode,
  type TimelineOptions,
  type TimelineSnapshot,
} from "../simulation/timeline";

export interface UseTimelineOptions extends TimelineOptions {
  resetKey: string;
  surfaceKey?: string;
  surfaceRef: RefObject<HTMLElement | null>;
}

/** Both playback modes own one clock and one exact fractional playhead. */
export function useTimeline({
  frameCount,
  intervalMs,
  resetKey,
  surfaceRef,
  surfaceKey,
  durationsMs,
  keyframes,
}: UseTimelineOptions) {
  const clockRef = useRef<TimelineClock | null>(null);
  if (!clockRef.current) {
    clockRef.current = new TimelineClock({
      frameCount,
      intervalMs,
      durationsMs,
      keyframes,
    });
  }
  const clock = clockRef.current;
  const identity = `${resetKey}\0${frameCount}`;
  const identityRef = useRef(identity);
  identityRef.current = identity;
  const configuredIdentity = useRef(identity);
  const [state, setState] = useState(() => ({
    identity,
    snapshot: clock.snapshot(),
  }));
  const cancelScheduled = useRef<(() => void) | null>(null);
  const offscreen = useRef(false);
  // Value signatures keep callers' inline arrays from restarting the scheduler.
  const durationsKey = JSON.stringify(durationsMs ?? []);
  const keyframesKey = JSON.stringify(keyframes ?? null);

  const publish = useCallback((snapshot: TimelineSnapshot) => {
    setState({ identity: identityRef.current, snapshot });
  }, []);

  const cancel = useCallback(() => {
    cancelScheduled.current?.();
    cancelScheduled.current = null;
  }, []);

  const pause = useCallback(() => {
    cancel();
    publish(clock.pause(performance.now()));
  }, [cancel, clock, publish]);

  useEffect(() => {
    cancel();
    const now = performance.now();
    const changedModel = configuredIdentity.current !== identity;
    if (changedModel) clock.reset(now);
    clock.configure({ frameCount, intervalMs, durationsMs, keyframes }, now);
    configuredIdentity.current = identity;
    publish(clock.snapshot());
  }, [
    clock,
    cancel,
    publish,
    identity,
    frameCount,
    intervalMs,
    durationsKey,
    keyframesKey,
  ]);

  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) pause();
    };
    document.addEventListener("visibilitychange", onVisibility);
    offscreen.current = false;
    const surface = surfaceRef.current;
    const observer =
      typeof IntersectionObserver === "undefined"
        ? null
        : new IntersectionObserver(
            (entries) => {
              const entry = entries.find((item) => item.target === surface);
              if (!entry) return;
              offscreen.current =
                !entry.isIntersecting || entry.intersectionRatio < 0.05;
              if (offscreen.current) pause();
            },
            { threshold: 0.05 },
          );
    if (surface) observer?.observe(surface);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      observer?.disconnect();
    };
  }, [pause, resetKey, surfaceRef, surfaceKey]);

  useEffect(() => {
    if (!state.snapshot.playing) return;
    let cancelled = false;
    let raf: number | undefined;
    let timer: number | undefined;
    const tick = () => {
      if (cancelled) return;
      const snapshot = clock.advance(performance.now());
      publish(snapshot);
      if (!snapshot.playing) return;
      schedule();
    };
    const schedule = () => {
      if (clock.snapshot().mode === "continuous") {
        raf = window.requestAnimationFrame(tick);
      } else {
        timer = window.setTimeout(tick, clock.nextStageDelayMs());
      }
    };
    const stop = () => {
      cancelled = true;
      if (raf !== undefined) window.cancelAnimationFrame(raf);
      if (timer !== undefined) window.clearTimeout(timer);
    };
    cancelScheduled.current = stop;
    schedule();
    return () => {
      stop();
      if (cancelScheduled.current === stop) cancelScheduled.current = null;
    };
  }, [
    clock,
    publish,
    state.snapshot.playing,
    state.snapshot.mode,
    state.snapshot.speed,
    identity,
    intervalMs,
    durationsKey,
    keyframesKey,
  ]);

  useEffect(() => cancel, [cancel]);

  const play = useCallback(() => {
    if (!document.hidden && !offscreen.current)
      publish(clock.play(performance.now()));
  }, [clock, publish]);
  const setMode = useCallback(
    (mode: TimelineMode) => {
      if (mode !== clock.snapshot().mode) cancel();
      publish(clock.setMode(mode, performance.now()));
    },
    [cancel, clock, publish],
  );
  const setSpeed = useCallback(
    (speed: number) => {
      publish(clock.setSpeed(speed, performance.now()));
    },
    [clock, publish],
  );
  const seek = useCallback(
    (position: number) => {
      cancel();
      publish(clock.seek(position, performance.now()));
    },
    [cancel, clock, publish],
  );
  const step = useCallback(() => {
    cancel();
    publish(clock.step(performance.now()));
  }, [cancel, clock, publish]);
  const reset = useCallback(() => {
    cancel();
    publish(clock.reset(performance.now()));
  }, [cancel, clock, publish]);

  // A changed model must never render one frame of the previous trajectory.
  const snapshot =
    state.identity === identity
      ? state.snapshot
      : {
          ...state.snapshot,
          index: 0,
          fraction: 0,
          position: 0,
          elapsedMs: 0,
          playing: false,
        };
  return { ...snapshot, setMode, setSpeed, pause, play, step, reset, seek };
}
