import { useCallback, useEffect, useState } from "react";
export function usePlayback({
  frameCount,
  intervalMs,
  resetKey,
}: {
  frameCount: number;
  intervalMs: number;
  resetKey: string;
}) {
  const [state, setState] = useState({
    index: 0,
    playing: false,
    key: resetKey,
  });
  const current =
    state.key === resetKey
      ? state
      : { index: 0, playing: false, key: resetKey };
  const index = Math.min(current.index, Math.max(0, frameCount - 1));
  const playing = current.playing;
  useEffect(() => {
    setState({ index: 0, playing: false, key: resetKey });
  }, [resetKey]);
  const pause = useCallback(
    () => setState((s) => ({ ...s, playing: false })),
    [],
  );
  useEffect(() => {
    const visibility = () => {
      if (document.hidden) pause();
    };
    document.addEventListener("visibilitychange", visibility);
    const surface = document.querySelector("[data-playback-surface]");
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries[0].isIntersecting) pause();
      },
      { threshold: 0.05 },
    );
    if (surface) observer.observe(surface);
    return () => {
      document.removeEventListener("visibilitychange", visibility);
      observer.disconnect();
    };
  }, [pause, resetKey]);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () =>
        setState((s) => {
          const next = Math.min(s.index + 1, frameCount - 1);
          return { ...s, index: next, playing: next < frameCount - 1 };
        }),
      intervalMs,
    );
    return () => window.clearInterval(timer);
  }, [playing, frameCount, intervalMs, resetKey]);
  return {
    index,
    playing,
    pause,
    play: () => {
      if (!document.hidden)
        setState({
          key: resetKey,
          index: index >= frameCount - 1 ? 0 : index,
          playing: frameCount > 1,
        });
    },
    step: () =>
      setState({
        key: resetKey,
        index: Math.min(index + 1, frameCount - 1),
        playing: false,
      }),
    reset: () => setState({ key: resetKey, index: 0, playing: false }),
    seek: (value: number) =>
      setState({
        key: resetKey,
        index: Math.max(0, Math.min(Math.trunc(value), frameCount - 1)),
        playing: false,
      }),
  };
}
