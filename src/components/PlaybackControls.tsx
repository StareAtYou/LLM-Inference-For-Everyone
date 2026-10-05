import { Play, Pause, SkipForward, RotateCcw } from "lucide-react";
import type { usePlayback } from "../hooks/usePlayback";
export default function PlaybackControls({
  player,
  count,
}: {
  player: ReturnType<typeof usePlayback>;
  count: number;
}) {
  return (
    <div className="playback-controls">
      <button
        className="button primary"
        onClick={player.playing ? player.pause : player.play}
      >
        {player.playing ? <Pause size={16} /> : <Play size={16} />}{" "}
        {player.playing ? "暂停" : "播放"}
      </button>
      <button
        className="control-button"
        onClick={player.step}
        disabled={player.index === count - 1}
      >
        <SkipForward size={17} />
        <span>单步</span>
      </button>
      <button className="control-button" onClick={player.reset}>
        <RotateCcw size={17} />
        <span>重置</span>
      </button>
      <input
        type="range"
        aria-label="跳转演示步骤"
        min={0}
        max={count - 1}
        value={player.index}
        onChange={(e) => player.seek(Number(e.target.value))}
      />
      <span className="mono frame-position" data-testid="frame-position">
        {player.index + 1} / {count}
      </span>
    </div>
  );
}
