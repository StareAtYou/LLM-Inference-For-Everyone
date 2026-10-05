import type { BatchResult } from "../simulation/batching";
export const requestColors = [
  "#ae8cdd",
  "#87bba1",
  "#d5b079",
  "#80b0d0",
  "#d695ab",
  "#b3bd75",
];
export default function BatchTimeline({
  result,
  index,
  capacity,
  maxTicks,
  label,
}: {
  result: BatchResult;
  index: number;
  capacity: number;
  maxTicks: number;
  label: string;
}) {
  return (
    <div className="batch-timeline">
      <div className="timeline-heading">
        <strong>{label}</strong>
        <span className="mono">{result.frames.length} 模拟单位</span>
      </div>
      <div className="timeline-scroll">
        <div
          className="timeline-grid"
          style={{ gridTemplateColumns: `55px repeat(${maxTicks},22px)` }}
        >
          <span className="timeline-label">时间 →</span>
          {Array.from({ length: maxTicks }, (_, t) => (
            <span className="timeline-tick mono" key={t}>
              {t + 1}
            </span>
          ))}
          {Array.from({ length: capacity }, (_, s) => (
            <div className="timeline-row" key={s}>
              <span className="timeline-label">槽位 {s + 1}</span>
              {Array.from({ length: maxTicks }, (_, t) => {
                const id = result.frames[t]?.slots[s];
                return (
                  <span
                    className={
                      "timeline-cell " +
                      (t > index ? "unplayed" : "") +
                      (t === index ? " current" : "")
                    }
                    key={t}
                    style={{
                      background: id
                        ? requestColors[(id.charCodeAt(0) - 65) % 6]
                        : "#35313e",
                    }}
                    title={`时间 ${t + 1} · 槽位 ${s + 1} · ${id ?? "空闲"}`}
                  >
                    {id ?? "·"}
                  </span>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <p className="timeline-caption">
        已观察 {Math.min(index + 1, maxTicks)} 步；该模式占用 {result.busySlots}{" "}
        / {result.totalSlots} 槽位单位，利用率{" "}
        {result.totalSlots
          ? ((result.busySlots / result.totalSlots) * 100).toFixed(1)
          : 0}
        %。
      </p>
    </div>
  );
}
