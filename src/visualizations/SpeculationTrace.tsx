import type { SpeculationResult } from "../simulation/speculation";
export default function SpeculationTrace({
  result,
  index,
}: {
  result: SpeculationResult;
  index: number;
}) {
  return (
    <div className="speculation-trace">
      {result.rounds.map((round, i) => (
        <div
          className={
            "spec-round " +
            (i >= index ? "unplayed" : "") +
            (i === index - 1 ? " current" : "")
          }
          key={i}
        >
          <div className="round-number mono">
            ROUND {String(i + 1).padStart(2, "0")}
          </div>
          <div className="trace-row">
            <span>草稿</span>
            <div>
              {round.draft.map((t, j) => (
                <span
                  key={j}
                  className={
                    "draft-token " +
                    (j < round.accepted.length ? "accepted" : "rejected")
                  }
                >
                  {t}
                  <small>{j < round.accepted.length ? "接受" : "丢弃"}</small>
                </span>
              ))}
            </div>
          </div>
          <div className="trace-row">
            <span>目标验证</span>
            <div>
              {round.accepted.map((t, j) => (
                <span className="verified-token" key={j}>
                  {t}
                </span>
              ))}
              {round.correction && (
                <span className="correction-token">
                  {round.correction}
                  <small>{round.rejected.length ? "修正" : "补充"}</small>
                </span>
              )}
            </div>
          </div>
          <p>
            {round.rejected.length
              ? "第一次不匹配后，剩余草稿全部丢弃；后面即使碰巧正确也不接受。"
              : "整个草稿前缀匹配，目标可再补一个 Token；序列结束时不再补充。"}
          </p>
        </div>
      ))}
      <div className="chart-legend">
        <span>
          <i className="legend-dot green" />
          接受前缀
        </span>
        <span>
          <i style={{ background: "#bb8983" }} className="legend-dot" />
          拒绝后的草稿
        </span>
        <span>
          <i className="legend-dot purple" />
          目标修正或补充
        </span>
      </div>
    </div>
  );
}
