import { Link } from "react-router";
import { useExperimentParams } from "../hooks/useExperimentParams";
import { usePlayback } from "../hooks/usePlayback";
import { targetTokens, speculationScenarios } from "../content/scenarios";
import { simulateSpeculation } from "../simulation/speculation";
import SpeculationTrace from "../visualizations/SpeculationTrace";
import ParameterControl from "../components/ParameterControl";
import PlaybackControls from "../components/PlaybackControls";
import ShareConfig from "../components/ShareConfig";
export default function SpeculativeDecoding() {
  const { values, update, warning } = useExperimentParams(
    "speculative-decoding",
  );
  const scenario = speculationScenarios.find((s) => s.id === values.scenario)!;
  const result = simulateSpeculation(
    targetTokens,
    scenario.draft,
    Number(values.draftLength),
    { draft: Number(values.draftCost), verify: Number(values.verifyCost) },
  );
  const player = usePlayback({
    frameCount: result.rounds.length + 1,
    intervalMs: 1000,
    resetKey: JSON.stringify(values),
  });
  const shown = result.rounds
    .slice(0, player.index)
    .flatMap((r) => [...r.accepted, ...(r.correction ? [r.correction] : [])]);
  const accepted = result.rounds.reduce((s, r) => s + r.accepted.length, 0);
  const total = result.rounds.reduce((s, r) => s + r.draft.length, 0);
  const gain = result.baselineCost / result.speculativeCost;
  return (
    <>
      <div className="experiment-layout">
        <section className="dark-board" data-playback-surface>
          <div className="board-heading">
            <span className="mono">DRAFT → VERIFY → ACCEPT</span>
            <span className="board-label">Greedy 教学验证</span>
          </div>
          <div className="pool-stats">
            <div>
              <span>草稿接受率 · 实际轨迹</span>
              <strong>{((accepted / total) * 100).toFixed(1)}%</strong>
            </div>
            <div>
              <span>成本比值 · 普通 / 投机</span>
              <strong>{gain.toFixed(2)}×</strong>
            </div>
          </div>
          <div className="cost-comparison">
            <div>
              <span>普通 Decode</span>
              <i
                style={{
                  width:
                    (result.baselineCost /
                      Math.max(result.baselineCost, result.speculativeCost)) *
                      70 +
                    "%",
                }}
              />
              <b>{result.baselineCost.toFixed(1)}</b>
            </div>
            <div>
              <span>投机验证</span>
              <i
                style={{
                  width:
                    (result.speculativeCost /
                      Math.max(result.baselineCost, result.speculativeCost)) *
                      70 +
                    "%",
                }}
              />
              <b>{result.speculativeCost.toFixed(1)}</b>
            </div>
          </div>
          <PlaybackControls player={player} count={result.rounds.length + 1} />
          <div className="spec-output">
            <span className="mono">
              ACCEPTED OUTPUT / {shown.length} TOKENS
            </span>
            <div className="token-row" data-testid="spec-output">
              {shown.length ? (
                shown.map((t, i) => (
                  <span className="token output" key={i}>
                    {t}
                  </span>
                ))
              ) : (
                <span className="empty-token">单步开始第一轮验证</span>
              )}
            </div>
          </div>
          <SpeculationTrace result={result} index={player.index} />
          <p className="dark-description">
            {gain < 1
              ? "当前成本下投机更慢。"
              : "收益取决于接受前缀和草稿成本。"}
            成本为抽象单位：普通每 Token 计一次目标成本，投机每轮计草稿
            Token成本加一次批验证成本；没有模拟 GPU 并行效率。
          </p>
        </section>
        <aside className="parameter-panel">
          <div className="eyebrow">SPECULATION CONTROLS</div>
          <h2>
            猜得多，
            <br />
            也要验证得划算。
          </h2>
          <label className="select-field">
            草稿场景
            <select
              aria-label="草稿场景"
              value={scenario.id}
              onChange={(e) => update("scenario", e.target.value)}
            >
              {speculationScenarios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </label>
          <ParameterControl
            label="每轮草稿长度"
            value={Number(values.draftLength)}
            min={1}
            max={6}
            onChange={(n) => update("draftLength", n)}
          />
          <ParameterControl
            label="每 Token 草稿成本"
            value={Number(values.draftCost)}
            min={0.1}
            max={3}
            step={0.1}
            onChange={(n) => update("draftCost", n)}
          />
          <ParameterControl
            label="每轮目标验证成本"
            value={Number(values.verifyCost)}
            min={1}
            max={5}
            step={0.5}
            onChange={(n) => update("verifyCost", n)}
          />
          <p className="small-note">
            同一目标序列，参数改变后重新构建验证轨迹。没有执行 Qwen 权重、MTP
            或独立草稿模型。
          </p>
          <ShareConfig />
        </aside>
      </div>
      {warning && <p className="notice">{warning}</p>}
      <div className="experiment-reading">
        <h2>Greedy 匹配，是理解机制的起点。</h2>
        <p>
          本实验保证最终输出与固定目标
          Token序列一致。随机采样下的无损投机还需要概率接受与残差校正，不能把逐项相等判断直接推广过去。
        </p>
        <Link className="text-link" to="/learn/speculation">
          理解 MTP、草稿与概率校正的区别 →
        </Link>
      </div>
    </>
  );
}
