import { Link } from "react-router";
import { useExperimentParams } from "../hooks/useExperimentParams";
import { usePlayback } from "../hooks/usePlayback";
import { batchScenarios } from "../content/scenarios";
import { simulateBatching } from "../simulation/batching";
import BatchTimeline, { requestColors } from "../visualizations/BatchTimeline";
import ParameterControl from "../components/ParameterControl";
import PlaybackControls from "../components/PlaybackControls";
import ShareConfig from "../components/ShareConfig";
export default function ContinuousBatching() {
  const { values, update, warning } = useExperimentParams(
    "continuous-batching",
  );
  const scenario = batchScenarios.find((s) => s.id === values.scenario)!;
  const capacity = Number(values.capacity);
  const fixed = simulateBatching(scenario.requests, capacity, "static");
  const continuous = simulateBatching(
    scenario.requests,
    capacity,
    "continuous",
  );
  const count = Math.max(fixed.frames.length, continuous.frames.length);
  const player = usePlayback({
    frameCount: count,
    intervalMs: 600,
    resetKey: JSON.stringify(values),
  });
  const gain = fixed.frames.length / continuous.frames.length;
  return (
    <>
      <div className="experiment-layout">
        <section className="dark-board" data-playback-surface>
          <div className="board-heading">
            <span className="mono">SCHEDULER / SAME REQUESTS, SAME CLOCK</span>
            <span className="board-label">机制模拟</span>
          </div>
          <div className="pool-stats">
            <div>
              <span>静态批完成时间</span>
              <strong>
                {fixed.frames.length} <small>模拟单位</small>
              </strong>
            </div>
            <div>
              <span>连续批完成时间</span>
              <strong>
                {continuous.frames.length} <small>模拟单位</small>
              </strong>
            </div>
          </div>
          <BatchTimeline
            label="静态批 · 等整批完成"
            result={fixed}
            index={player.index}
            capacity={capacity}
            maxTicks={count}
          />
          <BatchTimeline
            label="连续批 · 空位立即补充"
            result={continuous}
            index={player.index}
            capacity={capacity}
            maxTicks={count}
          />
          <div className="batch-request-legend">
            {scenario.requests.map((r, i) => (
              <span key={r.id}>
                <i style={{ background: requestColors[i % 6] }} />
                {r.id} · 到达 {r.arrival} / 输出 {r.outputTokens}
              </span>
            ))}
          </div>
          <PlaybackControls player={player} count={count} />
          <p className="dark-description">
            {gain > 1
              ? `当前教学负载的完成时间比值为 ${gain.toFixed(2)}×。`
              : "当前负载没有完成时间收益；槽位数与输出长度决定是否存在可填补的空档。"}
            每 Decode 步固定 1 单位，两个模式都忽略 Prefill
            成本、通信、缓存约束与内核形状变化，不代表 GPU 实测加速。
          </p>
        </section>
        <aside className="parameter-panel">
          <div className="eyebrow">SCHEDULING CONTROLS</div>
          <h2>
            让空出的槽位，
            <br />
            接住下一个请求。
          </h2>
          <label className="select-field">
            请求场景
            <select
              aria-label="请求场景"
              value={scenario.id}
              onChange={(e) => update("scenario", e.target.value)}
            >
              {batchScenarios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </label>
          <ParameterControl
            label="同时运行槽位"
            value={capacity}
            min={1}
            max={4}
            onChange={(n) => update("capacity", n)}
          />
          <div className="completion-order">
            <h3>完成时间 · 模拟单位</h3>
            {scenario.requests.map((r) => (
              <div key={r.id}>
                <span>{r.id}</span>
                <span>静态 {fixed.completions[r.id]}</span>
                <span>连续 {continuous.completions[r.id]}</span>
              </div>
            ))}
          </div>
          <ShareConfig />
        </aside>
      </div>
      {warning && <p className="notice">{warning}</p>}
      <div className="experiment-reading">
        <h2>占用更多，不等于延迟更低。</h2>
        <p>
          实际系统需要在 Token 预算、Prefill 与
          Decode、公平性和显存容量之间做选择。这里隔离了“每轮补位”机制，方便理解空闲来自哪里。
        </p>
        <Link className="text-link" to="/frameworks/vllm?module=scheduler">
          查看真实调度器 →
        </Link>
      </div>
    </>
  );
}
