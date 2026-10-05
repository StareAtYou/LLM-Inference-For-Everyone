import type { FrameworkDefinition } from "../content/frameworks";
export const edgeLabels = {
  call: "调用",
  message: "进程消息",
  data: "数据关系",
};
export default function FrameworkGraph({
  framework,
  selected,
  onSelect,
}: {
  framework: FrameworkDefinition;
  selected: string;
  onSelect: (id: string) => void;
}) {
  const positions =
    framework.id === "vllm"
      ? [
          [0, 0],
          [1, 0],
          [2, 0],
          [2, 1],
          [1, 1],
          [0, 1],
          [2, 2],
          [1, 2],
        ]
      : [
          [0, 0],
          [1, 0],
          [2, 0],
          [1, 1],
          [2, 1],
          [2, 2],
          [0, 2],
        ];
  const position = (id: string) =>
    positions[framework.modules.findIndex((m) => m.id === id)];
  const adjacent = framework.edges.filter(
    (e) => e.from === selected || e.to === selected,
  );
  const neighbors = new Set(adjacent.flatMap((e) => [e.from, e.to]));
  return (
    <>
      <div className="framework-graph">
        <svg
          className="framework-wires"
          viewBox="0 0 600 410"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            <marker
              id="arrow-call"
              markerWidth="7"
              markerHeight="7"
              refX="6"
              refY="3"
              orient="auto"
            >
              <path d="M0,0 L6,3 L0,6" fill="none" stroke="#9980b5" />
            </marker>
          </defs>
          {framework.edges.map((e, i) => {
            const a = position(e.from),
              b = position(e.to);
            const x1 = 100 + a[0] * 200,
              y1 = 55 + a[1] * 150,
              x2 = 100 + b[0] * 200,
              y2 = 55 + b[1] * 150;
            return (
              <path
                className={
                  e.kind +
                  (e.from === selected || e.to === selected
                    ? " highlighted"
                    : "")
                }
                key={i}
                d={`M${x1},${y1} L${x2},${y2}`}
                markerEnd="url(#arrow-call)"
              />
            );
          })}
        </svg>
        {framework.modules.map((m, i) => (
          <button
            key={m.id}
            style={{
              gridColumn: positions[i][0] + 1,
              gridRow: positions[i][1] + 1,
            }}
            onClick={() => onSelect(m.id)}
            className={
              "framework-node " +
              (m.id === selected ? "selected" : "") +
              (neighbors.has(m.id) ? " adjacent" : "")
            }
            aria-pressed={m.id === selected}
            aria-label={m.name}
          >
            <small className="mono">
              0{i + 1} / {m.id.toUpperCase()}
            </small>
            <strong>
              {m.name
                .replace(/(Manager|Runner|Pool|Core|Cache)$/, " $1")
                .replace("GPUModel", "GPU Model")}
            </strong>
            <span>
              {m.id === "cache" || m.id === "blocks"
                ? "STATE & MEMORY"
                : m.id === "model" || m.id === "runner"
                  ? "COMPUTE"
                  : "REQUEST & CONTROL"}
            </span>
          </button>
        ))}
      </div>
      <div className="chart-legend framework-legend">
        <span>
          <i className="legend-line" />
          调用 · 可折叠中间层
        </span>
        <span>
          <i className="legend-line dashed" />
          进程消息
        </span>
        <span>
          <i className="legend-line data" />
          数据关系
        </span>
      </div>
      <div className="edge-details">
        <span className="mono">SELECTED MODULE / CONNECTIONS</span>
        {adjacent.map((e, i) => (
          <button
            key={i}
            onClick={() => onSelect(e.from === selected ? e.to : e.from)}
          >
            <span className={"edge-kind " + e.kind}>{edgeLabels[e.kind]}</span>
            <span>
              {framework.modules.find((m) => m.id === e.from)?.name} →{" "}
              {framework.modules.find((m) => m.id === e.to)?.name}
            </span>
            <small>{e.label}</small>
          </button>
        ))}
      </div>
    </>
  );
}
