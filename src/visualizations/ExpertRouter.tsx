export default function ExpertRouter({
  routed,
  expertCount,
  sharedExperts,
}: {
  routed: { id: number; weight: number }[];
  expertCount: number;
  sharedExperts: number;
}) {
  return (
    <div className="expert-router">
      <div className="expert-summary">
        <span className="mono">TOP-K ROUTER / TEACHING LOGITS</span>
        <strong>
          {routed.length} routed + {sharedExperts} shared
        </strong>
        <span>从 {expertCount} 个路由专家中选择；共享专家另计</span>
      </div>
      <div
        className="expert-grid"
        role="img"
        aria-label={`${expertCount} 个专家中激活 ${routed.length} 个：${routed.map((x) => x.id).join("、")}`}
      >
        {Array.from({ length: expertCount }, (_, id) => {
          const r = routed.find((x) => x.id === id);
          return (
            <span
              key={id}
              className={r ? "active" : ""}
              title={`专家 ${id}${r ? " · " + (r.weight * 100).toFixed(1) + "%" : ""}`}
            />
          );
        })}
      </div>
      <div className="expert-weights">
        {routed.map((x) => (
          <span key={x.id}>
            E{x.id} <b>{(x.weight * 100).toFixed(1)}%</b>
          </span>
        ))}
      </div>
      <div className="shared-expert">
        <span>共享专家 × {sharedExperts}</span>
        <span>独立分支 → 与路由输出合并</span>
      </div>
      <p className="dark-description">
        方格表示专家身份，不表示参数大小。教学路由分数，未运行实际专家权重。
      </p>
    </div>
  );
}
