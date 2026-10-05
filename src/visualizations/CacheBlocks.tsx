import type { PagedState } from "../simulation/paging";
export default function CacheBlocks({ state }: { state: PagedState }) {
  const ids = Object.keys(state.requests);
  return (
    <>
      <div
        className="physical-blocks"
        role="img"
        aria-label={`物理池 ${state.capacity} 块，空闲 ${state.freeBlocks.length} 块`}
      >
        {Array.from({ length: state.capacity }, (_, block) => {
          const id = ids.find((id) =>
            state.requests[id].blocks.includes(block),
          );
          const req = id ? state.requests[id] : undefined;
          const logical = req?.blocks.indexOf(block) ?? 0;
          const filled = req
            ? Math.min(state.blockSize, req.tokens - logical * state.blockSize)
            : 0;
          return (
            <div
              key={block}
              className={"physical-block " + (id ? "occupied" : "free")}
              style={
                {
                  "--request-color": [
                    "#ad8be6",
                    "#92cab2",
                    "#e6bb81",
                    "#91b7dc",
                  ][ids.indexOf(id ?? "") % 4],
                } as React.CSSProperties
              }
            >
              <span className="mono">P{block.toString().padStart(2, "0")}</span>
              <strong>{id ?? "FREE"}</strong>
              <div>
                {Array.from(
                  { length: Math.min(state.blockSize, 8) },
                  (_, i) => (
                    <i
                      className={
                        i <
                        Math.ceil(
                          (filled / state.blockSize) *
                            Math.min(state.blockSize, 8),
                        )
                          ? "filled"
                          : ""
                      }
                      key={i}
                    />
                  ),
                )}
              </div>
              <small>
                {filled} / {state.blockSize}
              </small>
            </div>
          );
        })}
      </div>
      <div className="chart-legend">
        <span>
          <i className="legend-dot purple" />
          已分配请求
        </span>
        <span>
          <i className="legend-dot gray" />
          空闲块
        </span>
        <span>每块 {state.blockSize} Token · 槽格压缩示意</span>
      </div>
    </>
  );
}
