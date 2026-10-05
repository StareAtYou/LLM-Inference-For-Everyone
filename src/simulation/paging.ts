export type PagedState = {
  capacity: number;
  blockSize: number;
  requests: Record<string, { tokens: number; blocks: number[] }>;
  freeBlocks: number[];
};
export function createPagedState(
  capacity: number,
  blockSize: number,
): PagedState {
  if (
    !Number.isInteger(capacity) ||
    capacity < 1 ||
    capacity > 128 ||
    !Number.isInteger(blockSize) ||
    blockSize < 1 ||
    blockSize > 64
  )
    throw new Error("块池参数无效");
  return {
    capacity,
    blockSize,
    requests: {},
    freeBlocks: Array.from({ length: capacity }, (_, i) => i),
  };
}
export function stepPaged(
  state: PagedState,
  event:
    | { type: "allocate"; id: string; tokens: number }
    | { type: "release"; id: string },
): PagedState {
  if (
    !/^[A-Za-z0-9-]{1,40}$/.test(event.id) ||
    ["constructor", "prototype"].includes(event.id)
  )
    throw new Error("请求 ID 无效");
  if (event.type === "release") {
    if (!Object.hasOwn(state.requests, event.id)) throw new Error("请求不存在");
    const requests = { ...state.requests };
    const blocks = requests[event.id].blocks;
    delete requests[event.id];
    return {
      ...state,
      requests,
      freeBlocks: [...state.freeBlocks, ...blocks].sort((a, b) => a - b),
    };
  }
  if (Object.hasOwn(state.requests, event.id)) throw new Error("请求已经存在");
  if (
    !Number.isInteger(event.tokens) ||
    event.tokens < 1 ||
    event.tokens > 262144
  )
    throw new Error("Token 数无效");
  const need = Math.ceil(event.tokens / state.blockSize);
  if (need > state.freeBlocks.length)
    throw new Error(
      `空闲块不足：需要 ${need} 块，仅剩 ${state.freeBlocks.length} 块`,
    );
  return {
    ...state,
    requests: {
      ...state.requests,
      [event.id]: {
        tokens: event.tokens,
        blocks: state.freeBlocks.slice(0, need),
      },
    },
    freeBlocks: state.freeBlocks.slice(need),
  };
}
