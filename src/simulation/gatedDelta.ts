export type Matrix2 = [[number, number], [number, number]];
export type DeltaFrame = {
  state: Matrix2;
  decayed: Matrix2;
  buffer: number[];
  conv: number;
  key: number[];
  value: number[];
  error: number[];
  output: number[];
};
// value × key convention, paper equation 10; all vectors below are teaching fixtures.
export function buildDeltaTrace(alpha: number, beta: number): DeltaFrame[] {
  if (
    !Number.isFinite(alpha) ||
    !Number.isFinite(beta) ||
    alpha < 0 ||
    alpha > 1 ||
    beta < 0 ||
    beta > 1
  )
    throw Error("门控系数必须在 0 到 1 之间");
  let state: Matrix2 = [
      [0, 0],
      [0, 0],
    ],
    buffer = [0, 0, 0];
  const frames: DeltaFrame[] = [
    {
      state,
      decayed: state,
      buffer,
      conv: 0,
      key: [0, 0],
      value: [0, 0],
      error: [0, 0],
      output: [0, 0],
    },
  ];
  const fixtures = [
    { x: 1, k: [1, 0], v: [1, 0.5], q: [1, 0] },
    { x: -0.5, k: [0, 1], v: [0.2, 1], q: [0, 1] },
    { x: 0.8, k: [0.6, 0.8], v: [0.7, -0.3], q: [0.6, 0.8] },
    { x: 0.2, k: [1, 0], v: [0.1, 0.8], q: [1, 0] },
  ];
  for (const f of fixtures) {
    buffer = [...buffer.slice(1), f.x];
    const decayed = state.map((row) => row.map((x) => alpha * x)) as Matrix2;
    const prediction = decayed.map((row) =>
      row.reduce((s, x, j) => s + x * f.k[j], 0),
    );
    const error = f.v.map((x, i) => x - prediction[i]);
    state = decayed.map((row, i) =>
      row.map((x, j) => x + beta * error[i] * f.k[j]),
    ) as Matrix2;
    frames.push({
      state,
      decayed,
      buffer,
      conv: buffer[0] * 0.2 + buffer[1] * 0.3 + buffer[2] * 0.5,
      key: f.k,
      value: f.v,
      error,
      output: state.map((row) => row.reduce((s, x, j) => s + x * f.q[j], 0)),
    });
  }
  return frames;
}
