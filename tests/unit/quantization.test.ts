import { expect, test } from "vitest";
import { quantizeSymmetric } from "../../src/simulation/quantization";
test("int4_reconstructs_symmetric_endpoints_and_counts_packed_bytes", () => {
  const r = quantizeSymmetric([-1, 0, 1], 4);
  expect(r.codes).toEqual([-7, 0, 7]);
  expect(r.scale).toBeCloseTo(1 / 7);
  expect(r.meanSquaredError).toBe(0);
  expect(r.payloadBytes).toBe(2);
});
test("zeros_are_finite_invalid_weights_rejected_and_int8_has_less_error", () => {
  expect(quantizeSymmetric([0, 0], 4).scale).toBe(1);
  expect(() => quantizeSymmetric([], 4)).toThrow();
  expect(() => quantizeSymmetric([NaN], 8)).toThrow();
  const w = [-0.93, 0.27, 0.68, 1];
  expect(quantizeSymmetric(w, 8).meanSquaredError).toBeLessThan(
    quantizeSymmetric(w, 4).meanSquaredError,
  );
});
