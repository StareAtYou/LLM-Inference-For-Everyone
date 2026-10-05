import { expect, test } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ContinuousJourneyScene from "../../src/visualizations/ContinuousJourneyScene";
import { buildInferenceTrace } from "../../src/simulation/inferenceTrace";
import { getModel } from "../../src/content/models";

test("continuous_tensor_preview_crops_columns_without_mixing_original_rows", () => {
  const trace = buildInferenceTrace({
    modelId: "qwen38-dense",
    scenarioId: "sky",
    mode: "single",
    capacity: 1,
    temperature: 0,
    topK: 6,
    topP: 1,
    outputLimit: 1,
  });
  const original = trace.find((f) => f.stage === "ffn")!;
  const tensor = original.tensors.find((t) => t.name === "SiLU 门控")!;
  expect(tensor.cols).toBe(6);
  const frame = { ...original, tensors: [tensor] };
  const html = renderToStaticMarkup(
    createElement(ContinuousJourneyScene, {
      trace,
      frame,
      fraction: 0.5,
      position: frame.index + 0.5,
      model: getModel("qwen38-dense"),
      requestId: "R1",
      onSeek: () => {},
    }),
  );
  const grid = html.match(/class="cj-values"[\s\S]*?<\/div>/)![0];
  const displayed = [...grid.matchAll(/>(-?\d+\.\d+)<\/span>/g)].map((m) =>
    Number(m[1]),
  );
  expect(displayed).toEqual(
    [...tensor.values.slice(0, 4), ...tensor.values.slice(6, 10)].map((n) =>
      Number(n.toFixed(2)),
    ),
  );
});
