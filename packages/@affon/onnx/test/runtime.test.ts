import { test, expect, beforeAll, afterAll } from "std:test";
import fs from "std:fs";
import { getEnv, run } from "std:process";
import { tensor } from "affon:compute";
import type { Device } from "affon:compute";
import { prepare_pad } from "../src/spatial.ts";
import { load_graph, semantic_loss_report } from "../src/index.ts";
const directory = "packages/@affon/onnx/test/fixtures";
const device = (getEnv("AFFON_DEVICE") ?? "cpu") as Device;
let temporary = "";
beforeAll(async () => {
  temporary = (
    await run({ cmd: "mktemp", args: ["-d", "/tmp/affon-onnx-test.XXXXXX"] })
  ).stdout.trim();
});
afterAll(async () => {
  if (temporary) await run({ cmd: "rm", args: ["-rf", temporary] });
});
test("import reports preserved inferred decomposed unsupported and source-export-lost meaning", () => {
  const report = semantic_loss_report(directory);
  expect(report.format).toBe("affon-program-import-semantics/v1");
  expect(Array.isArray(report.preserved)).toBe(true);
  expect(Array.isArray(report.inferred)).toBe(true);
  expect(Array.isArray(report.decomposed)).toBe(true);
  expect(Array.isArray(report.unsupported)).toBe(true);
  expect(report.source_export_lost.length > 0).toBe(true);
  expect(report.unsupported.length).toBe(0);
});
test("generic graph matches independent ONNX outputs including scalar Gather and affine normalization", () => {
  const reference = JSON.parse(fs.readFileSync(`${directory}/reference.json`));
  const model = load_graph(directory, device);
  for (let repeat = 0; repeat < 2; repeat++) {
    const outputs = model.forward({
      x: tensor(reference.input, { dtype: "f32", device }),
    });
    for (const [name, value] of Object.entries(reference.expected)) {
      const expected = (value as number[]).flat(Infinity) as number[];
      const actual = (outputs[name].to_array() as number[]).flat(
        Infinity,
      ) as number[];
      expect(actual.length).toBe(expected.length);
      expect(
        actual.every(
          (x, i) =>
            Number.isFinite(x) &&
            Math.abs(x - expected[i]) <= 1e-5 + 1e-5 * Math.abs(expected[i]),
        ),
      ).toBe(true);
    }
    expect(outputs.selected.shape).toEqual([1, 4]);
    expect(outputs.result.shape).toEqual([1, 2]);
  }
});
test("graph input contract rejects wrong shape, dtype and names", () => {
  const model = load_graph(directory, device);
  const reference = JSON.parse(fs.readFileSync(`${directory}/reference.json`));
  expect(() =>
    model.forward({ x: tensor([1, 2], { dtype: "f32", device }) }),
  ).toThrow();
  expect(() =>
    model.forward({
      x: tensor(reference.input, { dtype: "f64", device: "cpu" }),
    }),
  ).toThrow();
  expect(() => model.forward({})).toThrow();
  expect(() =>
    model.forward({ other: tensor(reference.input, { dtype: "f32", device }) }),
  ).toThrow();
});
test("unsupported operations and missing graph dependencies fail before loading weights", () => {
  const manifest = JSON.parse(fs.readFileSync(`${directory}/graph.json`));
  manifest.nodes[0].op = "Unsupported";
  fs.writeFileSync(`${temporary}/graph.json`, JSON.stringify(manifest));
  expect(() => load_graph(temporary, device)).toThrow();
  manifest.nodes[0].op = "Conv";
  manifest.nodes[0].inputs[0] = "missing";
  fs.writeFileSync(`${temporary}/graph.json`, JSON.stringify(manifest));
  expect(() => load_graph(temporary, device)).toThrow();
});

test("spatial graph matches ONNX for batches, grouped/depthwise/overlapping convolution, asymmetric pads and dilation", () => {
  const reference = JSON.parse(
    fs.readFileSync(`${directory}/spatial/reference.json`),
  );
  const model = load_graph(`${directory}/spatial`, device);
  const outputs = model.forward({
    x: tensor(reference.input, { dtype: "f32", device }),
  });
  for (const [name, value] of Object.entries(reference.expected)) {
    const expected = (value as number[]).flat(Infinity) as number[];
    const actual = (outputs[name].to_array() as number[]).flat(
      Infinity,
    ) as number[];
    expect(actual.length).toBe(expected.length);
    expect(
      actual.every(
        (v, i) =>
          Number.isFinite(v) &&
          Math.abs(v - expected[i]) <= 1e-5 + 1e-5 * Math.abs(expected[i]),
      ),
    ).toBe(true);
  }
});

test("constant padding stays zero even when the gathered border source is nonfinite", () => {
  const pad = prepare_pad([1, 1, 1, 2], [1, 1, 1, 1], device);
  const values = (
    pad(
      tensor([[[[Infinity, NaN]]]], { dtype: "f32", device }),
    ).to_array() as number[]
  ).flat(Infinity) as number[];
  expect(values.length).toBe(12);
  expect(values[5]).toBe(Infinity);
  expect(Number.isNaN(values[6])).toBe(true);
  expect(values.every((v, i) => i === 5 || i === 6 || v === 0)).toBe(true);
});

test('NCW grouped dilated convolution and bounded Slice match ONNX Runtime', () => {
  const root=`${directory}/conv1d`,reference=JSON.parse(fs.readFileSync(`${root}/reference.json`))
  const model=load_graph(root,device),result=model.forward({x:tensor(reference.input,{dtype:'f32',device})})
  for(const name of ['conv','sliced']) {
    const actual=(result[name].to_array() as number[][][]).flat(2),expected=reference.expected[name].flat(2)
    expect(actual.every((x,i)=>Math.abs(x-expected[i])<1e-5)).toBe(true)
  }
})
