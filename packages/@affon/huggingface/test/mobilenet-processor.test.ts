import { test, expect, beforeAll, afterAll } from "std:test";
import fs from "std:fs";
import { run } from "std:process";
import { load_processor } from "../src/processor.ts";
let directory = "";
beforeAll(async () => {
  directory = (
    await run({
      cmd: "mktemp",
      args: ["-d", "/tmp/affon-mobile-processor.XXXXXX"],
    })
  ).stdout.trim();
  fs.writeFileSync(
    `${directory}/config.json`,
    JSON.stringify({ model_type: "mobilenet_v2" }),
  );
  fs.writeFileSync(
    `${directory}/preprocessor_config.json`,
    JSON.stringify({
      size: { shortest_edge: 8 },
      crop_size: { height: 6, width: 6 },
    }),
  );
});
afterAll(async () => {
  if (directory) await run({ cmd: "rm", args: ["-rf", directory] });
});
test("MobileNet RGB preprocessing matches HF across aspect ratios and upsampling", () => {
  const processor = load_processor(directory, { task: "image-classification" });
  const cases = JSON.parse(
    fs.readFileSync(
      "packages/@affon/huggingface/test/fixtures/mobilenet-processor.json",
    ),
  );
  for (const { height, width, pixels } of cases) {
    const rgb = Array.from({ length: height }, (_, y) =>
      Array.from({ length: width }, (_, x) =>
        [0, 1, 2].map((c) => (x * 3 + y * 5 + c * 47) % 256),
      ),
    );
    expect(processor.process(rgb).to_array()).toEqual(pixels);
  }
  expect(() => processor.process([[[256, 0, 0]]])).toThrow();
  expect(() => processor.process([])).toThrow();
});
test("MobileNet rejects unsupported resize filters", () => {
  fs.writeFileSync(
    `${directory}/preprocessor_config.json`,
    JSON.stringify({ resample: 3 }),
  );
  expect(() =>
    load_processor(directory, { task: "image-classification" }).process([
      [[0, 0, 0]],
    ]),
  ).toThrow();
});
