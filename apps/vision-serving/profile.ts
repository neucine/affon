// Fixed-workload CPU audit: uninstrumented phases and separate operator timings.
import fs from "std:fs";
import { getEnv } from "std:process";
import telemetry from "std:telemetry";
import { load_graph } from "../../packages/@affon/onnx/src/index.ts";
import { process_mobilenet_image } from "../../packages/@affon/huggingface/src/processors/mobilenet.ts";
const directory = getEnv("MODEL_DIR")!;
const count = Number(getEnv("PROFILE_RUNS") ?? "30");
const batch = Number(getEnv("PROFILE_BATCH") ?? "10");
const mode = getEnv("PROFILE_MODE") ?? "all";
const rgb = Array.from({ length: 260 }, (_, y) =>
  Array.from({ length: 320 }, (_, x) =>
    [0, 1, 2].map((c) => (x * 3 + y * 5 + c * 47) % 256),
  ),
);
const model = load_graph(directory, "cpu");
const pixels = process_mobilenet_image(`${directory}/source`, rgb, "cpu");
let output = model.forward({ pixels }).logits;
for (let i = 0; i < 5; i++) output = model.forward({ pixels }).logits;
function metrics() {
  return telemetry.metrics().filter((m) => m.scope.startsWith("compute."));
}
const before = metrics();
const graph: number[] = [],
  readback: number[] = [],
  preprocessing: number[] = [];
for (let i = 0; i < count; i++) {
  if (mode !== "preprocess") {
    const start = Date.now();
    for (let j = 0; j < batch; j++) output = model.forward({ pixels }).logits;
    graph.push((Date.now() - start) / batch);
    const read = Date.now();
    for (let j = 0; j < batch; j++) output.to_array();
    readback.push((Date.now() - read) / batch);
  }
  if (mode !== "graph") {
    const start = Date.now();
    for (let j = 0; j < batch; j++)
      process_mobilenet_image(`${directory}/source`, rgb, "cpu");
    preprocessing.push((Date.now() - start) / batch);
  }
}
const after = metrics();
const operators: any[] = [];
if (mode === "all")
  for (let i = 0; i < 10; i++)
    model.forward({ pixels }, (event) => operators.push(event));
fs.writeFileSync(
  getEnv("PROFILE_OUTPUT")!,
  JSON.stringify(
    {
      batch,
      graph_ms: graph,
      readback_ms: readback,
      preprocessing_ms: preprocessing,
      operators,
      before,
      after,
      logits: output.to_array(),
    },
    null,
    2,
  ),
);
