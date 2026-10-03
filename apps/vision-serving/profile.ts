// Fixed-workload CPU audit: uninstrumented phases and separate operator timings.
import fs from "std:fs";
import { getEnv } from "std:process";
import telemetry from "std:telemetry";
import { load_graph } from "../../packages/@affon/onnx/src/index.ts";
import { process_mobilenet_image } from "../../packages/@affon/huggingface/src/processors/mobilenet.ts";
import { Session, type Tensor } from 'affon:compute'
const directory = getEnv("MODEL_DIR")!;
const count = Number(getEnv("PROFILE_RUNS") ?? "30");
const batch = Number(getEnv("PROFILE_BATCH") ?? "10");
const mode = getEnv("PROFILE_MODE") ?? "all";
const rgb = Array.from({ length: 260 }, (_, y) =>
  Array.from({ length: 320 }, (_, x) =>
    [0, 1, 2].map((c) => (x * 3 + y * 5 + c * 47) % 256),
  ),
);
const model = load_graph(directory);
const session = new Session({ device: 'cpu' })
const state = session.initialize(model.forward, { parameters: model.parameters })
const executable = session.compile(model.forward)
const run = () => {
  const input = session.tensor(pixels.to_array())
  try {
    const value = executable.run({ pixels: input }, state) as Tensor | Tensor[]
    return (Array.isArray(value) ? value : [value])[model.output_names.indexOf('logits')]
  } finally { input.dispose() }
}
const pixels = process_mobilenet_image(`${directory}/source`, rgb, "cpu");
let output = run();
for (let i = 0; i < 5; i++) { output.dispose(); output = run() }
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
    for (let j = 0; j < batch; j++) { output.dispose(); output = run() }
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
const operators = model.forward.inspect().nodes.map(node => ({ id: node.id, op: node.op, shape: node.spec.shape }));
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
output.dispose(); pixels.dispose(); state.dispose(); session.dispose()
