import fs from "std:fs";
import { getEnv } from "std:process";
import telemetry from "std:telemetry";
import checkpoint from "affon:checkpoint";
import { Session, type Device, type Tensor } from "affon:compute";
import { load_graph } from "../../../packages/@affon/onnx/src/index.ts";
import { load_vit } from "../../../packages/@affon/huggingface/src/adapters/vit.ts";
import { load_model } from "../../../packages/@affon/huggingface/src/index.ts";
import type { OnnxImageClassifier } from "../../../packages/@affon/huggingface/src/index.ts";
import { compare_values } from "../audit/compare.ts";
const directory = getEnv("AFFON_ONNX_DIR")!;
const modelDirectory = getEnv("AFFON_HF_MODEL_DIR")!;
const device = (getEnv("AFFON_DEVICE") ?? "cpu") as Device;
const iterations = Number(getEnv("AFFON_ONNX_ITERATIONS") ?? 3);
const warmups = Number(getEnv("AFFON_ONNX_WARMUPS") ?? 1);
if (
  !Number.isInteger(iterations) ||
  iterations < 1 ||
  iterations > 100 ||
  !Number.isInteger(warmups) ||
  warmups < 0 ||
  warmups > 10
)
  throw Error("Invalid audit run counts");
const route = getEnv("AFFON_ONNX_ROUTE") ?? "graph";
if (
  !directory ||
  (route !== "graph" && !modelDirectory) ||
  !getEnv("AFFON_HF_REPORT")
)
  throw Error("Set AFFON_ONNX_DIR, AFFON_HF_MODEL_DIR and AFFON_HF_REPORT");
if (!["graph", "adapter", "hf-onnx"].includes(route) || !["cpu", "metal"].includes(device))
  throw Error("Use graph/adapter and cpu/metal");
const memory = () =>
  Object.fromEntries(
    telemetry
      .metrics()
      .filter((x) =>
        [
          "compute.storage.live_bytes",
          "compute.storage.peak_bytes",
          "runtime.memory.resident_bytes",
        ].includes(`${x.scope}.${x.name}`),
      )
      .map((x) => [`${x.scope}.${x.name}`, x.value]),
  );
const start = Date.now();
const model =
  route === "graph"
    ? load_graph(directory, device)
    : route === "hf-onnx" ? load_model(modelDirectory, {task:"image-classification",backend:"onnx",graph_dir:directory,device}) : load_vit(modelDirectory, device);
const load_ms = Date.now() - start,
  after_load = memory();
const ref = checkpoint.load(`${directory}/reference.safetensors`) as Record<
  string,
  Tensor
>;
const session = new Session({ device });
const flatten = (x: Tensor) =>
  (x.to_array() as number[]).flat(Infinity) as number[];
const results = [];
for (let i = 0; i < 2; i++) {
  const input = session.tensor(ref[`case_${i}.pixels`].to_array() as any, { dtype: ref[`case_${i}.pixels`].dtype });
  const times: number[] = [];
  let actual: number[] = [];
  for (let run = -warmups; run < iterations; run++) {
    const start = Date.now();
    const out =
      route === "graph"
        ? (model as ReturnType<typeof load_graph>).forward({ pixels: input })
            .logits
        : (model as ReturnType<typeof load_vit> | OnnxImageClassifier).forward(input).output;
    actual = flatten(out);
    out.dispose();
    if (run >= 0) times.push(Date.now() - start);
  }
  const onnx = compare_values(actual, flatten(ref[`case_${i}.onnx`]));
  const pytorch = compare_values(actual, flatten(ref[`case_${i}.pytorch`]));
  const top1 = actual.indexOf(Math.max(...actual));
  results.push({
    case: i,
    onnx,
    pytorch,
    top1,
    forward_ms: times,
    memory: memory(),
  });
  input.dispose();
}
const report = {
  route,
  device,
  semantic_loss:
    route === "graph"
      ? (model as ReturnType<typeof load_graph>).semanticLoss
      : null,
  warmups,
  iterations,
  load_ms,
  after_load,
  results,
  passed: results.every((x) => x.onnx.passed && x.pytorch.passed),
};
fs.writeFileSync(getEnv("AFFON_HF_REPORT")!, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
if (!report.passed) throw Error("ONNX parity check failed; see saved report");
session.dispose();
(model as any).dispose?.();
