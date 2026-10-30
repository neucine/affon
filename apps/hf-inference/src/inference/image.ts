import { type InferenceModels } from "./models.ts";
import type { InferenceOptions } from "./models.ts";
import { rank_classes } from "./classification.ts";
import { VitProgramRuntime, GraphProgramRuntime } from "./program-runtime.ts";

const nativeRuntimes = new WeakMap<InferenceModels, Map<object, VitProgramRuntime>>()

function native_runtime(models: InferenceModels, model: ConstructorParameters<typeof VitProgramRuntime>[0]) {
  let entries = nativeRuntimes.get(models)
  if (!entries) { entries = new Map(); nativeRuntimes.set(models, entries) }
  let runtime = entries.get(model)
  if (!runtime) { runtime = new VitProgramRuntime(model, models.device); entries.set(model, runtime) }
  return runtime
}

export function dispose_image_runtimes(models: InferenceModels) {
  const entries = nativeRuntimes.get(models)
  if (!entries) return
  nativeRuntimes.delete(models)
  for (const runtime of entries.values()) runtime.dispose()
}

/** Preprocess packed RGB8 for the selected model, run inference, and rank logits. */
export function classify_image(
  models: InferenceModels,
  pixels: Uint8Array,
  width: number,
  height: number,
  device: InferenceOptions["device"],
  model_key = "vit-native",
) {
  const vision = Object.hasOwn(models.images, model_key)
    ? models.images[model_key]
    : undefined;
  if (!vision) throw Error("Unknown or unavailable image model");
  const start = Date.now();
  const rgb = Array.from({ length: height }, (_, y) =>
    Array.from({ length: width }, (_, x) =>
      Array.from({ length: 3 }, (_, c) => pixels[(y * width + x) * 3 + c]),
    ),
  );
  const processed = vision.processor.process(rgb);
  const inference_start = Date.now();
  const result = vision.backend === 'native'
    ? (() => {
        const runtime = native_runtime(models, vision.model)
        const input = runtime.import(processed)
        try {
          const outputs = runtime.forward(input)
          return { output: outputs.output, dispose: () => { outputs.dispose(); input.dispose() } }
        } catch (error) { input.dispose(); throw error }
      })()
    : (() => {
        const runtime = new GraphProgramRuntime(vision.model, device)
        const outputs = runtime.forward({ [vision.model.input_name]: processed })
        const output = outputs[vision.model.output_name]
        return { output, dispose: () => { for (const value of Object.values(outputs)) value.dispose(); runtime.dispose() } }
      })();
  let logits: number[]
  try { logits = (result.output.to_array() as number[][])[0] }
  finally { result.dispose(); processed.dispose() }
  const inference_ms = Date.now() - inference_start;
  const predictions = rank_classes(logits, vision.model.config.id2label ?? {});
  return {
    predictions,
    elapsed_ms: Date.now() - start,
    inference_ms,
    model: vision.id,
    model_key,
    label: vision.label,
    backend: vision.backend,
    device,
  };
}
