import { type InferenceModels } from "./models.ts";
import type { InferenceOptions } from "./models.ts";
import { rank_classes } from "./classification.ts";
import { execute_vit, GraphProgramRuntime } from "./program-runtime.ts";

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
    ? execute_vit(vision.model, processed, device)
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
