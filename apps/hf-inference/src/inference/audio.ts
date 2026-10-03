import { decode_wav } from '../../../../packages/@affon/huggingface/src/index.ts'
import type { InferenceModels, InferenceOptions } from './models.ts'
import { rank_classes } from './classification.ts'
import { GraphProgramRuntime } from './program-runtime.ts'

/** Decode, downmix, resample and classify WAV bytes without browser/Python processing. */
export function classify_audio(
  models: InferenceModels,
  bytes: Uint8Array,
  device: InferenceOptions['device'],
) {
  if (!models.audio) throw Error('Audio model is not configured')
  if (bytes.length > 16 * 1024 * 1024)
    throw Error('Choose a WAV file no larger than 16 MiB')
  const start = Date.now(),
    audio = decode_wav(bytes)
  const features = models.audio.processor.process(
    audio.samples,
    audio.sampling_rate,
  )
  const inference_start = Date.now()
  const runtime = new GraphProgramRuntime(models.audio.model, device)
  const outputs = runtime.forward({ [models.audio.model.input_name]: features })
  const output = outputs[models.audio.model.output_name]
  let logits: number[]
  try { logits = (output.to_array() as number[][])[0] }
  finally { for (const value of Object.values(outputs)) value.dispose(); runtime.dispose(); features.dispose() }
  const inference_ms = Date.now() - inference_start
  const predictions = rank_classes(logits, models.audio.model.config.id2label)
  return {
    predictions,
    model: models.audio.id,
    backend: 'onnx',
    device,
    inference_ms,
    elapsed_ms: Date.now() - start,
    duration_seconds: audio.duration_seconds,
    sampling_rate: audio.sampling_rate,
    channels: audio.channels,
    processed_seconds: Math.min(
      audio.duration_seconds,
      models.audio.processor.max_samples / 16000,
    ),
    truncated:
      audio.duration_seconds > models.audio.processor.max_samples / 16000,
  }
}
