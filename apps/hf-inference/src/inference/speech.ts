import { decode_wav } from '../../../../packages/@affon/huggingface/src/index.ts'
import type { InferenceModels, InferenceOptions } from './models.ts'
export function transcribe_audio(
  models: InferenceModels,
  bytes: Uint8Array,
  device: InferenceOptions['device'],
) {
  if (!models.speech) throw Error('Whisper is not configured')
  if (bytes.length > 16 * 1024 * 1024)
    throw Error('Choose a WAV no larger than 16 MiB')
  const start = Date.now(),
    audio = decode_wav(bytes)
  const features = models.speech.processor.process(
    audio.samples,
    audio.sampling_rate,
  )
  const preprocessing_ms = Date.now() - start
  const result = models.speech.model.transcribe(features)
  return {
    ...result,
    preprocessing_ms,
    elapsed_ms: Date.now() - start,
    model: models.speech.id,
    backend: 'onnx',
    device,
    duration_seconds: audio.duration_seconds,
    sampling_rate: audio.sampling_rate,
    channels: audio.channels,
  }
}
