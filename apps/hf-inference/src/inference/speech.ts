import { decode_wav } from '../../../../packages/@affon/huggingface/src/index.ts'
import type { InferenceModels, InferenceOptions } from './models.ts'
import { WhisperProgramRuntime } from './program-runtime.ts'

// A missed EOS previously let a synchronous request run for roughly 20 minutes
// on Metal. Short-form UI transcription favors a predictable upper bound.
export const TRANSCRIPTION_MAX_NEW_TOKENS = 32

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
  const runtime = new WhisperProgramRuntime(models.speech.model, device)
  let result: ReturnType<WhisperProgramRuntime['transcribe']>
  try {
    result = runtime.transcribe(
      features,
      undefined,
      undefined,
      TRANSCRIPTION_MAX_NEW_TOKENS,
    )
  }
  finally { runtime.dispose(); features.dispose() }
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
