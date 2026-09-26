import fs from 'std:fs'
import { getEnv } from 'std:process'
import { load_whisper } from '../../../packages/@affon/huggingface/src/whisper.ts'
import { load_whisper_processor } from '../../../packages/@affon/huggingface/src/whisper-processor.ts'
import { decode_wav } from '../../../packages/@affon/huggingface/src/audio.ts'
import type { Device } from 'affon:compute'
const root = getEnv('AFFON_WHISPER_DIR') ?? '/private/tmp/affon-onnx-whisper'
const device = (getEnv('AFFON_DEVICE') ?? 'metal') as Device
const model = load_whisper(`${root}/source`, {
  task: 'automatic-speech-recognition',
  backend: 'onnx',
  graph_dir: root,
  device,
})
const processor = load_whisper_processor(`${root}/source`, device)
// The harness supplies WAV bytes as JSON; file I/O is outside request timing.
const path = getEnv('PROFILE_AUDIO_BYTES')!
const bytes = new Uint8Array(JSON.parse(fs.readFileSync(path)))
function run() {
  const start = Date.now()
  const audio = decode_wav(bytes)
  const features = processor.process(audio.samples, audio.sampling_rate)
  const preprocessing_ms = Date.now() - start
  const result = model.transcribe(features)
  return { ...result, preprocessing_ms, elapsed_ms: Date.now() - start }
}
run() // Warm model, allocator and kernels before the measured requests.
const runs = Array.from({ length: 3 }, run)
fs.writeFileSync(
  getEnv('PROFILE_OUTPUT') ?? '/private/tmp/whisper-benchmark.json',
  JSON.stringify({ path, device, warmup: 1, runs }, null, 2),
)
console.log(
  JSON.stringify(
    runs.map(({ preprocessing_ms, encoder_ms, decoder_ms, elapsed_ms }) => ({
      preprocessing_ms,
      encoder_ms,
      decoder_ms,
      elapsed_ms,
    })),
  ),
)
