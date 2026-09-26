// Keep the telemetry console available after a bounded reference transcription.
import telemetry from 'std:telemetry'
import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { getEnv } from 'std:process'
import type { Tensor, Device } from 'affon:compute'
import { load_whisper } from '../../../packages/@affon/huggingface/src/whisper.ts'
const root = getEnv('AFFON_WHISPER_DIR') ?? '/private/tmp/affon-onnx-whisper'
const device = (getEnv('AFFON_DEVICE') ?? 'metal') as Device
const refs = checkpoint.load(`${root}/reference.safetensors`) as Record<
  string,
  Tensor
>
const model = load_whisper(`${root}/source`, {
  task: 'automatic-speech-recognition',
  backend: 'onnx',
  graph_dir: root,
  device,
})
const before = telemetry.metrics()
const result = telemetry.trace('whisper.reference', () =>
  model.transcribe(refs.features.to(device)),
)
fs.writeFileSync(
  getEnv('PROFILE_OUTPUT') ?? '/private/tmp/whisper-telemetry.json',
  JSON.stringify({ result, before, after: telemetry.metrics() }, null, 2),
)
console.log(
  'Profile complete; telemetry console remains available for 60 seconds',
)
setTimeout(() => {}, 60000)
