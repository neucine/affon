// Keep the telemetry console available after a bounded reference transcription.
import telemetry from 'std:telemetry'
import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { getEnv } from 'std:process'
import { Session, type Tensor, type Device } from 'affon:compute'
import { load_whisper } from '../../../packages/@affon/huggingface/src/adapters/whisper.ts'
import { WhisperProgramRuntime } from '../src/inference/program-runtime.ts'
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
})
const runtime = new WhisperProgramRuntime(model, device)
const session = new Session({ device })
const features = session.tensor(refs.features.to_array() as any, { dtype: refs.features.dtype })
const before = telemetry.metrics()
const result = telemetry.trace('whisper.reference', () =>
  runtime.transcribe(features),
)
fs.writeFileSync(
  getEnv('PROFILE_OUTPUT') ?? '/private/tmp/whisper-telemetry.json',
  JSON.stringify({ result, before, after: telemetry.metrics() }, null, 2),
)
console.log(
  'Profile complete; telemetry console remains available for 60 seconds',
)
setTimeout(() => {}, 60000)
features.dispose(); session.dispose(); runtime.dispose()
