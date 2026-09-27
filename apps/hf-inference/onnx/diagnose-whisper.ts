// Diagnostic only: phase observers and metric snapshots perturb wall-clock time.
import fs from 'std:fs'
import telemetry from 'std:telemetry'
import { getEnv } from 'std:process'
import { load_whisper } from '../../../packages/@affon/huggingface/src/whisper.ts'
import { load_whisper_processor } from '../../../packages/@affon/huggingface/src/processors/whisper.ts'
import { decode_wav } from '../../../packages/@affon/huggingface/src/processors/shared/audio.ts'
const root = getEnv('AFFON_WHISPER_DIR') ?? '/private/tmp/affon-onnx-whisper'
const model = load_whisper(`${root}/source`, {
  task: 'automatic-speech-recognition',
  backend: 'onnx',
  graph_dir: root,
  device: 'metal',
})
const processor = load_whisper_processor(`${root}/source`, 'metal')
const bytes = new Uint8Array(
  JSON.parse(fs.readFileSync(getEnv('PROFILE_AUDIO_BYTES')!)),
)
const audio = decode_wav(bytes)
const features = processor.process(audio.samples, audio.sampling_rate)
model.transcribe(features)
function snapshot() {
  return Object.fromEntries(
    telemetry
      .metrics()
      .filter(
        (m) => m.scope === 'compute.execution' || m.scope === 'compute.storage',
      )
      .map((m) => [`${m.scope}/${m.name}`, m.value]),
  )
}
const runs = Array.from({ length: 3 }, () => {
  let before = snapshot()
  const phases: Record<
    string,
    { calls: number; elapsed_ms: number; counters: Record<string, number> }
  > = {}
  const result = model.transcribe(features, undefined, (name, elapsed_ms) => {
    const after = snapshot()
    const phase = (phases[name] ??= { calls: 0, elapsed_ms: 0, counters: {} })
    phase.calls++
    phase.elapsed_ms += elapsed_ms
    for (const [key, value] of Object.entries(after))
      phase.counters[key] =
        (phase.counters[key] ?? 0) + value - (before[key] ?? 0)
    before = after
  })
  return { result, phases }
})
fs.writeFileSync(
  getEnv('PROFILE_OUTPUT')!,
  JSON.stringify({ warmup: 1, runs }, null, 2),
)
console.log(
  JSON.stringify(
    runs.map((r) => ({
      inference_ms: r.result.inference_ms,
      phases: Object.fromEntries(
        Object.entries(r.phases).map(([k, v]) => [k, v.elapsed_ms]),
      ),
    })),
  ),
)
