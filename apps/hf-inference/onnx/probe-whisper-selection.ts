import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { getEnv } from 'std:process'
import type { Tensor } from 'affon:compute'
import { load_whisper } from '../../../packages/@affon/huggingface/src/adapters/whisper.ts'
const root = getEnv('AFFON_WHISPER_DIR') ?? '/private/tmp/affon-onnx-whisper'
const policy = JSON.parse(fs.readFileSync(`${root}/whisper.json`))
const suppress = new Set<number>(policy.suppress_tokens)
const begin = new Set<number>(policy.begin_suppress_tokens)
const model = load_whisper(`${root}/source`, {
  task: 'automatic-speech-recognition',
  backend: 'onnx',
  graph_dir: root,
  device: 'metal',
})
const refs = checkpoint.load(`${root}/reference.safetensors`) as Record<
  string,
  Tensor
>
function select(logits: number[], step: number, scoreFirst: boolean) {
  let best = -1,
    score = -Infinity
  for (let i = 0; i < logits.length; i++) {
    const eligible = scoreFirst
      ? logits[i] > score && !suppress.has(i) && !(step === 0 && begin.has(i))
      : !suppress.has(i) && !(step === 0 && begin.has(i)) && logits[i] > score
    if (eligible) {
      best = i
      score = logits[i]
    }
  }
  return best
}
const cases: unknown[] = []
const result = model.transcribe(refs.features.to('metal'), (step, logits) => {
  const expected = select(logits, step, false)
  if (select(logits, step, true) !== expected) throw Error('Selection mismatch')
  const times = [0, 0]
  // Alternate order to reduce a consistent first-run advantage.
  for (let repeat = 0; repeat < 6; repeat++)
    for (const version of repeat % 2 ? [1, 0] : [0, 1]) {
      const start = Date.now()
      if (select(logits, step, version === 1) !== expected)
        throw Error('Selection mismatch')
      times[version] += Date.now() - start
    }
  cases.push({
    step,
    expected,
    repeats: 6,
    set_first_ms: times[0],
    score_first_ms: times[1],
  })
})
// Preserve first-index tie behavior, suppression and non-finite comparisons.
for (const logits of [
  [1, 1, 0],
  [NaN, -Infinity, 2],
  [-Infinity, -Infinity],
  Array(policy.eos + 1).fill(0),
])
  for (const step of [0, 1])
    if (select(logits, step, false) !== select(logits, step, true))
      throw Error('Edge case mismatch')
fs.writeFileSync(
  getEnv('PROFILE_OUTPUT')!,
  JSON.stringify({ cases, result }, null, 2),
)
