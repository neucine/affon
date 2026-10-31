// Write the first cached decode logit row for comparison with PyTorch.
import fs from 'std:fs'
import { getEnv } from 'std:process'
import { type Tensor } from 'affon:compute'
import { load_apertus } from '../../../packages/@affon/huggingface/src/adapters/apertus.ts'
import { CausalProgramRuntime } from '../../../apps/hf-inference/src/inference/program-runtime.ts'

const checkpoint = getEnv('AFFON_APERTUS_CHECKPOINT')
const output = getEnv('AFFON_APERTUS_LOGITS_PATH')
if (!checkpoint || !output) throw Error('Set AFFON_APERTUS_CHECKPOINT and AFFON_APERTUS_LOGITS_PATH')
const model = load_apertus(checkpoint)
const runtime = new CausalProgramRuntime(model, 'cuda')
const session = runtime.session
const prompt = [1, 1784, 8961, 1307, 32265, 1395]
const capacity = 320
const firstToken = 10518
const ids = session.tensor(prompt, { dtype: 'i64' })
const mask = session.tensor([[Array.from({ length: prompt.length }, (_, row) =>
  Array.from({ length: prompt.length }, (_, column) => column > row ? 1 : 0))]], { dtype: 'i64' })
let caches: Tensor[] = []
try {
  const prefill = session.compile(model.prefill(prompt.length, capacity))
  try {
    const [logits, ...states] = prefill.run({ ids, mask }, runtime.state) as Tensor[]
    caches = states
    try {
      const values = (logits.to_array() as number[][][])[0][0]
      let chosen = 0
      for (let index = 1; index < values.length; index++)
        if (values[index] > values[chosen]) chosen = index
      if (chosen !== firstToken) throw Error(`Unexpected prefill token ${chosen}`)
    } finally { logits.dispose() }
  } finally { prefill.dispose() }
  const token = session.tensor([firstToken], { dtype: 'i64' })
  const position = session.tensor([prompt.length], { dtype: 'i64' })
  try {
    const inputs: Record<string, Tensor> = { ids: token, position }
    for (let index = 0; index < caches.length / 2; index++) {
      inputs[`past_${index}_k`] = caches[index * 2]
      inputs[`past_${index}_v`] = caches[index * 2 + 1]
    }
    const decode = session.compile(model.decode(capacity))
    try {
      const [logits, ...nextCaches] = decode.run(inputs, runtime.state) as Tensor[]
      try {
        fs.writeFileSync(output, JSON.stringify(Array.from(new Uint8Array(logits.to_bytes()))))
        console.log(JSON.stringify({ suite: 'apertus_affon_cached_logits', prompt,
          firstToken, capacity, shape: logits.shape, dtype: logits.dtype, output }))
      } finally {
        logits.dispose()
        for (const value of nextCaches) value.dispose()
      }
    } finally { decode.dispose() }
  } finally { token.dispose(); position.dispose() }
} finally {
  for (const value of caches) value.dispose()
  ids.dispose(); mask.dispose(); runtime.dispose()
}
