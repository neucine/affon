// Write one full-prefix Apertus logit row for comparison with PyTorch.
import fs from 'std:fs'
import { getEnv } from 'std:process'
import { load_apertus } from '../../../packages/@affon/huggingface/src/adapters/apertus.ts'
import { CausalProgramRuntime } from '../../../apps/hf-inference/src/inference/program-runtime.ts'

const checkpoint = getEnv('AFFON_APERTUS_CHECKPOINT')
const output = getEnv('AFFON_APERTUS_LOGITS_PATH')
if (!checkpoint || !output) throw Error('Set AFFON_APERTUS_CHECKPOINT and AFFON_APERTUS_LOGITS_PATH')
const runtime = new CausalProgramRuntime(load_apertus(checkpoint), 'cuda')
const prompt = [1, 1784, 8961, 1307, 32265, 1395]
try {
  const result = runtime.forward(prompt, prompt.length - 1)
  try {
    fs.writeFileSync(output, JSON.stringify(Array.from(new Uint8Array(result.logits.to_bytes()))))
    console.log(JSON.stringify({ suite: 'apertus_affon_logits', prompt,
      shape: result.logits.shape, dtype: result.logits.dtype, output }))
  } finally {
    result.logits.dispose()
    for (const value of result.hidden_states) value.dispose()
  }
} finally { runtime.dispose() }
