// Trace one registered DistilGPT-2 Metal forward call separately from timing.
import { getEnv } from 'std:process'
import { load_gpt2, load_processor } from '../../../packages/@affon/huggingface/src/index.ts'
import { CausalProgramRuntime } from '../../../apps/hf-inference/src/inference/program-runtime.ts'

const directory = getEnv('AFFON_HF_SNAPSHOT')
if (!directory) throw Error('Set AFFON_HF_SNAPSHOT to the verified DistilGPT-2 revision')
const processor = load_processor(directory, { task: 'text-generation', device: 'metal' })
const model = load_gpt2(directory)
const ids = processor.encode('The future of computing is')
const runtime = new CausalProgramRuntime(model, 'metal')
try {
  const result = runtime.forward(ids, ids.length - 1)
  try {
    const logits = result.logits.to_array() as number[][][]
    const row = logits[0]?.at(-1)
    if (!row || row.length !== model.config.vocab_size || row.some(value => !Number.isFinite(value)))
      throw Error('Invalid traced DistilGPT-2 logits')
    console.log(JSON.stringify({ prompt_tokens: ids.length, logits: row.length, finite: true }))
  } finally {
    result.logits.dispose()
    for (const hidden of result.hidden_states) hidden.dispose()
  }
} finally {
  runtime.dispose()
}
