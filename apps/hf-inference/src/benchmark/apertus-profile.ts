// Standalone fixed-cache probe for Nsight Systems on the CUDA host.
import { getEnv } from 'std:process'
import { load_apertus } from '../../../../packages/@affon/huggingface/src/adapters/apertus.ts'
import { CausalProgramRuntime } from '../inference/program-runtime.ts'

const checkpoint = getEnv('AFFON_APERTUS_CHECKPOINT')
if (!checkpoint) throw Error('Set AFFON_APERTUS_CHECKPOINT to the verified snapshot directory')
const model = load_apertus(checkpoint)
const runtime = new CausalProgramRuntime(model, 'cuda')
const prompt = [1, 1784, 8961, 1307, 32265, 1395]
const budget = 16
try {
  runtime.prepareFixedCache(320, [prompt.length])
  const generation = runtime.createGeneration(prompt, budget)
  const steps: number[] = []
  let output = prompt
  try {
    while (!generation.done) {
      const start = Date.now()
      const result = generation.next()
      steps.push(Date.now() - start)
      output = result.ids
    }
  } finally { generation.close() }
  console.log('APERTUS_PROFILE', JSON.stringify({capacity: 320, prompt, output, stepMs: steps}))
} finally { runtime.dispose() }
