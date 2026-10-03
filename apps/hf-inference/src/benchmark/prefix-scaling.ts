// Full-prefix Program execution measures honest decode scaling without claiming a KV cache.
import fs from 'std:fs'
import telemetry from 'std:telemetry'
import { getEnv } from 'std:process'
import { load_model, load_processor, snapshot_download } from '../../../../packages/@affon/huggingface/src/index.ts'
import { TEXT_MODEL } from '../inference/models.ts'
import { CausalProgramRuntime } from '../inference/program-runtime.ts'
import type { Device } from 'affon:compute'

const device = getEnv('AFFON_DEVICE') ?? 'cpu'
if (device !== 'cpu' && device !== 'metal') throw Error('Use cpu or metal')
const path = getEnv('AFFON_BENCH_REPORT')
if (!path) throw Error('Set AFFON_BENCH_REPORT')
const directory = await snapshot_download(TEXT_MODEL.id, {
  revision: TEXT_MODEL.revision, cache_dir: getEnv('AFFON_HF_CACHE') ?? '/tmp/affon-hub-cache', local_files_only: true,
})
const model = load_model(directory, { task: 'text-generation' })
const processor = load_processor(directory, { task: 'text-generation', device })
const seed = processor.encode('The history of science is full of discoveries that changed how people understand the world. ')
function memory() {
  return Object.fromEntries(telemetry.metrics().filter(x =>
    ['compute.storage.live_bytes', 'runtime.memory.resident_bytes', 'runtime.memory.physical_footprint_bytes']
      .includes(`${x.scope}.${x.name}`)).map(x => [`${x.scope}.${x.name}`, x.value]))
}
function iteration(prompt: number[], budget: number) {
  const runtime = new CausalProgramRuntime(model, device as Device)
  const before = memory()
  function step(ids: number[]) {
    const result = runtime.forward(ids, ids.length - 1)
    const row = (result.logits.to_array() as number[][][])[0][0]
    if (row.some(x => !Number.isFinite(x))) throw Error('Nonfinite logits')
    let best = 0
    for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i
    result.logits.dispose()
    for (const hidden of result.hidden_states) hidden.dispose()
    return best
  }
  const started = Date.now()
  const generated = [step(prompt)]
  const prefill_ms = Date.now() - started
  const after_prefill = memory()
  const decodeStart = Date.now()
  const decode_step_ms: number[] = []
  for (let i = 1; i < budget; i++) {
    const start = Date.now()
    generated.push(step([...prompt, ...generated]))
    decode_step_ms.push(Date.now() - start)
  }
  const decode_ms = Date.now() - decodeStart
  const after_decode = memory()
  runtime.dispose()
  const after_dispose = memory()
  return { prefill_ms, decode_ms, total_ms: prefill_ms + decode_ms, decode_step_ms,
    generated, memory: { before, after_prefill, after_decode, after_dispose } }
}
const results = []
for (const [prompt_tokens, new_tokens] of [[16,16], [128,16], [512,16], [128,64]]) {
  const prompt = Array.from({ length: prompt_tokens }, (_, i) => seed[i % seed.length])
  const warmup = iteration(prompt, new_tokens)
  const samples = []
  for (let i = 0; i < 3; i++) {
    const sample = iteration(prompt, new_tokens)
    if (JSON.stringify(sample.generated) !== JSON.stringify(warmup.generated)) throw Error('Unstable greedy output')
    samples.push(sample)
    console.log(`${device}: ${prompt_tokens} prompt / ${new_tokens} new: ${i + 1}/3, ${sample.total_ms} ms`)
  }
  results.push({ prompt_tokens, new_tokens, prompt_ids: prompt, warmup, samples })
  fs.writeFileSync(path, JSON.stringify({ format: 'affon-prefix-scaling/v1', device, model: TEXT_MODEL,
    clock: 'Date.now (1ms)', warmup_runs: 1, measured_runs: 3,
    policy: 'Full-prefix Program execution with one-token output windows; no KV cache is claimed. Fixed-length greedy decoding, EOS ignored; one model per process, scenarios sequential; offline f32 batch one.', results }, null, 2))
}
