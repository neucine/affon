// Fixed-length greedy decoding isolates prefill, decode and retained KV storage.
import fs from 'std:fs'
import telemetry from 'std:telemetry'
import { getEnv } from 'std:process'
import { load_model, load_processor, snapshot_download } from '../../../../packages/@affon/huggingface/src/index.ts'
import { TEXT_MODEL } from '../inference/models.ts'

const device = getEnv('AFFON_DEVICE') ?? 'cpu'
if (device !== 'cpu' && device !== 'metal') throw Error('Use cpu or metal')
const path = getEnv('AFFON_BENCH_REPORT')
if (!path) throw Error('Set AFFON_BENCH_REPORT')
const directory = await snapshot_download(TEXT_MODEL.id, {
  revision: TEXT_MODEL.revision, cache_dir: getEnv('AFFON_HF_CACHE') ?? '/tmp/affon-hub-cache', local_files_only: true,
})
const options = { task: 'text-generation', device } as const
const model = load_model(directory, options)
const processor = load_processor(directory, options)
const seed = processor.encode('The history of science is full of discoveries that changed how people understand the world. ')
const bytesPerToken = 2 * model.config.n_layer * model.config.n_embd * 4
function memory() {
  return Object.fromEntries(telemetry.metrics().filter(x =>
    ['compute.storage.live_bytes', 'runtime.memory.resident_bytes', 'runtime.memory.physical_footprint_bytes']
      .includes(`${x.scope}.${x.name}`)).map(x => [`${x.scope}.${x.name}`, x.value]))
}
function iteration(prompt: number[], budget: number) {
  const session = model.create_session()
  const before = memory()
  // Readback synchronizes GPU work. Only scalar token IDs escape this function,
  // so memory samples retain the session cache but no logits/hidden-state outputs.
  function step(ids: number[]) {
    const result = session.forward(ids)
    const row = (result.logits.slice([0, ids.length - 1, ':']).to_array() as number[]).flat(Infinity) as number[]
    if (row.some(x => !Number.isFinite(x))) throw Error('Nonfinite logits')
    let best = 0
    for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i
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
    generated.push(step([generated[generated.length - 1]]))
    decode_step_ms.push(Date.now() - start)
  }
  const decode_ms = Date.now() - decodeStart
  const after_decode = memory()
  const cache_tokens = session.length
  session.reset()
  const after_reset = memory()
  const retained_cache_bytes = after_decode['compute.storage.live_bytes'] - before['compute.storage.live_bytes']
  if (retained_cache_bytes !== cache_tokens * bytesPerToken) throw Error('Unexpected retained KV storage')
  if (after_reset['compute.storage.live_bytes'] !== before['compute.storage.live_bytes']) throw Error('Cache storage retained after reset')
  return { prefill_ms, decode_ms, total_ms: prefill_ms + decode_ms, decode_step_ms,
    generated, cache_tokens, retained_cache_bytes, memory: { before, after_prefill, after_decode, after_reset } }
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
  fs.writeFileSync(path, JSON.stringify({ format: 'affon-kv-scaling/v1', device, model: TEXT_MODEL,
    bytes_per_cache_token: bytesPerToken, clock: 'Date.now (1ms)', warmup_runs: 1, measured_runs: 3,
    policy: 'Fixed-length greedy decoding, EOS ignored; one model per process, scenarios sequential; offline f32 batch one. Prefill includes first token selection. Decode excludes first token. Memory sampling between phases excluded from reported time. Repeated seed token IDs produce exact prompt lengths; this is a systems benchmark, not a quality evaluation.', results }, null, 2))
}
