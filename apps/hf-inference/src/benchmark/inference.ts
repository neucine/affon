import fs from 'std:fs'
import telemetry from 'std:telemetry'
import { getEnv } from 'std:process'
import {
  snapshot_download,
  load_model,
  load_processor,
} from '../../../../packages/@affon/huggingface/src/index.ts'
import { TEXT_MODEL, IMAGE_MODEL } from '../inference/models.ts'
import { rank_classes } from '../inference/classification.ts'

const device = getEnv('AFFON_DEVICE') ?? 'cpu'
if (device !== 'cpu' && device !== 'metal') throw Error('Use cpu or metal')
const family = getEnv('AFFON_BENCH_FAMILY') ?? 'vit'
if (family !== 'vit' && family !== 'gpt2') throw Error('Use vit or gpt2')
const count = Number(getEnv('AFFON_BENCH_ITERATIONS') ?? 5)
if (!Number.isInteger(count) || count < 3 || count > 100)
  throw Error('Use 3–100 measured iterations')
const use_cache = getEnv('AFFON_BENCH_KV_CACHE') === '1'
const path = getEnv('AFFON_BENCH_REPORT')
if (!path) throw Error('Set AFFON_BENCH_REPORT')
const spec = family === 'vit' ? IMAGE_MODEL : TEXT_MODEL
const memory_samples: { phase: string; values: Record<string, number> }[] = []
function memory(phase: string) {
  memory_samples.push({
    phase,
    values: Object.fromEntries(
      telemetry
        .metrics()
        .filter(
          (x) =>
            ['runtime.memory', 'compute.storage', 'compute.memory'].includes(
              x.scope,
            ) &&
            (x.name.includes('bytes') || x.name.includes('footprint')),
        )
        .map((x) => [`${x.scope}.${x.name}`, x.value]),
    ),
  })
}
function measure<T>(fn: () => T) {
  const start = Date.now()
  const value = fn()
  return { value, ms: Date.now() - start }
}
function stats(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return {
    samples: values,
    min: sorted[0],
    median:
      sorted.length % 2
        ? sorted[middle]
        : (sorted[middle - 1] + sorted[middle]) / 2,
    mean: values.reduce((a, b) => a + b, 0) / values.length,
    max: sorted[sorted.length - 1],
  }
}
memory('before_load')
const start = Date.now()
const directory = await snapshot_download(spec.id, {
  revision: spec.revision,
  cache_dir: getEnv('AFFON_HF_CACHE') ?? '/tmp/affon-hub-cache',
  local_files_only: true,
})
const cache_validation_ms = Date.now() - start
const observations: Record<string, number[]> = {}
let first_run: Record<string, number> = {}
let output: unknown
let model_load_ms = 0,
  processor_load_ms = 0
let workload: unknown
function record(values: Record<string, number>, index: number) {
  if (index === -1) first_run = values
  else
    for (const [name, value] of Object.entries(values))
      (observations[name] ??= []).push(value)
  memory(index === -1 ? 'after_warmup' : `after_iteration_${index + 1}`)
  console.log(
    `${family}/${device}: ${index === -1 ? 'warmup' : `iteration ${index + 1}`} complete`,
  )
}
if (family === 'vit') {
  const options = { task: 'image-classification' as const, device } as const
  const processor = measure(() => load_processor(directory, options))
  processor_load_ms = processor.ms
  const loaded = measure(() => load_model(directory, options))
  model_load_ms = loaded.ms
  memory('after_load')
  // Fixed input construction is excluded; native resize/normalize is measured.
  const rgb = Array.from({ length: 256 }, (_, y) =>
    Array.from({ length: 320 }, (_, x) =>
      Array.from({ length: 3 }, (_, c) => (x * 3 + y * 5 + c * 47) % 256),
    ),
  )
  workload = {
    input: 'deterministic RGB8 (x*3+y*5+c*47)%256',
    width: 320,
    height: 256,
    batch: 1,
  }
  function iteration(index: number) {
    const prepared = measure(() => {
      const pixels = processor.value.process(rgb)
      pixels.to_array()
      return pixels
    })
    const forward = measure(
      () =>
        (
          loaded.value.forward(prepared.value).output.to_array() as number[][]
        )[0],
    )
    if (forward.value.some((x) => !Number.isFinite(x)))
      throw Error('Nonfinite output')
    const ranking = measure(() =>
      rank_classes(forward.value, loaded.value.config.id2label),
    )
    if (ranking.value[0].id !== 733)
      throw Error('ViT benchmark output changed: expected class 733')
    output = ranking.value[0]
    return {
      preprocess_with_readback_ms: prepared.ms,
      forward_with_readback_ms: forward.ms,
      ranking_ms: ranking.ms,
    }
  }
  for (let i = -1; i < count; i++) record(iteration(i), i)
} else {
  const options = { task: 'text-generation' as const, device } as const
  const processor = measure(() => load_processor(directory, options))
  processor_load_ms = processor.ms
  const loaded = measure(() => load_model(directory, options))
  model_load_ms = loaded.ms
  memory('after_load')
  const prompt = 'The future of computing is'
  const budget = 16
  workload = {
    prompt,
    prompt_tokens: processor.value.encode(prompt).length,
    max_new_tokens: budget,
    batch: 1,
    kv_cache: use_cache,
  }
  function iteration() {
    // Batch the very short tokenizer operation above millisecond clock resolution.
    const encoding = measure(() => {
      let ids: number[] = []
      for (let i = 0; i < 1000; i++) ids = processor.value.encode(prompt)
      return ids
    })
    const generation = measure(() =>
      loaded.value.generate(encoding.value, budget, { use_cache }),
    )
    const decoding = measure(() => processor.value.decode(generation.value))
    const tokens = generation.value.length - encoding.value.length
    if (
      !decoding.value.startsWith('The future of computing is a matter of time')
    )
      throw Error('Text benchmark output changed')
    output = { text: decoding.value, generated_tokens: tokens }
    return {
      tokenize_per_call_ms: encoding.ms / 1000,
      generate_ms: generation.ms,
      decode_ms: decoding.ms,
      generated_tokens: tokens,
      tokens_per_second: (tokens * 1000) / Math.max(1, generation.ms),
    }
  }
  for (let i = -1; i < count; i++) record(iteration(), i)
}
const report = {
  format: 'affon-inference-benchmark/v1',
  model_id: spec.id,
  revision: spec.revision,
  family,
  device,
  dtype: 'f32',
  build_profile: getEnv('AFFON_AUDIT_BUILD') ?? 'unspecified',
  clock: 'Date.now (1 ms resolution)',
  cache_policy:
    'offline; SHA-256 validation measured separately; OS file cache not flushed',
  cache_validation_ms,
  processor_load_ms,
  model_load_ms,
  warmup_runs: 1,
  measured_runs: count,
  workload,
  first_run,
  metrics: Object.fromEntries(
    Object.entries(observations).map(([name, values]) => [name, stats(values)]),
  ),
  output,
  memory_samples,
  caveats: [
    'Readback completes GPU work before timing ends.',
    'ViT preprocessing timing includes extra readback for synchronization.',
    'Short fixed-workload run; memory samples do not prove leak freedom.',
    'Tensor peaks are process-lifetime high-water marks, including loading.',
  ],
}
fs.writeFileSync(path, JSON.stringify(report, null, 2))
console.log(`Saved ${path}`)
