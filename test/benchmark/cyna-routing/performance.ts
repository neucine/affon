// Run with a ReleaseFast Affon build. Model construction, compilation, state
// initialization, and input construction happen before warmup and timing.
import fs from 'std:fs'
import { getEnv } from 'std:process'
import { Session, Tensor } from 'affon:core'
import { create_llama } from '../../../packages/@affon/models/src/llama/model.ts'

const backendSymbol = Symbol.for('affon.compute.executable_backend')
const routingSymbol = Symbol.for('affon.compute.executable_routing')

const reportPath = required('AFFON_CYNA_BENCH_REPORT')
const expectedBackend = required('AFFON_CYNA_BENCH_EXPECTED_BACKEND')
const expectedReason = required('AFFON_CYNA_BENCH_EXPECTED_REASON')
const buildProfile = getEnv('AFFON_CYNA_BENCH_BUILD_PROFILE') ?? 'unspecified'
const warmupCount = integerEnv('AFFON_CYNA_BENCH_WARMUPS', 2, 0, 100)
const sampleCount = integerEnv('AFFON_CYNA_BENCH_SAMPLES', 7, 3, 100)
const executionsPerSample = integerEnv('AFFON_CYNA_BENCH_EXECUTIONS_PER_SAMPLE', 1, 1, 100)
const comparison = { absolute: 3e-4, relative: 3e-4 }

const config = {
  width: 128,
  innerWidth: 344,
  heads: 4,
  kvHeads: 2,
  layers: 2,
  contextLength: 256,
  vocabSize: 1024,
  epsilon: 1e-5,
  ropeTheta: 10_000,
} as const
const prefillLength = 64
const decodePosition = 128

function required(name: string): string {
  const value = getEnv(name)
  if (!value) throw Error(`Set ${name}`)
  return value
}

function integerEnv(name: string, fallback: number, minimum: number, maximum: number): number {
  const text = getEnv(name)
  const value = text === null ? fallback : Number(text)
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw Error(`${name} must be an integer from ${minimum} to ${maximum}`)
  }
  return value
}

function seededValues(shape: readonly number[], seed: number, center: number, scale: number): any {
  let state = (seed >>> 0) || 1
  const next = (): number => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    return center + (state / 0x1_0000_0000 * 2 - 1) * scale
  }
  const build = (dimension: number): any => dimension === shape.length
    ? next()
    : Array.from({ length: shape[dimension] }, () => build(dimension + 1))
  return build(0)
}

function hashName(name: string): number {
  let hash = 2166136261
  for (let index = 0; index < name.length; index++) {
    hash ^= name.charCodeAt(index)
    hash = Math.imul(hash, 16777619) >>> 0
  }
  return hash
}

const session = new Session({ device: 'cpu' })
const value = (name: string, shape: readonly number[], scale: number, center = 0): Tensor =>
  session.tensor(seededValues(shape, hashName(name), center, scale), { dtype: 'f32' })

const headWidth = config.width / config.heads
const kvWidth = config.kvHeads * headWidth
const blocks = Array.from({ length: config.layers }, (_, layer) => ({
  attentionNorm: value(`block_${layer}_attentionNorm`, [config.width], 0.02, 1),
  feedForwardNorm: value(`block_${layer}_feedForwardNorm`, [config.width], 0.02, 1),
  query: value(`block_${layer}_query`, [config.width, config.width], 0.04),
  key: value(`block_${layer}_key`, [config.width, kvWidth], 0.04),
  value: value(`block_${layer}_value`, [config.width, kvWidth], 0.04),
  attentionOutput: value(`block_${layer}_attentionOutput`, [config.width, config.width], 0.04),
  gate: value(`block_${layer}_gate`, [config.width, config.innerWidth], 0.04),
  up: value(`block_${layer}_up`, [config.width, config.innerWidth], 0.04),
  down: value(`block_${layer}_down`, [config.innerWidth, config.width], 0.04),
}))
const model = create_llama(config, {
  tokenEmbedding: value('token_embedding', [config.vocabSize, config.width], 0.06),
  finalNorm: value('final_norm', [config.width], 0.02, 1),
  blocks,
})

const prefillSource = model.prefill(prefillLength)
const decodeSource = model.decode(decodePosition)
const prefillExecutable = session.compile(prefillSource)
const decodeExecutable = session.compile(decodeSource)
const prefillState = session.initialize(prefillSource, { parameters: model.parameters })
const decodeState = session.initialize(decodeSource, { parameters: model.parameters })

const prefillIds = session.tensor(
  Array.from({ length: prefillLength }, (_, index) => (index * 17 + 3) % config.vocabSize),
  { dtype: 'i64' },
)
const causalMask = Array.from({ length: prefillLength }, (_, row) =>
  Array.from({ length: prefillLength }, (_, column) => row >= column ? 0 : 1),
)
const prefillMask = session.tensor([[causalMask]], { dtype: 'i64' })
const decodeIds = session.tensor([37], { dtype: 'i64' })
const decodeInputs: Record<string, Tensor> = { ids: decodeIds }
for (let layer = 0; layer < config.layers; layer++) {
  const shape = [1, config.kvHeads, decodePosition, headWidth]
  decodeInputs[`past_${layer}_k`] = session.tensor(seededValues(shape, hashName(`past_${layer}_k`), 0, 0.03), { dtype: 'f32' })
  decodeInputs[`past_${layer}_v`] = session.tensor(seededValues(shape, hashName(`past_${layer}_v`), 0, 0.03), { dtype: 'f32' })
}

function checkRoute(name: string, executable: any): Record<string, unknown> {
  const backend = executable[backendSymbol]()
  const routing = executable[routingSymbol]()
  if (backend !== expectedBackend) throw Error(`${name} backend=${backend}, expected ${expectedBackend}`)
  if (routing.reason !== expectedReason) throw Error(`${name} routing reason=${routing.reason}, expected ${expectedReason}`)
  return {
    backend,
    reason: routing.reason,
    category: routing.category,
    configured_backend: routing.configured_backend,
  }
}

const routes = {
  prefill: checkRoute('prefill', prefillExecutable),
  decode: checkRoute('decode', decodeExecutable),
}

function runPrefill(): Tensor[] {
  return prefillExecutable.run({ ids: prefillIds, mask: prefillMask }, prefillState) as Tensor[]
}

function runDecode(): Tensor[] {
  return decodeExecutable.run(decodeInputs, decodeState) as Tensor[]
}

function disposeOutputs(outputs: Tensor[]): void {
  for (let index = outputs.length - 1; index >= 0; index--) outputs[index].dispose()
}

function captureLogits(run: () => Tensor[]): number[] {
  const outputs = run()
  try {
    const logits = outputs[0]
    if (!logits || logits.dtype !== 'f32' || logits.shape.length !== 3 || logits.shape[0] !== 1 || logits.shape[1] !== 1 || logits.shape[2] !== config.vocabSize) {
      throw Error(`Unexpected logits contract: dtype=${logits?.dtype}, shape=${JSON.stringify(logits?.shape)}`)
    }
    const values = (logits.to_array() as any[]).flat(Infinity) as number[]
    if (values.length !== config.vocabSize || values.some(x => !Number.isFinite(x))) throw Error('Llama logits are incomplete or nonfinite')
    return values
  } finally {
    disposeOutputs(outputs)
  }
}

// This run establishes the reference values but contributes no timing sample.
const reference = {
  prefill_logits: captureLogits(runPrefill),
  decode_logits: captureLogits(runDecode),
}

function preciseMedian(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

function statistics(samples: number[]) {
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length
  const variance = samples.length > 1
    ? samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (samples.length - 1)
    : 0
  return {
    samples_ms: samples,
    min_ms: Math.min(...samples),
    max_ms: Math.max(...samples),
    mean_ms: mean,
    median_ms: preciseMedian(samples),
    variance_ms2: variance,
    standard_deviation_ms: Math.sqrt(variance),
  }
}

function measure(run: () => Tensor[], tokens: number) {
  for (let warmup = 0; warmup < warmupCount; warmup++) disposeOutputs(run())
  const samples: number[] = []
  for (let sample = 0; sample < sampleCount; sample++) {
    const start = performance.now()
    for (let execution = 0; execution < executionsPerSample; execution++) disposeOutputs(run())
    samples.push((performance.now() - start) / executionsPerSample)
  }
  const values = statistics(samples)
  return { ...values, tokens, tokens_per_second: tokens * 1000 / values.median_ms }
}

const prefill = measure(runPrefill, prefillLength)
const decode = measure(runDecode, 1)
const report = {
  schema_version: 1,
  format: 'affon-cyna-llama-performance/v1',
  build_profile: buildProfile,
  expectation: { backend: expectedBackend, reason: expectedReason },
  routes,
  workload: {
    family: 'llama',
    variant: 'deterministic-synthetic-weights',
    dtype: 'f32',
    parameter_generator: 'LCG-seeded by parameter name',
    dimensions: config,
    head_width: headWidth,
    kv_width: kvWidth,
    prefill_length: prefillLength,
    decode_position: decodePosition,
    prefill_tokens: prefillLength,
    decode_tokens: 1,
  },
  measurement: {
    warmups: warmupCount,
    samples: sampleCount,
    executions_per_sample: executionsPerSample,
    prefill,
    decode,
    timing_scope: [
      'model construction excluded',
      'Program creation excluded',
      'compilation excluded',
      'state initialization excluded',
      'parameter and input allocation excluded',
      'fixed inputs and parameters reused',
      'warmups excluded',
      'run plus disposal of every returned output is timed',
    ],
    completion_boundary: 'The invocation completes after every returned Tensor is disposed.',
  },
  reference,
  limitations: [
    'Synthetic weights measure execution speed, not trained-model quality.',
    'This shape and dtype do not establish broad shape, dtype, or model coverage.',
    'Host wall-clock timing includes process-local allocator and runtime overhead.',
    'Measurements in separate processes are sequential and not a load test.',
  ],
}

try {
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n')
  console.log(JSON.stringify({
    format: report.format,
    backend: expectedBackend,
    prefill_median_ms: prefill.median_ms,
    prefill_tokens_per_second: prefill.tokens_per_second,
    decode_median_ms: decode.median_ms,
    report: reportPath,
  }))
} finally {
  for (const input of Object.values(decodeInputs)) input.dispose()
  prefillMask.dispose()
  prefillIds.dispose()
  prefillState.dispose()
  decodeState.dispose()
  prefillExecutable.dispose()
  decodeExecutable.dispose()
  for (const parameter of Object.values(model.parameters)) parameter.dispose()
  session.dispose()
}
