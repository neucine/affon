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
const captureSelectedPlans = getEnv('AFFON_CYNA_BENCH_SELECTED_PLANS') === '1'
const optimizationGoal = (() => {
  const goal = getEnv('AFFON_CYNA_BENCH_OPTIMIZATION_GOAL') ?? 'balanced'
  if (goal !== 'balanced' && goal !== 'latency' && goal !== 'throughput' && goal !== 'memory')
    throw Error(`Unsupported optimization goal: ${goal}`)
  return goal
})()
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

const session = new Session({ device: 'cpu', ...(captureSelectedPlans ? { telemetry: { captureRecords: 512 } } : {}) })
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
const prefillExecutable = session.compile(prefillSource, { optimizationGoal })
const decodeExecutable = session.compile(decodeSource, { optimizationGoal })
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
type ObservedPlan = {
  positions: Map<number, number>
  kernelCounts: Map<number, number>
  kernelCountTotal: number
}

function observedPlan(executable: any, run: () => Tensor[]): ObservedPlan {
  session.resetEvidence()
  disposeOutputs(run())
  const evidence = session.runtimeEvidence(executable) as any
  if (evidence.execution_path !== 'prepared' || evidence.dropped_event_count !== 0 || evidence.observed_run_count !== 1)
    throw Error('Selected plan needs one complete prepared execution trace')
  if (evidence.kernel_count_evidence !== 'exact')
    throw Error('Selected plan needs exact kernel count evidence')
  const positions = new Map<number, number>()
  const kernelCounts = new Map<number, number>()
  let kernelCountTotal = 0
  for (const entry of evidence.entries) if (entry.graph_kind === 'prepared_physical') {
    if (entry.invocation_count !== 1 || entry.observed_schedule_position == null)
      throw Error(`Missing schedule position for Cyna node ${entry.step_index}`)
    if (!Number.isSafeInteger(entry.kernel_count) || entry.kernel_count < 0)
      throw Error(`Missing exact kernel count for Cyna node ${entry.step_index}`)
    positions.set(entry.step_index, entry.observed_schedule_position)
    kernelCounts.set(entry.step_index, entry.kernel_count)
    kernelCountTotal += entry.kernel_count
  }
  if (new Set(positions.values()).size !== positions.size)
    throw Error('Cyna execution has repeated schedule positions')
  return { positions, kernelCounts, kernelCountTotal }
}
const observedPlans = captureSelectedPlans && expectedBackend === 'cyna' ? {
  prefill: observedPlan(prefillExecutable, runPrefill),
  decode: observedPlan(decodeExecutable, runDecode),
} : null
function selectedPlan(executable: any, observed?: ObservedPlan): unknown[] {
  const explanation = executable.explain() as any
  if (expectedBackend === 'cyna') {
    const plan = explanation.cyna_selected_physical_nodes.map((node: any) => {
      if (!Array.isArray(node.origins) || node.origins.length !== node.source_instruction_count)
        throw Error(`Missing source origins for Cyna node ${node.node_id}`)
      if (!Array.isArray(node.program_origins))
        throw Error(`Missing Program origins for Cyna node ${node.node_id}`)
      const schedulePosition = observed?.positions.get(node.node_id)
      const kernelCount = observed?.kernelCounts.get(node.node_id)
      if (observed && schedulePosition == null) throw Error(`Missing schedule position for Cyna node ${node.node_id}`)
      if (observed && kernelCount == null) throw Error(`Missing kernel count for Cyna node ${node.node_id}`)
      return {
        node: node.node_id,
        schedule_position: schedulePosition,
        kernel_count: kernelCount,
        action: node.action,
        operation: node.operation,
        source_instructions: node.program_origins,
        source_instruction_count: node.program_origins.length,
        cyna_step_origins: node.origins,
        implementation: node.implementation,
        is_view: node.is_view,
        output_bytes: node.output_bytes,
      }
    })
    if (observed && observed.kernelCounts.size !== plan.length)
      throw Error(`Cyna observed ${observed.kernelCounts.size} physical nodes for a ${plan.length}-node plan`)
    return plan
  }
  const nodes = explanation.executable.physical_nodes
  return explanation.executable.physical_order.map((index: number) => {
    const node = nodes[index]
    return {
      node: node.id,
      action: node.action,
      source_instructions: node.origins,
      implementation: node.implementation_id,
      view_transform: node.view_transform,
    }
  })
}
function selectedPlanMetrics(executable: any, observed?: ObservedPlan): unknown {
  const explanation = executable.explain() as any
  return expectedBackend === 'cyna'
    ? {
        selected_peak_active_bytes: explanation.selected_physical_liveness.peak_active_bytes,
        kernel_count_evidence: observed ? 'exact' : null,
        observed_kernel_count: observed?.kernelCountTotal ?? null,
        ...(explanation.cyna_selection_comparison ?? {}),
      }
    : { selected_peak_active_bytes: explanation.executable.selection?.selected_peak_active_bytes ?? null }
}
const report = {
  schema_version: 1,
  format: 'affon-cyna-llama-performance/v1',
  build_profile: buildProfile,
  expectation: { backend: expectedBackend, reason: expectedReason },
  routes,
  workload: {
    family: 'llama',
    optimization_goal: optimizationGoal,
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
  ...(captureSelectedPlans ? { selected_physical_plans: {
    prefill: selectedPlan(prefillExecutable, observedPlans?.prefill),
    decode: selectedPlan(decodeExecutable, observedPlans?.decode),
  }, selected_physical_values: {
    prefill: (prefillExecutable.explain() as any).selected_physical_values,
    decode: (decodeExecutable.explain() as any).selected_physical_values,
  }, selected_physical_liveness: {
    prefill: (prefillExecutable.explain() as any).selected_physical_liveness,
    decode: (decodeExecutable.explain() as any).selected_physical_liveness,
  }, selected_plan_metrics: {
    prefill: selectedPlanMetrics(prefillExecutable, observedPlans?.prefill),
    decode: selectedPlanMetrics(decodeExecutable, observedPlans?.decode),
  } } : {}),
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
