import fs from 'std:fs'
import { getEnv } from 'std:process'
import { matmul, parameter, seed, sum, transpose } from 'affon:compute/legacy'

type Shape = { name: string; m: number; k: number; n: number }

const shapes: Shape[] = [
  { name: 'attention-projection', m: 1024, k: 256, n: 256 },
  { name: 'ffn-expansion', m: 1024, k: 256, n: 1024 },
  { name: 'ffn-contraction', m: 1024, k: 1024, n: 256 },
  { name: 'vocabulary-projection', m: 1024, k: 256, n: 50257 },
  { name: 'per-head-attention', m: 256, k: 32, n: 256 },
]

const warmups = 3
const samples = 10

type BaselineCase = {
  case_id: string
  suite: 'compute_backward'
  requested_path: string
  actual_path: string
  backend: 'metal'
  dtype: 'f32'
  inputs: { shape: number[]; layout: { strides: number[]; offset: number; contiguous: boolean } }[]
  output: { shape: number[]; layout: { strides: number[]; offset: number; contiguous: boolean } }
  sample_count: number
  warmup_count: number
  iterations_per_sample: number
  durations_ns: number[]
  median_ns: number
  p95_ns: number
  counters: Record<string, number | null>
  representative: { values: number[]; digest: string }
  gradients: { finite: boolean; norm: number | null; representative_values: number[]; digest: string | null }
  loss: number | null
  optimizer_step: null
}

function now(): number {
  const runtime = globalThis as typeof globalThis & { performance?: { now(): number } }
  return runtime.performance?.now() ?? Date.now()
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

function percentile(values: number[], fraction: number): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)]
}

function flatten(value: unknown, output: number[] = []): number[] {
  if (Array.isArray(value)) {
    for (const item of value) flatten(item, output)
  } else if (typeof value === 'number') {
    output.push(value)
  }
  return output
}

function signature(values: number[]): string {
  return values.map((value) => value.toPrecision(9)).join(',')
}

function emptyCounters(): Record<string, number | null> {
  return {
    peak_host_bytes: null,
    peak_device_bytes: null,
    allocation_count: null,
    dispatch_count: null,
    materialization_count: null,
    synchronization_count: null,
  }
}

function backwardIteration(): { loss: number; gradients: number[] } {
  const base = parameter([3, 2], { dtype: 'f32' }).randn().to('metal')
  const rhs = parameter([3, 4], { dtype: 'f32' }).randn().to('metal')
  const loss = sum(matmul(transpose(base, 0, 1), rhs))
  loss.backward()
  const gradients = [
    ...flatten(base.grad?.to('cpu').to_array()),
    ...flatten(rhs.grad?.to('cpu').to_array()),
  ]
  return { loss: loss.to('cpu').item(), gradients }
}

function benchBackward(): BaselineCase {
  seed(20260930)
  for (let i = 0; i < warmups; i += 1) backwardIteration()
  const durations: number[] = []
  let representative = { loss: 0, gradients: [] as number[] }
  for (let i = 0; i < samples; i += 1) {
    const start = now()
    representative = backwardIteration()
    durations.push(Math.round((now() - start) * 1_000_000))
  }
  const gradientNorm = Math.sqrt(representative.gradients.reduce((total, value) => total + value * value, 0))
  return {
    case_id: 'transposed_matmul_backward',
    suite: 'compute_backward',
    requested_path: 'affon_eager_autograd',
    actual_path: 'affon_eager_autograd',
    backend: 'metal',
    dtype: 'f32',
    inputs: [
      { shape: [3, 2], layout: { strides: [2, 1], offset: 0, contiguous: true } },
      { shape: [3, 4], layout: { strides: [4, 1], offset: 0, contiguous: true } },
    ],
    output: { shape: [1], layout: { strides: [1], offset: 0, contiguous: true } },
    sample_count: samples,
    warmup_count: warmups,
    iterations_per_sample: 1,
    durations_ns: durations,
    median_ns: Math.round(median(durations)),
    p95_ns: Math.round(percentile(durations, 0.95)),
    counters: emptyCounters(),
    representative: { values: [representative.loss], digest: signature([representative.loss]) },
    gradients: {
      finite: representative.gradients.every(Number.isFinite),
      norm: gradientNorm,
      representative_values: representative.gradients,
      digest: signature(representative.gradients),
    },
    loss: representative.loss,
    optimizer_step: null,
  }
}

function writeBaseline(cases: BaselineCase[]): void {
  const output = getEnv('AFFON_BASELINE_OUTPUT')
  if (!output) return
  const document = {
    schema_version: 1,
    run: {
      repository: 'affon',
      git_revision: getEnv('AFFON_BASELINE_GIT_REVISION') ?? 'unknown',
      git_dirty: (getEnv('AFFON_BASELINE_GIT_DIRTY') ?? 'false') === 'true',
      machine: getEnv('AFFON_BASELINE_MACHINE') ?? 'unknown',
      os: getEnv('AFFON_BASELINE_OS') ?? 'unknown',
      arch: getEnv('AFFON_BASELINE_ARCH') ?? 'unknown',
      compiler: getEnv('AFFON_BASELINE_COMPILER') ?? 'affon',
      build_mode: getEnv('AFFON_BASELINE_BUILD_MODE') ?? 'release',
      seed: 20260930,
    },
    cases,
  }
  fs.writeFileSync(output, JSON.stringify(document, null, 2))
}

function bench(shape: Shape): void {
  const a = parameter([shape.m, shape.k], { dtype: 'f32' }).randn().to('metal')
  const b = parameter([shape.k, shape.n], { dtype: 'f32' }).randn().to('metal')

  for (let i = 0; i < warmups; i++) matmul(a, b)

  const elapsed: number[] = []
  for (let i = 0; i < samples; i++) {
    const start = now()
    matmul(a, b)
    elapsed.push(now() - start)
  }

  const med = median(elapsed)
  const p95 = percentile(elapsed, 0.95)
  const gflops = med > 0 ? (2 * shape.m * shape.k * shape.n) / (med * 1_000_000) : 0
  console.log(`${shape.name} ${shape.m}x${shape.k}x${shape.n} median_ms=${med} p95_ms=${p95} effective_gflops=${gflops.toFixed(2)}`)
}

function benchTransposed(): void {
  const baseA = parameter([256, 1024], { dtype: 'f32' }).randn().to('metal')
  const baseB = parameter([256, 1024], { dtype: 'f32' }).randn().to('metal')
  const a = transpose(baseA, 0, 1)
  const b = transpose(baseB, 0, 1)
  try {
    for (let i = 0; i < warmups; i++) matmul(a, b)
  } catch (_) {
    console.log('transposed-both unsupported-by-current-graph-path')
    return
  }
  const elapsed: number[] = []
  for (let i = 0; i < samples; i++) {
    const start = now()
    matmul(a, b)
    elapsed.push(now() - start)
  }
  const med = median(elapsed)
  const p95 = percentile(elapsed, 0.95)
  const gflops = med > 0 ? (2 * 1024 * 256 * 1024) / (med * 1_000_000) : 0
  console.log(`transposed-both 1024x256x1024 median_ms=${med} p95_ms=${p95} effective_gflops=${gflops.toFixed(2)}`)
}

function benchBatched(): void {
  const batch = 4
  const m = 256
  const k = 256
  const n = 256
  const a = parameter([batch, m, k], { dtype: 'f32' }).randn().to('metal')
  const b = parameter([batch, k, n], { dtype: 'f32' }).randn().to('metal')
  for (let i = 0; i < warmups; i++) matmul(a, b)
  const elapsed: number[] = []
  for (let i = 0; i < samples; i++) {
    const start = now()
    matmul(a, b)
    elapsed.push(now() - start)
  }
  const med = median(elapsed)
  const p95 = percentile(elapsed, 0.95)
  const gflops = med > 0 ? (2 * batch * m * k * n) / (med * 1_000_000) : 0
  console.log(`batched-4 4x256x256x256 median_ms=${med} p95_ms=${p95} effective_gflops=${gflops.toFixed(2)}`)
}

console.log(`metal-matmul-bench warmups=${warmups} samples=${samples} mode=selected-by-AFFON_METAL_MATMUL`)
for (const shape of shapes) bench(shape)
benchTransposed()
benchBatched()
const backwardBaseline = benchBackward()
console.log(`transposed-matmul-backward median_ms=${backwardBaseline.median_ns / 1_000_000} p95_ms=${backwardBaseline.p95_ns / 1_000_000} loss=${backwardBaseline.loss} grad_norm=${backwardBaseline.gradients.norm}`)
writeBaseline([backwardBaseline])
