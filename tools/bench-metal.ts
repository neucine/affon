import { matmul, parameter, transpose } from 'affon:compute'

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
