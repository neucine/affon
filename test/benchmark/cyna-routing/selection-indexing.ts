// Run with a ReleaseFast Affon build and AFFON_TEST_CYNA_SESSION=1.
import { Session, Tensor, program } from 'affon:core'
import { embedding, eq, scatter_add, topk, where } from 'affon:ops'

const backendSymbol = Symbol.for('affon.compute.executable_backend')
const routingSymbol = Symbol.for('affon.compute.executable_routing')
const samplesPerRoute = 7
const runsPerSample = 20
type Inputs = Record<string, Tensor>
type Case = Readonly<{
  name: string
  source: any
  inputs: (session: Session) => Inputs
}>

const matrix = (rows: number, columns: number, seed = 0): number[][] =>
  Array.from({ length: rows }, (_, row) =>
    Array.from({ length: columns }, (_, column) =>
      ((row * 17 + column * 13 + seed) % 101) / 101))

const cases: Case[] = [
  {
    name: 'broadcast_comparison',
    source: program('selection_bench_comparison', p => eq(
      p.argument('left', Tensor.f32([128, 128])),
      p.argument('right', Tensor.f32([1, 128])),
    )),
    inputs: session => ({
      left: session.tensor(matrix(128, 128), { dtype: 'f32' }),
      right: session.tensor(matrix(1, 128), { dtype: 'f32' }),
    }),
  },
  {
    name: 'broadcast_where',
    source: program('selection_bench_where', p => where(
      p.argument('mask', Tensor.i64([128, 1])),
      p.argument('when_true', Tensor.f32([128, 128])),
      p.argument('when_false', Tensor.f32([1, 128])),
    )),
    inputs: session => ({
      mask: session.tensor(Array.from({ length: 128 }, (_, row) => [row % 2]), { dtype: 'i64' }),
      when_true: session.tensor(matrix(128, 128, 1), { dtype: 'f32' }),
      when_false: session.tensor(matrix(1, 128, 2), { dtype: 'f32' }),
    }),
  },
  {
    name: 'embedding',
    source: program('selection_bench_embedding', p => embedding(
      p.argument('table', Tensor.f32([256, 64])),
      p.argument('indices', Tensor.i64([128])),
    )),
    inputs: session => ({
      table: session.tensor(matrix(256, 64), { dtype: 'f32' }),
      indices: session.tensor(Array.from({ length: 128 }, (_, index) => (index * 7) % 256), { dtype: 'i64' }),
    }),
  },
  {
    name: 'scatter_add',
    source: program('selection_bench_scatter', p => scatter_add(
      p.argument('base', Tensor.f32([128, 64])),
      p.argument('indices', Tensor.i64([64, 64])),
      p.argument('updates', Tensor.f32([64, 64])), 0,
    )),
    inputs: session => ({
      base: session.tensor(matrix(128, 64), { dtype: 'f32' }),
      indices: session.tensor(Array.from({ length: 64 }, (_, row) =>
        Array.from({ length: 64 }, () => (row * 3) % 128)), { dtype: 'i64' }),
      updates: session.tensor(matrix(64, 64, 3), { dtype: 'f32' }),
    }),
  },
  {
    name: 'topk',
    source: program('selection_bench_topk', p => topk(
      p.argument('value', Tensor.f32([64, 64])), 8, 1,
    )),
    inputs: session => ({ value: session.tensor(matrix(64, 64), { dtype: 'f32' }) }),
  },
]

function measure(entry: Case, forceZig: boolean) {
  const session = new Session(forceZig
    ? { device: 'cpu', telemetry: { captureRecords: 32 } }
    : { device: 'cpu' })
  const executable = session.compile(entry.source)
  const state = session.initialize(entry.source)
  const inputs = entry.inputs(session)
  const run = (): Tensor[] => {
    const result = executable.run(inputs, state)
    return Array.isArray(result) ? result : [result]
  }
  try {
    const backend = (executable as any)[backendSymbol]()
    const reason = (executable as any)[routingSymbol]().reason
    if (backend !== (forceZig ? 'zig' : 'cyna') || reason !== (forceZig ? 'evidence_capture' : 'lowered')) {
      throw Error(`${entry.name}: unexpected route ${backend}/${reason}`)
    }
    for (let warm = 0; warm < 3; warm++) for (const output of run()) output.dispose()
    const samples: number[] = []
    for (let sample = 0; sample < samplesPerRoute; sample++) {
      const start = performance.now()
      for (let repeat = 0; repeat < runsPerSample; repeat++) {
        for (const output of run()) output.dispose()
      }
      samples.push((performance.now() - start) * 1000 / runsPerSample)
    }
    const result = run()
    try {
      return {
        backend,
        median_us: [...samples].sort((a, b) => a - b)[Math.floor(samples.length / 2)],
        samples_us: samples,
        values: result.map(output => (output.to_array() as any[]).flat(Infinity) as number[]),
      }
    } finally { for (const output of result) output.dispose() }
  } finally {
    for (const input of Object.values(inputs)) input.dispose()
    state.dispose()
    executable.dispose()
    session.dispose()
  }
}

const results = cases.map(entry => {
  const cyna = measure(entry, false)
  const zig = measure(entry, true)
  if (cyna.values.length !== zig.values.length) throw Error(`${entry.name}: output count differs`)
  let maximumDifference = 0
  for (let output = 0; output < cyna.values.length; output++) {
    if (cyna.values[output].length !== zig.values[output].length) throw Error(`${entry.name}: output shape differs`)
    for (let index = 0; index < cyna.values[output].length; index++) {
      const actual = cyna.values[output][index]
      const expected = zig.values[output][index]
      const difference = Math.abs(actual - expected)
      if (!Number.isFinite(difference) || difference > 3e-4 + 3e-4 * Math.abs(expected)) {
        throw Error(`${entry.name}: output ${output}, element ${index}: Cyna ${actual}, Zig ${expected}`)
      }
      maximumDifference = Math.max(maximumDifference, difference)
    }
  }
  return {
    name: entry.name,
    cyna_median_us: cyna.median_us,
    zig_median_us: zig.median_us,
    ratio: cyna.median_us / zig.median_us,
    maximum_difference: maximumDifference,
  }
})

console.log(JSON.stringify({ samples_per_route: samplesPerRoute, runs_per_sample: runsPerSample, results }, null, 2))
