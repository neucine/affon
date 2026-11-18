// Run with a ReleaseFast Affon build and AFFON_TEST_CYNA_SESSION=1.
import { Session, Tensor, program } from 'affon:core'
import { getEnv } from 'std:process'
import { matmul, transpose } from 'affon:ops'

const size = Number(getEnv('AFFON_PROBE_SIZE') ?? '128')
if (!Number.isInteger(size) || size < 1 || size > 1024) {
  throw Error('AFFON_PROBE_SIZE must be an integer from 1 to 1024')
}
const batch = Number(getEnv('AFFON_PROBE_BATCH') ?? '1')
if (!Number.isInteger(batch) || batch < 1 || batch > 16) {
  throw Error('AFFON_PROBE_BATCH must be an integer from 1 to 16')
}
const matrix = (seed: number): number[][] => Array.from({ length: size }, (_, row) =>
  Array.from({ length: size }, (_, column) =>
    (((row * 131 + column * 17 + seed) % 101) - 50) / 101))

const transposed = getEnv('AFFON_PROBE_TRANSPOSE') !== '0'
const sampleCount = Number(getEnv('AFFON_PROBE_SAMPLES') ?? '7')
if (!Number.isInteger(sampleCount) || sampleCount < 7) throw Error('AFFON_PROBE_SAMPLES must be at least 7')
const source = program('transpose_matmul_probe', p => matmul(
  p.argument('left', Tensor.f32(batch === 1 ? [size, size] : [batch, size, size])),
  transposed
    ? transpose(p.argument('right', Tensor.f32([size, size])), [1, 0])
    : p.argument('right', Tensor.f32([size, size])),
))
const backendSymbol = Symbol.for('affon.compute.executable_backend')

function measure(forceZig: boolean) {
  const session = new Session(forceZig
    ? { device: 'cpu', cpuExecution: 'zig' }
    : { device: 'cpu' })
  const executable = session.compile(source)
  const state = session.initialize(source)
  const inputs = {
    left: session.tensor(batch === 1
      ? matrix(3)
      : Array.from({ length: batch }, (_, index) => matrix(3 + index)), { dtype: 'f32' }),
    right: session.tensor(matrix(7), { dtype: 'f32' }),
  }
  const run = (): Tensor => executable.run(inputs, state) as Tensor
  try {
    for (let index = 0; index < 3; index++) run().dispose()
    const samples: number[] = []
    for (let sample = 0; sample < sampleCount; sample++) {
      const start = performance.now()
      for (let index = 0; index < 20; index++) run().dispose()
      samples.push((performance.now() - start) / 20)
    }
    const output = run()
    try {
      const values = (output.to_array() as any).flat(batch === 1 ? 1 : 2) as number[]
      return {
        backend: (executable as any)[backendSymbol](),
        median_ms: [...samples].sort((a, b) => a - b)[Math.floor(samples.length / 2)],
        samples_ms: samples,
        values,
      }
    } finally { output.dispose() }
  } finally {
    inputs.left.dispose()
    inputs.right.dispose()
    state.dispose()
    executable.dispose()
    session.dispose()
  }
}

const cyna = measure(false)
const zig = measure(true)
if (cyna.backend !== 'cyna' || zig.backend !== 'zig') {
  throw Error(`Unexpected routes: ${cyna.backend}, ${zig.backend}`)
}
let maximumDifference = 0
for (let index = 0; index < cyna.values.length; index++) {
  const difference = Math.abs(cyna.values[index] - zig.values[index])
  const tolerance = 3e-4 + 3e-4 * Math.abs(zig.values[index])
  if (!Number.isFinite(difference) || difference > tolerance) {
    throw Error(`Logit ${index}: Cyna ${cyna.values[index]}, Zig ${zig.values[index]}`)
  }
  maximumDifference = Math.max(maximumDifference, difference)
}
console.log(JSON.stringify({
  shape: batch === 1 ? [size, size] : [batch, size, size],
  transposed,
  cyna: { backend: cyna.backend, median_ms: cyna.median_ms, samples_ms: cyna.samples_ms },
  zig: { backend: zig.backend, median_ms: zig.median_ms, samples_ms: zig.samples_ms },
  maximum_absolute_difference: maximumDifference,
  elements: cyna.values.length,
}))
