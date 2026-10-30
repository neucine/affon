// Run with a ReleaseFast Affon build. Construction and compilation are untimed.
import { Session, Tensor, program } from 'affon:compute'
import { quantized_matmul } from 'affon:ops'

for (const device of ['cpu', 'metal'] as const) for (const rows of [1, 8]) {
  const n = 2048, k = 1024, group_size = 64
  const session = new Session({ device })
  const source = program('quantized_benchmark', p => quantized_matmul(
    p.argument('x', Tensor.f32([rows, k])), p.argument('w', Tensor.u32([n, k / 4])),
    p.argument('s', Tensor.bf16([n, k / group_size])), p.argument('b', Tensor.bf16([n, k / group_size])),
    { bits: 8, group_size },
  ))
  const executable = session.compile(source)
  const x = session.tensor(Array.from({ length: rows }, (_, row) => Array.from({ length: k }, (_, col) => ((row + col) % 17 - 8) / 16)))
  const w = session.tensor(Array.from({ length: n }, (_, row) => Array.from({ length: k / 4 }, (_, word) => {
    let packed = 0
    for (let lane = 0; lane < 4; lane++) packed |= ((row * 3 + (word * 4 + lane) * 7) & 255) << (lane * 8)
    return packed >>> 0
  })), { dtype: 'u32' })
  const s = session.tensor(Array.from({ length: n }, () => Array(k / group_size).fill(1 / 128)), { dtype: 'bf16' })
  const b = session.tensor(Array.from({ length: n }, () => Array(k / group_size).fill(-1)), { dtype: 'bf16' })
  const run = () => {
    const y = executable.run({ x, w, s, b })
    y.to_array() // Completion boundary: includes GPU synchronization/readback.
    y.dispose()
  }
  for (let warmup = 0; warmup < 3; warmup++) run()
  const samples = []
  const iterations = 10
  for (let sample = 0; sample < 5; sample++) {
    const start = Date.now()
    for (let iteration = 0; iteration < iterations; iteration++) run()
    samples.push((Date.now() - start) / iterations)
  }
  samples.sort((a, b) => a - b)
  console.log(JSON.stringify({ device, m: rows, n, k, bits: 8, group_size, iterations,
    median_ms: samples[2], samples_ms: samples, includes_readback: true,
    packed_and_parameters_bytes: n * k + 2 * n * (k / group_size) * 2,
    dense_f32_weights_bytes: n * k * 4,
  }))
  b.dispose(); s.dispose(); w.dispose(); x.dispose(); session.dispose()
}
