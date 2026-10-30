import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { add, index_select } from 'affon:ops'

const id = getEnv('AFFON_CELL_ID') ?? getEnv('AFFON_PROBE_ID') ?? ''
const shape = JSON.parse(getEnv('AFFON_CELL_SHAPE') ?? getEnv('AFFON_PROBE_SHAPE') ?? '[]') as number[]
const indexShape = JSON.parse(getEnv('AFFON_CELL_INDEX_SHAPE') ?? getEnv('AFFON_PROBE_INDEX_SHAPE') ?? '[]') as number[]
const axis = Number(getEnv('AFFON_CELL_INDEX_AXIS') ?? getEnv('AFFON_PROBE_INDEX_AXIS'))
const kind = getEnv('AFFON_CELL_KIND') ?? 'probe'
const [backend, dtype, op] = id.split('/')
if (op !== 'index_select' || !['metal', 'cpu', 'cuda'].includes(backend) ||
    !['f32', 'bf16'].includes(dtype) || !id.includes('/batched') || shape.length !== 3 ||
    indexShape.length !== 1 || ![1, 2].includes(axis) || indexShape[0] < 1) throw Error('invalid batched index_select cell')
const [batches, rows, cols] = shape
const values = Array.from({ length: batches }, (_, batch) =>
  Array.from({ length: rows }, (_, row) =>
    Array.from({ length: cols }, (_, col) => ((batch * 31 + row * 97 + col * 53) % 1009) / 8)))
const indicesValues = Array.from({ length: indexShape[0] }, (_, index) => (index * 3 + 1) % shape[axis])
const session = new Session({ device: backend as 'metal' | 'cpu' | 'cuda', telemetry: false })
const input = session.tensor(values, { dtype: dtype as 'f32' | 'bf16' })
const indices = session.tensor(indicesValues, { dtype: 'i64' })
try {
  const source = program(`parity_${kind}_index_select`, p => {
    const x = p.argument('input', dtype === 'bf16' ? Tensor.bf16(shape) : Tensor.f32(shape))
    const positions = p.argument('indices', Tensor.i64(indexShape))
    const result = index_select(x, axis, positions)
    return kind === 'program_peer' ? add(result, result) : result
  })
  const executable = session.compile(source, { explanationLevel: 'detailed', denseInputs: false })
  try {
    const run = () => {
      const result = executable.run({ input, indices })
      try { return { flat: (result.to_array() as any[]).flat(Infinity) as number[],
        output_shape: result.shape, output_dtype: result.dtype } }
      finally { result.dispose() }
    }
    const output = run()
    const pick = Array.from(new Set([0, Math.floor(output.flat.length / 4), Math.floor(output.flat.length / 2),
      Math.floor(output.flat.length * 3 / 4), output.flat.length - 1]))
    const report = { id, shape, input_shapes: [shape, indexShape], output_shape: output.output_shape,
      output_dtype: output.output_dtype, output_count: output.flat.length,
      output_checksum: output.flat.reduce((sum, value) => sum + value, 0),
      output_samples: pick.map(index => ({ index, value: output.flat[index] })) }
    const explanation = executable.explain() as any
    if (kind === 'probe') console.log(JSON.stringify({ schema_version: 1, suite: 'affon_capability_probe', ...report, explanation }))
    else {
      const selected_path = explanation.executable.physical_nodes.map((node: any) => ({
        implementation_id: node.implementation_id || (node.view_permutation?.length ? 'view.transpose' : `view.${node.view_transform}`),
        implementation_version: node.implementation_version, configuration_digest: node.configuration_digest,
        ...(node.action === 'view' ? { view_slice_ranges: node.view_slice_ranges, view_permutation: node.view_permutation } : {}) }))
      const iterations = Number(getEnv('AFFON_CELL_ITERATIONS') ?? '10')
      const samples = Number(getEnv('AFFON_CELL_SAMPLES') ?? '5')
      const warmup = Number(getEnv('AFFON_CELL_WARMUP') ?? '20')
      for (let i = 0; i < warmup; i++) run()
      const samples_ms: number[] = [], execution_ms: number[] = [], readback_ms: number[] = []
      for (let i = 0; i < samples; i++) {
        const start = performance.now()
        let execution = 0, readback = 0
        for (let j = 0; j < iterations; j++) {
          let phase = performance.now()
          const result = executable.run({ input, indices })
          execution += performance.now() - phase
          phase = performance.now()
          result.to_array()
          readback += performance.now() - phase
          result.dispose()
        }
        samples_ms.push((performance.now() - start) / iterations)
        execution_ms.push(execution / iterations)
        readback_ms.push(readback / iterations)
      }
      console.log(JSON.stringify({ schema_version: 1, suite: 'affon_cell_samples', ...report, kind,
        iterations, samples_ms, execution_ms, readback_ms, selected_path }))
    }
  } finally { executable.dispose() }
} finally { indices.dispose(); input.dispose(); session.dispose() }
