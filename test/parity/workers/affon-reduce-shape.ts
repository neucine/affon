import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { add, reduce_to_shape, slice } from 'affon:ops'

const id = getEnv('AFFON_CELL_ID') ?? getEnv('AFFON_PROBE_ID') ?? ''
const shape = JSON.parse(getEnv('AFFON_CELL_SHAPE') ?? getEnv('AFFON_PROBE_SHAPE') ?? '[]') as number[]
const target = JSON.parse(getEnv('AFFON_TARGET_SHAPE') ?? '[]') as number[]
const kind = getEnv('AFFON_CELL_KIND') ?? 'probe'
const [backend, dtype, op] = id.split('/')
if (op !== 'reduce_to_shape' || !['metal', 'cpu', 'cuda'].includes(backend) || dtype !== 'f32' ||
    shape.length !== 2 || target.length > 2 || (target.length === 0 && !id.endsWith('/scalar'))) throw Error('invalid reduce-to-shape cell')
const [rows, cols] = shape
const offset = id.includes('/offset_')
const strided = id.split('/')[3] === 'strided' && !offset
const iterations = Number(getEnv('AFFON_CELL_ITERATIONS') ?? '10')
const samples = Number(getEnv('AFFON_CELL_SAMPLES') ?? '5')
const warmup = Number(getEnv('AFFON_CELL_WARMUP') ?? '20')
const denseInputs = (getEnv('AFFON_CELL_DENSE_INPUTS') ?? getEnv('AFFON_PROBE_DENSE_INPUTS')) === '1'
const logical = Array.from({ length: rows }, (_, row) => Array.from({ length: cols }, (_, col) => (row * 3 + col) / 8))
const physical = offset ? [Array(cols + 1).fill(0), ...logical.map(row => [0, ...row])] :
  strided ? logical.map(row => row.flatMap(value => [value, 0])) : logical
const physicalShape = [physical.length, physical[0].length]
const session = new Session({ device: backend as 'cpu' | 'metal' | 'cuda', telemetry: false })
try {
  const x = session.tensor(physical, { dtype: 'f32' })
  try {
    const source = program(`parity_${kind}_reduce_shape`, p => {
      const input = p.argument('x', Tensor.f32(physicalShape))
      const view = offset ? slice(input, [{ start: 1, stop: rows + 1 }, { start: 1, stop: cols + 1 }]) :
        strided ? slice(input, [{ start: 0, stop: rows }, { start: 0, stop: cols * 2, step: 2 }]) : input
      const result = reduce_to_shape(view, target)
      return kind === 'program_peer' ? add(result, result) : result
    })
    const executable = session.compile(source, { explanationLevel: 'detailed', denseInputs })
    try {
      const run = () => {
        const result = executable.run({ x })
        try {
          const raw = result.to_array() as number | number[] | number[][]
          return { flat: typeof raw === 'number' ? [raw] : raw.flat(Infinity) as number[], output_shape: result.shape, output_dtype: result.dtype }
        } finally { result.dispose() }
      }
      const output = run()
      const pick = Array.from(new Set([0, Math.floor(output.flat.length / 4), Math.floor(output.flat.length / 2), Math.floor(output.flat.length * 3 / 4), output.flat.length - 1]))
      const baseReport = { id, shape, input_shapes: [physicalShape],
        output_shape: output.output_shape, output_dtype: output.output_dtype, output_count: output.flat.length,
        output_checksum: output.flat.reduce((total, value) => total + value, 0),
        output_samples: pick.map(index => ({ index, value: output.flat[index] })) }
      const explanation = executable.explain() as any
      if (kind === 'probe') {
        console.log(JSON.stringify({ schema_version: 1, suite: 'affon_capability_probe', ...baseReport, explanation }))
      } else {
        const selected_path = explanation.executable.physical_nodes.map((node: any) => ({
          implementation_id: node.implementation_id || (node.view_permutation?.length ? 'view.transpose' : `view.${node.view_transform}`),
          implementation_version: node.implementation_version, configuration_digest: node.configuration_digest,
          ...(node.action === 'view' ? { view_slice_ranges: node.view_slice_ranges, view_permutation: node.view_permutation } : {}) }))
        for (let i = 0; i < warmup; i++) run()
        const samples_ms: number[] = [], execution_ms: number[] = [], readback_ms: number[] = []
        for (let i = 0; i < samples; i++) {
          const start = performance.now()
          let execution = 0, readback = 0
          for (let j = 0; j < iterations; j++) {
            let phase = performance.now()
            const result = executable.run({ x })
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
        console.log(JSON.stringify({ schema_version: 1, suite: 'affon_cell_samples', ...baseReport, kind,
          iterations, samples_ms, execution_ms, readback_ms, selected_path }))
      }
    } finally { executable.dispose() }
  } finally { x.dispose() }
} finally { session.dispose() }
