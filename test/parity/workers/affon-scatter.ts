import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { add, scatter_add, slice } from 'affon:ops'

const id = getEnv('AFFON_CELL_ID') ?? getEnv('AFFON_PROBE_ID') ?? ''
const shape = JSON.parse(getEnv('AFFON_CELL_SHAPE') ?? getEnv('AFFON_PROBE_SHAPE') ?? '[]') as number[]
const indexShape = JSON.parse(getEnv('AFFON_CELL_INDEX_SHAPE') ?? getEnv('AFFON_PROBE_INDEX_SHAPE') ?? '[]') as number[]
const axis = Number(getEnv('AFFON_CELL_INDEX_AXIS') ?? getEnv('AFFON_PROBE_INDEX_AXIS') ?? '1')
const kind = getEnv('AFFON_CELL_KIND') ?? 'probe'
const [backend, dtype, op] = id.split('/')
if (op !== 'scatter_add' || !['metal', 'cpu', 'cuda'].includes(backend) || !['f32', 'bf16'].includes(dtype) ||
    ![2, 3].includes(shape.length) || indexShape.length !== shape.length ||
    !Number.isInteger(axis) || axis < 0 || axis >= shape.length ||
    (shape.length === 3 && !id.includes('/batched')) ||
    indexShape.some((size, dimension) => dimension !== axis && size !== shape[dimension])) throw Error('invalid scatter cell')
const [rows, cols] = shape
const batched = shape.length === 3
const offset = id.includes('/offset_')
const stridedColumns = id.includes('/strided/cols_')
const stridedRows = id.includes('/strided/rows_')
const iterations = Number(getEnv('AFFON_CELL_ITERATIONS') ?? '10')
const samples = Number(getEnv('AFFON_CELL_SAMPLES') ?? '5')
const warmup = Number(getEnv('AFFON_CELL_WARMUP') ?? '20')
const denseInputs = (getEnv('AFFON_CELL_DENSE_INPUTS') ?? getEnv('AFFON_PROBE_DENSE_INPUTS')) === '1'
const baseValues = batched ? Array.from({ length: shape[0] }, (_, batch) =>
  Array.from({ length: shape[1] }, (_, row) =>
    Array.from({ length: shape[2] }, (_, col) => (batch * 5 + row * 3 + col) / 8))) :
  Array.from({ length: rows }, (_, row) => Array.from({ length: cols }, (_, col) => (row * 3 + col) / 8))
const indexValues = batched ? Array.from({ length: indexShape[0] }, (_, batch) =>
  Array.from({ length: indexShape[1] }, (_, row) =>
    Array.from({ length: indexShape[2] }, (_, col) => (batch * 7 + row * 3 + col * 5) % shape[axis]))) :
  Array.from({ length: indexShape[0] }, (_, row) => Array.from({ length: indexShape[1] }, (_, col) =>
    (row * 3 + col * 5) % shape[axis]))
if (id.includes('/invalid_index')) (indexValues as number[][])[0][0] = shape[axis] + 1
const updateValues = batched ? Array.from({ length: indexShape[0] }, (_, batch) =>
  Array.from({ length: indexShape[1] }, (_, row) =>
    Array.from({ length: indexShape[2] }, (_, col) => ((batch + row + col * 2) % 7 - 3) / 8))) :
  Array.from({ length: indexShape[0] }, (_, row) => Array.from({ length: indexShape[1] }, (_, col) =>
    ((row + col * 2) % 7 - 3) / 8))
const pad = <T>(values: T[][], zero: T): T[][] =>
  [Array(values[0].length + 1).fill(zero), ...values.map(row => [zero, ...row])]
const physical = (values: any, zero: number): any => batched ? values : offset ? pad(values, zero) :
  stridedColumns ? values.map(row => row.flatMap(value => [value, zero])) :
  stridedRows ? values.flatMap(row => [row, Array(row.length).fill(zero)]) : values
const basePhysical = physical(baseValues, 0)
const indexPhysical = physical(indexValues, 0)
const updatePhysical = physical(updateValues, 0)
const baseShape = batched ? shape : [basePhysical.length, basePhysical[0].length]
const indicesShape = batched ? indexShape : [indexPhysical.length, indexPhysical[0].length]
const updatesShape = batched ? indexShape : [updatePhysical.length, updatePhysical[0].length]
const spec = (dims: number[]) => dtype === 'bf16' ? Tensor.bf16(dims) : Tensor.f32(dims)
const session = new Session({ device: backend as 'cpu' | 'metal' | 'cuda', telemetry: false })
try {
  const base = session.tensor(basePhysical, { dtype: dtype as 'f32' | 'bf16' })
  const indices = session.tensor(indexPhysical, { dtype: 'i64' })
  const updates = session.tensor(updatePhysical, { dtype: dtype as 'f32' | 'bf16' })
  try {
    const source = program(`parity_${kind}_scatter_add`, p => {
      const x = p.argument('base', spec(baseShape))
      const i = p.argument('indices', Tensor.i64(indicesShape))
      const u = p.argument('updates', spec(updatesShape))
      const view = (tensor: any, dims: number[]) => offset ?
        slice(tensor, [{ start: 1, stop: dims[0] + 1 }, { start: 1, stop: dims[1] + 1 }]) :
        stridedColumns ? slice(tensor, [{ start: 0, stop: dims[0] }, { start: 0, stop: dims[1] * 2, step: 2 }]) :
        stridedRows ? slice(tensor, [{ start: 0, stop: dims[0] * 2, step: 2 }, { start: 0, stop: dims[1] }]) : tensor
      const result = scatter_add(view(x, shape), view(i, indexShape), view(u, indexShape), axis)
      return kind === 'program_peer' ? add(result, result) : result
    })
    const executable = session.compile(source, { explanationLevel: 'detailed', denseInputs })
    try {
      const inputs = { base, indices, updates }
      const run = () => {
        const result = executable.run(inputs)
        try {
          const flat = (result.to_array() as any[]).flat(Infinity) as number[]
          return { flat, output_shape: result.shape, output_dtype: result.dtype }
        } finally { result.dispose() }
      }
      const output = run()
      const pick = Array.from(new Set([0, Math.floor(output.flat.length / 4), Math.floor(output.flat.length / 2), Math.floor(output.flat.length * 3 / 4), output.flat.length - 1]))
      const baseReport = { id, shape, input_shapes: [baseShape, indicesShape, updatesShape],
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
            const result = executable.run(inputs)
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
  } finally { updates.dispose(); indices.dispose(); base.dispose() }
} finally { session.dispose() }
