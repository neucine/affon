import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { add, dense_cross_entropy, log_softmax_nll, slice } from 'affon:ops'

const id = getEnv('AFFON_CELL_ID') ?? getEnv('AFFON_PROBE_ID') ?? ''
const shape = JSON.parse(getEnv('AFFON_CELL_SHAPE') ?? getEnv('AFFON_PROBE_SHAPE') ?? '[]') as number[]
const kind = getEnv('AFFON_CELL_KIND') ?? 'probe'
const [backend, dtype, op] = id.split('/')
if (!['cross_entropy', 'log_softmax_nll'].includes(op) || !['metal', 'cpu', 'cuda'].includes(backend) ||
    !['f32', 'bf16'].includes(dtype) || ![2, 3].includes(shape.length) || shape[0] < 1 ||
    shape[1] < 3 || (shape.length === 3 && (shape[2] < 1 || !id.includes('/batched')))) throw Error('invalid dense loss cell')
const [rows, classes, spatial] = shape
const batched = shape.length === 3
const offset = id.includes('/offset_')
const stridedColumns = id.includes('/strided/cols_')
const stridedRows = id.includes('/strided/rows_')
const iterations = Number(getEnv('AFFON_CELL_ITERATIONS') ?? '10')
const samples = Number(getEnv('AFFON_CELL_SAMPLES') ?? '5')
const warmup = Number(getEnv('AFFON_CELL_WARMUP') ?? '20')
const denseInputs = (getEnv('AFFON_CELL_DENSE_INPUTS') ?? getEnv('AFFON_PROBE_DENSE_INPUTS')) === '1'
const logitsValues = batched ? Array.from({ length: rows }, (_, batch) =>
  Array.from({ length: classes }, (_, col) => Array.from({ length: spatial }, (_, position) =>
    (batch * 5 + position * 3 + col) / 8))) :
  Array.from({ length: rows }, (_, row) => Array.from({ length: classes }, (_, col) => (row * 3 + col) / 8))
const targetValues = batched ? Array.from({ length: rows }, (_, batch) =>
  Array.from({ length: classes }, (_, col) => Array.from({ length: spatial }, (_, position) =>
    col === (batch + position * 3 + 1) % classes || col === (batch + position * 3 + 2) % classes ? 0.5 : 0))) :
  Array.from({ length: rows }, (_, row) => Array.from({ length: classes }, (_, col) =>
    col === (row * 3 + 1) % classes || col === (row * 3 + 2) % classes ? 0.5 : 0))
const pad = (values: number[][]) => [Array(values[0].length + 1).fill(0), ...values.map(row => [0, ...row])]
const physical = (values: any) => batched ? values : offset ? pad(values) :
  stridedColumns ? values.map(row => row.flatMap(value => [value, 0])) :
  stridedRows ? values.flatMap(row => [row, Array(row.length).fill(0)]) : values
const logitsPhysical = physical(logitsValues)
const targetsPhysical = physical(targetValues)
const physicalShape = batched ? shape : [logitsPhysical.length, logitsPhysical[0].length]
const spec = dtype === 'bf16' ? Tensor.bf16(physicalShape) : Tensor.f32(physicalShape)
const session = new Session({ device: backend as 'cpu' | 'metal' | 'cuda', telemetry: false })
try {
      const logits = session.tensor(logitsPhysical, { dtype: dtype as 'f32' | 'bf16' })
  const targets = session.tensor(targetsPhysical, { dtype: dtype as 'f32' | 'bf16' })
  try {
    const source = program(`parity_${kind}_dense_loss`, p => {
      const scores = p.argument('logits', spec)
      const weights = p.argument('targets', spec)
      const view = (tensor: any) => offset ? slice(tensor, [{ start: 1, stop: rows + 1 }, { start: 1, stop: classes + 1 }]) :
        stridedColumns ? slice(tensor, [{ start: 0, stop: rows }, { start: 0, stop: classes * 2, step: 2 }]) :
        stridedRows ? slice(tensor, [{ start: 0, stop: rows * 2, step: 2 }, { start: 0, stop: classes }]) : tensor
      const result = op === 'log_softmax_nll' ? log_softmax_nll(view(scores), view(weights)) : dense_cross_entropy(view(scores), view(weights))
      return kind === 'program_peer' ? add(result, result) : result
    })
    const executable = session.compile(source, { explanationLevel: 'detailed', denseInputs })
    try {
      const inputs = { logits, targets }
      const run = () => {
        const result = executable.run(inputs)
        try {
          const raw = result.to_array() as number[]
          return { flat: raw.flat(), output_shape: result.shape, output_dtype: result.dtype }
        } finally { result.dispose() }
      }
      const output = run()
      const baseReport = { id, shape, input_shapes: [physicalShape, physicalShape],
        output_shape: output.output_shape, output_dtype: output.output_dtype, output_count: output.flat.length,
        output_checksum: output.flat.reduce((total, value) => total + value, 0),
        output_samples: output.flat.map((value, index) => ({ index, value })) }
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
  } finally { targets.dispose(); logits.dispose() }
} finally { session.dispose() }
