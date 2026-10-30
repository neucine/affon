import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { add, slice, topk } from 'affon:ops'

const id = getEnv('AFFON_CELL_ID') ?? getEnv('AFFON_PROBE_ID') ?? ''
const shape = JSON.parse(getEnv('AFFON_CELL_SHAPE') ?? getEnv('AFFON_PROBE_SHAPE') ?? '[]') as number[]
const axis = Number(getEnv('AFFON_TOPK_AXIS'))
const k = Number(getEnv('AFFON_TOPK_K'))
const largest = (getEnv('AFFON_TOPK_LARGEST') ?? '1') === '1'
const kind = getEnv('AFFON_CELL_KIND') ?? 'probe'
const [backend, dtype, op, , variant] = id.split('/')
const layout = variant?.split('_')[0] ?? (id.split('/')[3] === 'strided' ? 'strided' : 'dense')
if (op !== 'topk' || !['metal', 'cpu', 'cuda'].includes(backend) || !['f32', 'bf16'].includes(dtype) ||
    !['dense', 'offset', 'strided'].includes(layout) || ![2, 3].includes(shape.length) ||
    (shape.length === 3 && (!id.includes('/batched') || layout !== 'dense')) ||
    !Number.isInteger(axis) || axis < 0 || axis >= shape.length ||
    !Number.isInteger(k) || k < 1 || k > shape[axis]) throw Error('invalid topk cell')
const [rows, cols] = shape
const physicalShape = shape.length === 3 ? shape :
  [rows + (layout === 'offset' ? 1 : 0), cols * (layout === 'strided' ? 2 : 1) + (layout === 'offset' ? 1 : 0)]
const physical = shape.length === 3 ? Array.from({ length: shape[0] }, (_, batch) =>
  Array.from({ length: shape[1] }, (_, row) =>
    Array.from({ length: shape[2] }, (_, col) => ((batch * 31 + row * 97 + col * 53) % 1009) / 8))) :
  Array.from({ length: physicalShape[0] }, (_, row) =>
    Array.from({ length: physicalShape[1] }, (_, col) => ((row * 97 + col * 53) % 1009) / 8))
const outputHash = (values: number[]) => values.reduce((hash, value) =>
  Math.imul(hash ^ (Math.round(value * 8) >>> 0), 16777619) >>> 0, 2166136261)
const session = new Session({ device: backend as 'cpu' | 'metal' | 'cuda', telemetry: false })
const input = session.tensor(physical, { dtype: dtype as 'f32' | 'bf16' })
try {
  const source = program(`parity_${kind}_topk`, p => {
    const x = p.argument('input', dtype === 'bf16' ? Tensor.bf16(physicalShape) : Tensor.f32(physicalShape))
    const viewed = shape.length === 3 ? x : layout === 'offset' ? slice(x, [{ start: 1, stop: rows + 1 }, { start: 1, stop: cols + 1 }]) :
      layout === 'strided' ? slice(x, [{ start: 0, stop: rows }, { start: 0, stop: cols * 2, step: 2 }]) : x
    const [values, indices] = topk(viewed, k, axis, largest)
    return kind === 'program_peer' ? [add(values, values), indices] : [values, indices]
  })
  const executable = session.compile(source, { explanationLevel: 'detailed', denseInputs: false })
  try {
    const run = () => {
      const [values, indices] = executable.run({ input })
      try {
        const left = (values.to_array() as any[]).flat(Infinity) as number[]
        const right = (indices.to_array() as any[]).flat(Infinity) as number[]
        return { flat: [...left, ...right], outputs: [
          { shape: values.shape, dtype: values.dtype, count: left.length },
          { shape: indices.shape, dtype: indices.dtype, count: right.length }] }
      } finally { values.dispose(); indices.dispose() }
    }
    const output = run()
    const pick = Array.from(new Set([0, Math.floor(output.flat.length / 4), Math.floor(output.flat.length / 2),
      Math.floor(output.flat.length * 3 / 4), output.flat.length - 1]))
    const report = { id, shape, input_shapes: [physicalShape], output_shape: output.outputs[0].shape,
      output_dtype: `pair:${output.outputs[0].dtype},${output.outputs[1].dtype}`, outputs: output.outputs,
      output_count: output.flat.length, output_hash: outputHash(output.flat),
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
          const pair = executable.run({ input })
          execution += performance.now() - phase
          phase = performance.now()
          pair[0].to_array(); pair[1].to_array()
          readback += performance.now() - phase
          pair[0].dispose(); pair[1].dispose()
        }
        samples_ms.push((performance.now() - start) / iterations)
        execution_ms.push(execution / iterations)
        readback_ms.push(readback / iterations)
      }
      console.log(JSON.stringify({ schema_version: 1, suite: 'affon_cell_samples', ...report, kind,
        iterations, samples_ms, execution_ms, readback_ms, selected_path }))
    }
  } finally { executable.dispose() }
} finally { input.dispose(); session.dispose() }
