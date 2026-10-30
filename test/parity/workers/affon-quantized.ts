import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { add, quantized_matmul, slice, transpose } from 'affon:ops'

const id = getEnv('AFFON_CELL_ID') ?? getEnv('AFFON_PROBE_ID') ?? ''
const shape = JSON.parse(getEnv('AFFON_CELL_SHAPE') ?? getEnv('AFFON_PROBE_SHAPE') ?? '[]') as number[]
const bits = Number(getEnv('AFFON_Q_BITS'))
const group_size = Number(getEnv('AFFON_Q_GROUP'))
const kind = getEnv('AFFON_CELL_KIND') ?? 'probe'
const [backend, dtype, op] = id.split('/')
if (op !== 'quantized_matmul' || !['metal', 'cpu', 'cuda'].includes(backend) ||
    !['f32', 'bf16'].includes(dtype) || shape.length !== 3 ||
    ![2, 4, 8].includes(bits) || !Number.isSafeInteger(group_size)) throw Error('invalid quantized cell')
const [rows, inner, outputs] = shape
const transposed = id.includes('/transposed/')
const offset = id.includes('/offset_')
if (transposed && offset) throw Error('ambiguous quantized layout')
const perWord = 32 / bits
if (inner % perWord || inner % group_size || rows < 1 || outputs < 1) throw Error('invalid quantized shape')
const iterations = Number(getEnv('AFFON_CELL_ITERATIONS') ?? '10')
const samples = Number(getEnv('AFFON_CELL_SAMPLES') ?? '5')
const warmup = Number(getEnv('AFFON_CELL_WARMUP') ?? '20')
const denseInputs = (getEnv('AFFON_CELL_DENSE_INPUTS') ?? getEnv('AFFON_PROBE_DENSE_INPUTS')) === '1'
const xValues = Array.from({ length: rows }, (_, row) => Array.from({ length: inner }, (_, col) => ((row * 3 + col) % 9 - 4) / 8))
const mask = (1 << bits) - 1
const q = Array.from({ length: outputs }, (_, out) => Array.from({ length: inner }, (_, col) => (out * 3 + col * 5) & mask))
const packedValues = Array.from({ length: outputs }, (_, out) => Array.from({ length: inner / perWord }, (_, word) =>
  q[out].slice(word * perWord, (word + 1) * perWord).reduce((value, digit, lane) => (value | (digit << (lane * bits))) >>> 0, 0)))
const scalesValues = Array.from({ length: outputs }, (_, out) => Array.from({ length: inner / group_size }, (_, block) => (1 + (out + block) % 3) / 4))
const biasesValues = Array.from({ length: outputs }, (_, out) => Array.from({ length: inner / group_size }, (_, block) => ((out + block) % 3 - 1) / 8))
const pad = <T>(values: T[][], zero: T): T[][] =>
  [Array(values[0].length + 1).fill(zero), ...values.map(row => [zero, ...row])]
const xPhysical = transposed ? Array.from({ length: inner }, (_, col) => xValues.map(row => row[col])) : offset ? pad(xValues, 0) : xValues
const wPhysical = transposed ? packedValues.map(row => [0, ...row]) : offset ? pad(packedValues, 0) : packedValues
const sPhysical = offset ? pad(scalesValues, 0) : scalesValues
const bPhysical = offset ? pad(biasesValues, 0) : biasesValues
const xShape = [xPhysical.length, xPhysical[0].length]
const wShape = [wPhysical.length, wPhysical[0].length]
const sShape = [sPhysical.length, sPhysical[0].length]
const bShape = [bPhysical.length, bPhysical[0].length]
const spec = (dims: number[]) => dtype === 'bf16' ? Tensor.bf16(dims) : Tensor.f32(dims)
const session = new Session({ device: backend as 'cpu' | 'metal' | 'cuda', telemetry: false })
try {
  const x = session.tensor(xPhysical, { dtype: dtype as 'f32' | 'bf16' })
  const w = session.tensor(wPhysical, { dtype: 'u32' })
  const s = session.tensor(sPhysical, { dtype: dtype as 'f32' | 'bf16' })
  const b = session.tensor(bPhysical, { dtype: dtype as 'f32' | 'bf16' })
  try {
    const source = program(`parity_${kind}_quantized_matmul`, p => {
      const xArg = p.argument('x', spec(xShape))
      const wArg = p.argument('w', Tensor.u32(wShape))
      const sArg = p.argument('s', spec(sShape))
      const bArg = p.argument('b', spec(bShape))
      const view = (tensor: any, height: number, width: number) =>
        slice(tensor, [{ start: 1, stop: height + 1 }, { start: 1, stop: width + 1 }])
      const logicalX = transposed ? transpose(xArg, [1, 0]) : offset ? view(xArg, rows, inner) : xArg
      const logicalW = transposed ? slice(wArg, [{ start: 0, stop: outputs }, { start: 1, stop: inner / perWord + 1 }]) :
        offset ? view(wArg, outputs, inner / perWord) : wArg
      const logicalS = offset ? view(sArg, outputs, inner / group_size) : sArg
      const logicalB = offset ? view(bArg, outputs, inner / group_size) : bArg
      const result = quantized_matmul(logicalX, logicalW, logicalS, logicalB, { bits, group_size })
      return kind === 'program_peer' ? add(result, result) : result
    })
    const executable = session.compile(source, { explanationLevel: 'detailed', denseInputs })
    try {
      const inputs = { x, w, s, b }
      const run = () => {
        const result = executable.run(inputs)
        try {
          const flat = (result.to_array() as number[][]).flat()
          return { flat, output_shape: result.shape, output_dtype: result.dtype }
        } finally { result.dispose() }
      }
      const output = run()
      const indices = Array.from(new Set([0, Math.floor(output.flat.length / 4), Math.floor(output.flat.length / 2), Math.floor(output.flat.length * 3 / 4), output.flat.length - 1]))
      const base = { id, shape, input_shapes: [xShape, wShape, sShape, bShape],
        output_shape: output.output_shape, output_dtype: output.output_dtype, output_count: output.flat.length,
        output_checksum: output.flat.reduce((total, value) => total + value, 0),
        output_samples: indices.map(index => ({ index, value: output.flat[index] })) }
      const explanation = executable.explain() as any
      if (kind === 'probe') {
        console.log(JSON.stringify({ schema_version: 1, suite: 'affon_capability_probe', ...base, explanation }))
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
        console.log(JSON.stringify({ schema_version: 1, suite: 'affon_cell_samples', ...base, kind, iterations,
          samples_ms, execution_ms, readback_ms, selected_path }))
      }
    } finally { executable.dispose() }
  } finally { b.dispose(); s.dispose(); w.dispose(); x.dispose() }
} finally { session.dispose() }
