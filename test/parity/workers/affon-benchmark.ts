import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { abs, add, argmax, argmin, cast, cat, contiguous, cos, cross_entropy, div, dot, embedding, eq, erf, exp, gather, gelu, gt, clamp, index_select, layer_norm, log_softmax, rms_norm, log, lt, masked_fill, matmul, max, mean, min, mul, neg, one_hot, permute, relu, reshape, sigmoid, sign, silu, sin, slice, softmax, squeeze, sqrt, stack, std, sub, sum, tanh, transpose, unsqueeze, variance, where } from 'affon:ops'

const id = getEnv('AFFON_CELL_ID') ?? ''
const shape = JSON.parse(getEnv('AFFON_CELL_SHAPE') ?? '[]') as number[]
const iterations = Number(getEnv('AFFON_CELL_ITERATIONS') ?? '100')
const samples = Number(getEnv('AFFON_CELL_SAMPLES') ?? '5')
const warmupCalls = Number(getEnv('AFFON_CELL_WARMUP') ?? '20')
const denseInputs = getEnv('AFFON_CELL_DENSE_INPUTS') === '1'
const kind = getEnv('AFFON_CELL_KIND') ?? 'operator_peer'
const [backend, dtype, op] = id.split('/')
if (!['cpu', 'metal', 'cuda'].includes(backend) || !['f32', 'bf16'].includes(dtype) || !['add', 'sub', 'mul', 'div', 'eq', 'lt', 'gt', 'cast', 'matmul', 'dot', 'sum_axis', 'softmax', 'neg', 'exp', 'log', 'sqrt', 'relu', 'sigmoid', 'abs', 'sign', 'silu', 'tanh', 'sin', 'cos', 'layer_norm', 'log_softmax', 'rms_norm', 'mean_axis', 'min_axis', 'max_axis', 'variance_axis', 'std_axis', 'reshape', 'transpose', 'permute', 'slice', 'contiguous', 'cat', 'stack', 'squeeze', 'unsqueeze', 'embedding', 'index_select', 'gather', 'erf', 'gelu', 'clamp', 'sum_all', 'mean_all', 'min_all', 'max_all', 'variance_all', 'std_all', 'argmin_all', 'argmax_all', 'argmin_axis', 'argmax_axis', 'where', 'masked_fill', 'cross_entropy_indexed', 'one_hot'].includes(op))
  throw Error(`invalid benchmark cell: ${id}`)
if (!Number.isSafeInteger(iterations) || iterations < 1 || !Number.isSafeInteger(samples) || samples < 3)
  throw Error('invalid benchmark repetitions')
if (!Number.isSafeInteger(warmupCalls) || warmupCalls < 0) throw Error('invalid warmup calls')
if (kind !== 'operator_peer' && kind !== 'program_peer') throw Error(`invalid benchmark kind: ${kind}`)

const session = new Session({ device: backend as 'cpu' | 'metal' | 'cuda', telemetry: false })
try {
  const view = op !== 'matmul' && (id.includes('/offset') || id.includes('_offset')) ? 'offset' : op !== 'matmul' && (id.includes('/strided') || id.includes('_strided')) ? 'strided' : null
  const indexing = op === 'embedding' || op === 'index_select' || op === 'gather' || op === 'cross_entropy_indexed'
  const invalidIndices = id.includes('/invalid_index')
  const broadcasting = id.includes('/broadcast')
  const masking = op === 'where' || op === 'masked_fill'
  const comparing = op === 'eq' || op === 'lt' || op === 'gt'
  const casting = op === 'cast'
  const classes = Number(getEnv('AFFON_CELL_CLASSES') ?? '0')
  if (op === 'one_hot' && (!Number.isSafeInteger(classes) || classes < 2 || dtype !== 'f32')) throw Error('invalid one_hot contract')
  const indexShape = JSON.parse(getEnv('AFFON_CELL_INDEX_SHAPE') ?? '[]') as number[]
  const indexAxis = Number(getEnv('AFFON_CELL_INDEX_AXIS') ?? (op === 'gather' ? '1' : '0'))
  const batchedMatmul = op === 'matmul' && shape.length === 4
  const lhsMatmulLayout = op === 'matmul' && id.includes('/lhs_') ? id.includes('/transposed/') ? 'transposed' : id.includes('/strided/') ? 'strided' : null : null
  const matmulLayout = op === 'matmul' && !lhsMatmulLayout && id.includes('/transposed') ? 'transposed' : op === 'matmul' && !lhsMatmulLayout && (id.includes('/strided') || id.includes('_strided')) ? 'strided' : null
  const reductionAxis = id.includes('/outer_axis') ? 0 : 1
  const inputShape = lhsMatmulLayout === 'transposed' ? [shape[1], shape[0]] : lhsMatmulLayout === 'strided' ? [shape[0], shape[1] * 2] : op === 'matmul' ? batchedMatmul ? [shape[1], shape[2]] : [shape[0], shape[1]] : op === 'dot' ? [view === 'strided' ? shape[0] * 2 : view === 'offset' ? shape[0] + 1 : shape[0]] : view === 'strided' ? [shape[0], shape[1] * 2] : view === 'offset' ? [shape[0] + 1, shape[1] + 1] : shape
  const rhsShape = indexing ? indexShape : op === 'matmul' ? batchedMatmul ? [shape[0], shape[2], shape[3]] : matmulLayout === 'transposed' ? [shape[2], shape[1]] : matmulLayout === 'strided' ? [shape[1], shape[2] * 2] : [shape[1], shape[2]] : broadcasting ? [1, shape[1]] : inputShape
  const values = (dims: number[], seed: number): any => dims.length === 0 ? seed :
    Array.from({ length: dims[0] }, (_, index) => values(dims.slice(1), seed + index / 8))
  const viewRightValues = (dims: number[]) => Array.from({ length: dims[0] }, (_, row) =>
    Array.from({ length: dims[1] }, (_, col) => 0.5 + row / 4 + col / 16))
  const compareRightValues = (dims: number[]): any => dims.length === 0 ? 1.5 : Array.from({ length: dims[0] }, (_, row) =>
    Array.from({ length: dims[1] }, (_, col) => 1 + row / 8 + col / 8 + (((row + col) % 3) - 1) / 2))
  const castValues = (dims: number[]): any => dims.length === 0 ? 1.1 : Array.from({ length: dims[0] }, (_, row) =>
    Array.from({ length: dims[1] }, (_, col) => 1.1 + row / 7 + col / 13))
  const input = new Float32Array(inputShape.reduce((product, dimension) => product * dimension, 1))
  if (op === 'dot') for (let index = 0; index < input.length; index++) input[index] = 1 + index / 8
  else for (let row = 0; row < inputShape[0]; row++) for (let col = 0; col < inputShape[1]; col++) input[row * inputShape[1] + col] = casting ? 1.1 + row / 7 + col / 13 : 1 + row / 8 + col / 8
  const rhs = new Float32Array(indexing ? 0 : rhsShape.reduce((product, dimension) => product * dimension, 1))
  if (shape.length === 0) rhs[0] = comparing ? 1.5 : 0.5
  if (batchedMatmul) {
    for (let batch = 0; batch < rhsShape[0]; batch++) for (let row = 0; row < rhsShape[1]; row++) for (let col = 0; col < rhsShape[2]; col++)
      rhs[(batch * rhsShape[1] + row) * rhsShape[2] + col] = 0.5 + batch / 8 + row / 8 + col / 8
  } else if (op === 'dot') {
    for (let index = 0; index < rhs.length; index++) rhs[index] = 0.5 + index / 8
  } else if (!indexing) {
    for (let row = 0; row < rhsShape[0]; row++) for (let col = 0; col < rhsShape[1]; col++) rhs[row * rhsShape[1] + col] = comparing ? 1 + row / 8 + col / 8 + (((row + col) % 3) - 1) / 2 : view ? 0.5 + row / 4 + col / 16 : 0.5 + row / 8 + col / 8
  }
  if (shape.length === 0) input[0] = casting ? 1.1 : 1
  const tensorSpec = (dims: number[]) => dtype === 'bf16' ? Tensor.bf16(dims) : Tensor.f32(dims)
  const bf16Bytes = (source: Float32Array) => {
    const encoded = new Uint16Array(source.length)
    const bits = new DataView(source.buffer, source.byteOffset, source.byteLength)
    for (let i = 0; i < source.length; i++) {
      const word = bits.getUint32(i * 4, true)
      encoded[i] = ((word + 0x7fff + ((word >>> 16) & 1)) >>> 16) & 0xffff
    }
    return encoded.buffer
  }
  const oneHotIndices = op !== 'one_hot' ? null : inputShape.length === 3 ?
    Array.from({ length: inputShape[0] }, (_, batch) => Array.from({ length: inputShape[1] }, (_, row) =>
      Array.from({ length: inputShape[2] }, (_, col) => (batch * 5 + row * 3 + col) % classes))) :
    Array.from({ length: inputShape[0] }, (_, row) => Array.from({ length: inputShape[1] }, (_, col) =>
      (invalidIndices && (row + col) % 3 !== 0 ? ((row + col) % 2 ? classes + col : -1) : (row * 3 + col) % classes)))
  const x = op === 'one_hot' ? session.tensor(oneHotIndices!, { dtype: 'i64' }) : dtype === 'bf16' ? input.length > 262144 ? session.fromBytes(bf16Bytes(input), tensorSpec(inputShape)) : session.tensor(casting ? castValues(inputShape) : values(inputShape, 1), { dtype: 'bf16' }) : session.fromBytes(input.buffer, tensorSpec(inputShape))
  const indexValues = (dims: number[]) => invalidIndices ? (dims.length === 1 ? Array.from({ length: dims[0] }, (_, col) => (col % 2 === 0 ? (col * 3 + 1) % shape[indexAxis] : shape[indexAxis] + col)) :
      Array.from({ length: dims[0] }, (_, row) => Array.from({ length: dims[1] }, (_, col) => ((row + col) % 2 === 0 ? (col * 3 + 1) % shape[indexAxis] : shape[indexAxis] + col)))) :
    (dims.length === 1 ? Array.from({ length: dims[0] }, (_, col) => op === 'cross_entropy_indexed' ? (col * 3 + 1) % shape[1] : (col * 3 + 1) % shape[indexAxis]) :
      Array.from({ length: dims[0] }, (_, row) => Array.from({ length: dims[1] }, (_, col) =>
        op === 'gather' ? (col * 3 + row) % shape[indexAxis] : (row * 5 + col * 3 + 1) % shape[0])))
  const y = op === 'one_hot' && inputShape.length === 3 ? session.tensor(values(rhsShape, 0.5), { dtype: 'f32' }) : indexing ? session.tensor(indexValues(indexShape), { dtype: 'i64' }) : dtype === 'bf16' ? rhs.length > 262144 ? session.fromBytes(bf16Bytes(rhs), tensorSpec(rhsShape)) : session.tensor(comparing ? compareRightValues(rhsShape) : view && op !== 'dot' ? viewRightValues(rhsShape) : values(rhsShape, 0.5), { dtype: 'bf16' }) : session.fromBytes(rhs.buffer, tensorSpec(rhsShape))
  const maskShape = shape.length === 0 ? [] : op === 'masked_fill' && broadcasting ? [1, shape[1]] : shape
  const maskValues = (dims: number[]): any => dims.length === 0 ? 1 : Array.from({ length: dims[0] }, (_, row) => Array.from({ length: dims[1] }, (_, col) => (row + col) % 2))
  const mask = masking ? session.tensor(maskValues(maskShape), { dtype: 'i64' }) : null
  const useBias = kind === 'program_peer' && op === 'matmul'
  const biasShape = [shape[shape.length - 1]]
  const bias = new Float32Array(biasShape[0])
  for (let col = 0; col < bias.length; col++) bias[col] = col / 8 - bias.length / 16
  const z = useBias ? dtype === 'bf16' ? session.tensor(Array.from(bias), { dtype: 'bf16' }) : session.fromBytes(bias.buffer, Tensor.f32(biasShape)) : null
  try {
    const source = program(`cell_${kind}_${op}`, p => {
      const viewOf = (tensor: any) => op === 'dot' && view === 'strided' ? slice(tensor, [{ start: 0, stop: shape[0] * 2, step: 2 }]) : op === 'dot' && view === 'offset' ? slice(tensor, [{ start: 1, stop: shape[0] + 1 }]) : view === 'strided' ? slice(tensor, [{ start: 0, stop: shape[0] }, { start: 0, stop: shape[1] * 2, step: 2 }]) : view === 'offset' ? slice(tensor, [{ start: 1, stop: shape[0] + 1 }, { start: 1, stop: shape[1] + 1 }]) : tensor
      const left = p.argument('x', op === 'one_hot' ? Tensor.i64(inputShape) : tensorSpec(inputShape))
      const a = lhsMatmulLayout === 'transposed' ? transpose(left, [1, 0]) : lhsMatmulLayout === 'strided' ? slice(left, [{ start: 0, stop: shape[0] }, { start: 0, stop: shape[1] * 2, step: 2 }]) : viewOf(left)
      const right = p.argument('y', indexing ? Tensor.i64(rhsShape) : tensorSpec(rhsShape))
      const b = indexing ? right : matmulLayout === 'transposed' ? transpose(right, [1, 0]) : matmulLayout === 'strided' ? slice(right, [{ start: 0, stop: shape[1] }, { start: 0, stop: shape[2] * 2, step: 2 }]) : viewOf(right)
      const condition = masking ? p.argument('mask', Tensor.i64(maskShape)) : null
      const result = op === 'add' ? add(a, b) : op === 'mul' ? mul(a, b) : op === 'sub' ? sub(a, b) : op === 'div' ? div(a, b) : op === 'eq' ? eq(a, b) : op === 'lt' ? lt(a, b) : op === 'gt' ? gt(a, b) : op === 'cast' ? cast(a, dtype === 'f32' ? 'bf16' : 'f32') : op === 'neg' ? neg(a) : op === 'exp' ? exp(a) : op === 'log' ? log(a) : op === 'sqrt' ? sqrt(a) : op === 'relu' ? relu(a) : op === 'sigmoid' ? sigmoid(a) : op === 'abs' ? abs(a) : op === 'sign' ? sign(a) : op === 'silu' ? silu(a) : op === 'tanh' ? tanh(a) : op === 'sin' ? sin(a) : op === 'cos' ? cos(a) : op === 'erf' ? erf(a) : op === 'gelu' ? gelu(a) : op === 'clamp' ? clamp(a, 1.25, 2.25) : op === 'layer_norm' ? layer_norm(a, reductionAxis) : op === 'log_softmax' ? log_softmax(a, reductionAxis) : op === 'rms_norm' ? rms_norm(a, reductionAxis) : op === 'reshape' ? id.endsWith('/materialized') ? reshape(transpose(a, [1, 0]), [shape[0], shape[1]]) : reshape(a, [shape[1], shape[0]]) : op === 'transpose' ? transpose(a, [1, 0]) : op === 'permute' ? permute(a, [1, 0]) : op === 'slice' ? slice(a, [{ start: 0, stop: Math.floor(shape[0] / 2) }, { start: 0, stop: Math.floor(shape[1] / 2) }]) : op === 'contiguous' ? id.endsWith('/view_only') ? contiguous(a) : contiguous(transpose(a, [1, 0])) : op === 'cat' ? cat([a, b], 1) : op === 'stack' ? stack([a, b], 0) : op === 'squeeze' ? squeeze(a, 0) : op === 'unsqueeze' ? unsqueeze(a, 0) : op === 'embedding' ? embedding(a, b) : op === 'index_select' ? index_select(a, indexAxis, b) : op === 'gather' ? gather(a, indexAxis, b) : op === 'where' ? where(condition!, a, b) : op === 'masked_fill' ? masked_fill(a, condition!, -2) : op === 'one_hot' ? one_hot(a, classes) : op === 'cross_entropy_indexed' ? cross_entropy(a, b) : op === 'matmul' ? matmul(a, b) : op === 'dot' ? dot(a, b) :
        op === 'argmin_all' ? argmin(a) : op === 'argmax_all' ? argmax(a) : op === 'argmin_axis' ? argmin(a, reductionAxis) : op === 'argmax_axis' ? argmax(a, reductionAxis) : op === 'sum_all' ? sum(a) : op === 'mean_all' ? mean(a) : op === 'min_all' ? min(a) : op === 'max_all' ? max(a) : op === 'variance_all' ? variance(a) : op === 'std_all' ? std(a) : op === 'sum_axis' ? sum(a, reductionAxis) : op === 'mean_axis' ? mean(a, reductionAxis) : op === 'min_axis' ? min(a, reductionAxis) : op === 'max_axis' ? max(a, reductionAxis) : op === 'variance_axis' ? variance(a, reductionAxis) : op === 'std_axis' ? std(a, reductionAxis) : softmax(a, reductionAxis)
      if (useBias) return add(result, p.argument('bias', tensorSpec(biasShape)))
      return kind === 'program_peer' ? add(result, result) : result
    })
    const executable = session.compile(source, { explanationLevel: 'detailed', denseInputs })
    try {
      const runInputs = mask ? { x, y, mask } : z ? { x, y, bias: z } : { x, y }
      const path = (executable.explain() as any).executable.physical_nodes.map((node: any) => ({
        implementation_id: node.implementation_id || (node.view_permutation?.length ? 'view.transpose' : `view.${node.view_transform}`),
        implementation_version: node.implementation_version,
        configuration_digest: node.configuration_digest,
        ...(node.action === 'view' ? { view_slice_ranges: node.view_slice_ranges, view_permutation: node.view_permutation } : {}),
      }))
      let outputShape: number[] | null = null
      let outputDtype: string | null = null
      const run = () => {
        const result = executable.run(runInputs)
        try { outputShape ??= result.shape; outputDtype ??= result.dtype; const raw = result.to_array() as any; return (Array.isArray(raw) ? raw.flat(Infinity) : [raw]) as number[] }
        finally { result.dispose() }
      }
      const output = run()
      const indices = Array.from(new Set([0, Math.floor(output.length / 4), Math.floor(output.length / 2), Math.floor(output.length * 3 / 4), output.length - 1]))
      for (let warmup = 0; warmup < warmupCalls; warmup++) run()
      const samples_ms: number[] = []
      const execution_ms: number[] = []
      const readback_ms: number[] = []
      for (let sample = 0; sample < samples; sample++) {
        const start = performance.now()
        let execution = 0, readback = 0
        for (let iteration = 0; iteration < iterations; iteration++) {
          let phase = performance.now()
          const result = executable.run(runInputs)
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
      console.log(JSON.stringify({ schema_version: 1, suite: 'affon_cell_samples', id, shape, kind, output_shape: outputShape, output_dtype: outputDtype,
        iterations, samples_ms, execution_ms, readback_ms, output_count: output.length,
        output_checksum: output.reduce((total, value) => total + value, 0),
        output_samples: indices.map(index => ({ index, value: output[index] })), selected_path: path }))
    } finally { executable.dispose() }
  } finally { mask?.dispose(); z?.dispose(); y.dispose(); x.dispose() }
} finally { session.dispose() }
