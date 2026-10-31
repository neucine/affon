import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { abs, add, argmax, argmin, cast, cat, contiguous, cos, cross_entropy, div, dot, embedding, eq, erf, exp, gather, gelu, gt, clamp, index_select, layer_norm, log_softmax, rms_norm, log, lt, masked_fill, matmul, max, mean, min, mul, neg, one_hot, permute, relu, reshape, sigmoid, sign, silu, sin, slice, softmax, squeeze, sqrt, stack, std, sub, sum, tanh, transpose, unsqueeze, variance, where } from 'affon:ops'

const id = getEnv('AFFON_PROBE_ID') ?? ''
const shape = JSON.parse(getEnv('AFFON_PROBE_SHAPE') ?? '[]') as number[]
const [backend, dtype, op] = id.split('/')
if (!['cpu', 'metal', 'cuda'].includes(backend) || !['f32', 'bf16'].includes(dtype) || !['add', 'sub', 'mul', 'div', 'eq', 'lt', 'gt', 'cast', 'matmul', 'dot', 'sum_axis', 'softmax', 'neg', 'exp', 'log', 'sqrt', 'relu', 'sigmoid', 'abs', 'sign', 'silu', 'tanh', 'sin', 'cos', 'layer_norm', 'log_softmax', 'rms_norm', 'mean_axis', 'min_axis', 'max_axis', 'variance_axis', 'std_axis', 'reshape', 'transpose', 'permute', 'slice', 'contiguous', 'cat', 'stack', 'squeeze', 'unsqueeze', 'embedding', 'index_select', 'gather', 'erf', 'gelu', 'clamp', 'sum_all', 'mean_all', 'min_all', 'max_all', 'variance_all', 'std_all', 'argmin_all', 'argmax_all', 'argmin_axis', 'argmax_axis', 'where', 'masked_fill', 'cross_entropy_indexed', 'one_hot'].includes(op))
  throw Error(`invalid probe: ${id}`)

const session = new Session({ device: backend as 'cpu' | 'metal' | 'cuda' })
try {
  const batchedMatmul = op === 'matmul' && shape.length === 4
  const indexing = op === 'embedding' || op === 'index_select' || op === 'gather' || op === 'cross_entropy_indexed'
  const invalidIndices = id.includes('/invalid_index')
  const broadcasting = id.includes('/broadcast')
  const masking = op === 'where' || op === 'masked_fill'
  const comparing = op === 'eq' || op === 'lt' || op === 'gt'
  const casting = op === 'cast'
  const classes = Number(getEnv('AFFON_PROBE_CLASSES') ?? '0')
  if (op === 'one_hot' && (!Number.isSafeInteger(classes) || classes < 2 || dtype !== 'f32')) throw Error('invalid one_hot contract')
  const indexShape = JSON.parse(getEnv('AFFON_PROBE_INDEX_SHAPE') ?? '[]') as number[]
  const indexAxis = Number(getEnv('AFFON_PROBE_INDEX_AXIS') ?? (op === 'gather' ? '1' : '0'))
  const reductionAxis = id.includes('/outer_axis') ? 0 : 1
  const view = op !== 'matmul' && (id.includes('/offset') || id.includes('_offset')) ? 'offset' : op !== 'matmul' && (id.includes('/strided') || id.includes('_strided')) ? 'strided' : null
  const lhsMatmulLayout = op === 'matmul' && id.includes('/lhs_') ? id.includes('/transposed/') ? 'transposed' : id.includes('/strided/') ? 'strided' : null : null
  const matmulLayout = op === 'matmul' && !lhsMatmulLayout && id.includes('/transposed') ? 'transposed' : op === 'matmul' && !lhsMatmulLayout && (id.includes('/strided') || id.includes('_strided')) ? 'strided' : null
  const logicalInputShape = op === 'matmul' ? (batchedMatmul ? [shape[1], shape[2]] : [shape[0], shape[1]]) : shape
  const inputShape = lhsMatmulLayout === 'transposed' ? [shape[1], shape[0]] : lhsMatmulLayout === 'strided' ? [shape[0], shape[1] * 2] : op === 'dot' ? [view === 'strided' ? shape[0] * 2 : view === 'offset' ? shape[0] + 1 : shape[0]] : view === 'strided' ? [shape[0], shape[1] * 2] : view === 'offset' ? [shape[0] + 1, shape[1] + 1] : logicalInputShape
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
  const rhsShape = indexing ? indexShape : op === 'matmul' ? (batchedMatmul ? [shape[0], shape[2], shape[3]] : matmulLayout === 'transposed' ? [shape[2], shape[1]] : matmulLayout === 'strided' ? [shape[1], shape[2] * 2] : [shape[1], shape[2]]) : broadcasting ? [1, shape[1]] : inputShape
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
  const indexValues = (dims: number[]) => invalidIndices ? (dims.length === 1 ? Array.from({ length: dims[0] }, (_, col) => (col % 2 === 0 ? (col * 3 + 1) % shape[indexAxis] : shape[indexAxis] + col)) :
      Array.from({ length: dims[0] }, (_, row) => Array.from({ length: dims[1] }, (_, col) => ((row + col) % 2 === 0 ? (col * 3 + 1) % shape[indexAxis] : shape[indexAxis] + col)))) :
    (dims.length === 1 ? Array.from({ length: dims[0] }, (_, col) => op === 'cross_entropy_indexed' ? (col * 3 + 1) % shape[1] : (col * 3 + 1) % shape[indexAxis]) :
      Array.from({ length: dims[0] }, (_, row) => Array.from({ length: dims[1] }, (_, col) =>
        op === 'gather' ? (col * 3 + row) % shape[indexAxis] : (row * 5 + col * 3 + 1) % shape[0])))
  const y = op === 'one_hot' && inputShape.length === 3 ? session.tensor(values(rhsShape, 0.5), { dtype: 'f32' }) : indexing ? session.tensor(indexValues(indexShape), { dtype: 'i64' }) : dtype === 'bf16' ? rhs.length > 262144 ? session.fromBytes(bf16Bytes(rhs), tensorSpec(rhsShape)) : session.tensor(comparing ? compareRightValues(rhsShape) : view && op !== 'dot' ? viewRightValues(rhsShape) : values(rhsShape, 0.5), { dtype: 'bf16' }) : session.fromBytes(rhs.buffer, tensorSpec(rhsShape))
  const maskShape = shape.length === 0 ? [] : op === 'masked_fill' && broadcasting ? [1, shape[1]] : shape
  const maskValues = (dims: number[]): any => dims.length === 0 ? 1 : Array.from({ length: dims[0] }, (_, row) => Array.from({ length: dims[1] }, (_, col) => (row + col) % 2))
  const mask = masking ? session.tensor(maskValues(maskShape), { dtype: 'i64' }) : null
  try {
    const source = program(`probe_${op}`, p => {
      const viewOf = (tensor: any) => op === 'dot' && view === 'strided' ? slice(tensor, [{ start: 0, stop: shape[0] * 2, step: 2 }]) : op === 'dot' && view === 'offset' ? slice(tensor, [{ start: 1, stop: shape[0] + 1 }]) : view === 'strided' ? slice(tensor, [{ start: 0, stop: shape[0] }, { start: 0, stop: shape[1] * 2, step: 2 }]) : view === 'offset' ? slice(tensor, [{ start: 1, stop: shape[0] + 1 }, { start: 1, stop: shape[1] + 1 }]) : tensor
      const left = p.argument('x', op === 'one_hot' ? Tensor.i64(inputShape) : tensorSpec(inputShape))
      const a = lhsMatmulLayout === 'transposed' ? transpose(left, [1, 0]) : lhsMatmulLayout === 'strided' ? slice(left, [{ start: 0, stop: shape[0] }, { start: 0, stop: shape[1] * 2, step: 2 }]) : viewOf(left)
      const right = p.argument('y', indexing ? Tensor.i64(rhsShape) : tensorSpec(rhsShape))
      const b = indexing ? right : matmulLayout === 'transposed' ? transpose(right, [1, 0]) : matmulLayout === 'strided' ? slice(right, [{ start: 0, stop: shape[1] }, { start: 0, stop: shape[2] * 2, step: 2 }]) : viewOf(right)
      const condition = masking ? p.argument('mask', Tensor.i64(maskShape)) : null
      switch (op) {
        case 'add': return add(a, b)
        case 'mul': return mul(a, b)
        case 'sub': return sub(a, b)
        case 'div': return div(a, b)
        case 'eq': return eq(a, b)
        case 'lt': return lt(a, b)
        case 'gt': return gt(a, b)
        case 'cast': return cast(a, dtype === 'f32' ? 'bf16' : 'f32')
        case 'neg': return neg(a)
        case 'exp': return exp(a)
        case 'log': return log(a)
        case 'sqrt': return sqrt(a)
        case 'relu': return relu(a)
        case 'sigmoid': return sigmoid(a)
        case 'abs': return abs(a)
        case 'sign': return sign(a)
        case 'silu': return silu(a)
        case 'tanh': return tanh(a)
        case 'sin': return sin(a)
        case 'cos': return cos(a)
        case 'erf': return erf(a)
        case 'gelu': return gelu(a)
        case 'clamp': return clamp(a, 1.25, 2.25)
        case 'layer_norm': return layer_norm(a, reductionAxis)
        case 'log_softmax': return log_softmax(a, reductionAxis)
        case 'rms_norm': return rms_norm(a, reductionAxis)
        case 'mean_axis': return mean(a, reductionAxis)
        case 'min_axis': return min(a, reductionAxis)
        case 'max_axis': return max(a, reductionAxis)
        case 'variance_axis': return variance(a, reductionAxis)
        case 'std_axis': return std(a, reductionAxis)
        case 'reshape': return id.endsWith('/materialized') ? reshape(transpose(a, [1, 0]), [shape[0], shape[1]]) : reshape(a, [shape[1], shape[0]])
        case 'transpose': return transpose(a, [1, 0])
        case 'permute': return permute(a, [1, 0])
        case 'slice': return slice(a, [{ start: 0, stop: Math.floor(shape[0] / 2) }, { start: 0, stop: Math.floor(shape[1] / 2) }])
        case 'contiguous': return id.endsWith('/view_only') ? contiguous(a) : contiguous(transpose(a, [1, 0]))
        case 'cat': return cat([a, b], 1)
        case 'stack': return stack([a, b], 0)
        case 'squeeze': return squeeze(a, 0)
        case 'unsqueeze': return unsqueeze(a, 0)
        case 'embedding': return embedding(a, b)
        case 'index_select': return index_select(a, indexAxis, b)
        case 'gather': return gather(a, indexAxis, b)
        case 'where': return where(condition!, a, b)
        case 'masked_fill': return masked_fill(a, condition!, -2)
        case 'cross_entropy_indexed': return cross_entropy(a, b)
        case 'one_hot': return one_hot(a, classes)
        case 'matmul': return matmul(a, b)
        case 'dot': return dot(a, b)
        case 'argmin_all': return argmin(a)
        case 'argmax_all': return argmax(a)
        case 'argmin_axis': return argmin(a, reductionAxis)
        case 'argmax_axis': return argmax(a, reductionAxis)
        case 'sum_all': return sum(a)
        case 'mean_all': return mean(id.includes('/rank1/') ? reshape(a, [shape[1]]) : a)
        case 'min_all': return min(a)
        case 'max_all': return max(a)
        case 'variance_all': return variance(a)
        case 'std_all': return std(a)
        case 'sum_axis': return sum(a, reductionAxis)
        default: return softmax(a, reductionAxis)
      }
    })
    const executable = session.compile(source, { explanationLevel: 'detailed',
      denseInputs: getEnv('AFFON_PROBE_DENSE_INPUTS') === '1' })
    try {
      const result = executable.run(mask ? { x, y, mask } : { x, y })
      try {
        const raw = result.to_array() as any
        const flat = (Array.isArray(raw) ? raw.flat(Infinity) : [raw]) as number[]
        if (!flat.length || flat.some(value => !Number.isFinite(value))) throw Error('nonfinite or empty output')
        const indices = Array.from(new Set([0, Math.floor(flat.length / 4), Math.floor(flat.length / 2), Math.floor(flat.length * 3 / 4), flat.length - 1]))
        console.log(JSON.stringify({ schema_version: 1, suite: 'affon_capability_probe', id, shape, input_shapes: mask ? [inputShape, rhsShape, maskShape] : [inputShape, rhsShape],
          output_shape: result.shape, output_dtype: result.dtype, output_count: flat.length, output_checksum: flat.reduce((total, value) => total + value, 0),
          output_samples: indices.map(index => ({ index, value: flat[index] })),
          explanation: executable.explain() }))
      } finally { result.dispose() }
    } finally { executable.dispose() }
  } finally { mask?.dispose(); y.dispose(); x.dispose() }
} finally { session.dispose() }
