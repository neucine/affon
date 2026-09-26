export type StaticTensorMeta = {
  shape: number[]
  dtype: 'f32' | 'f64' | 'i64'
}

function broadcastShapeRightAligned(a: readonly number[], b: readonly number[]): number[] | null {
  const rank = Math.max(a.length, b.length)
  const out = new Array<number>(rank)
  for (let i = 0; i < rank; i++) {
    const left = i < rank - a.length ? 1 : a[i - (rank - a.length)]
    const right = i < rank - b.length ? 1 : b[i - (rank - b.length)]
    if (left === right) {
      out[i] = left
      continue
    }
    if (left === 1) {
      out[i] = right
      continue
    }
    if (right === 1) {
      out[i] = left
      continue
    }
    return null
  }
  return out
}

function inferMatmulShape(a: readonly number[], b: readonly number[]): number[] | null {
  if (a.length === 0 || b.length === 0) return null
  if (a.length === 1 && b.length === 1) {
    return a[0] === b[0] ? [] : null
  }

  const leftContract = a[a.length - 1]
  const rightContract = b.length === 1 ? b[0] : b[b.length - 2]
  if (leftContract !== rightContract) return null

  const leftBatch = a.length <= 2 ? [] : a.slice(0, -2)
  const rightBatch = b.length <= 2 ? [] : b.slice(0, -2)
  const batch = broadcastShapeRightAligned(leftBatch, rightBatch)
  if (!batch) return null

  if (a.length === 1) {
    return [...batch, b[b.length - 1]!]
  }
  if (b.length === 1) {
    return [...batch, a[a.length - 2]!]
  }
  return [...batch, a[a.length - 2]!, b[b.length - 1]!]
}

function reduceShape(shape: readonly number[], axis?: number, keepdim?: boolean): number[] | null {
  if (axis == null) return [1]
  if (!Number.isInteger(axis) || axis < 0 || axis >= shape.length) return null
  if (keepdim) {
    const out = [...shape]
    out[axis] = 1
    return out
  }
  const out = shape.filter((_, index) => index !== axis)
  return out.length === 0 ? [1] : out
}

export function sameStaticMeta(left: StaticTensorMeta, right: StaticTensorMeta): boolean {
  return left.dtype === right.dtype
    && left.shape.length === right.shape.length
    && left.shape.every((dim, index) => dim === right.shape[index])
}

export function normalizeCoreMeta(value: unknown): StaticTensorMeta | null {
  if (!value || typeof value !== 'object') return null
  const meta = value as { shape?: unknown; dtype?: unknown }
  if (!Array.isArray(meta.shape)) return null
  if (meta.dtype !== 'f32' && meta.dtype !== 'f64' && meta.dtype !== 'i64') return null
  const shape = meta.shape.map((dim) => Number(dim))
  if (!shape.every((dim) => Number.isInteger(dim) && dim >= 0)) return null
  return { shape, dtype: meta.dtype }
}

export function shadowCapturedMeta(kind: string, inputs: readonly StaticTensorMeta[], options: any): StaticTensorMeta | undefined {
  const first = inputs[0]
  const second = inputs[1]
  switch (kind) {
    case 'neg':
    case 'relu':
    case 'abs':
    case 'exp':
    case 'log':
    case 'sqrt':
    case 'sigmoid':
    case 'silu':
    case 'erf':
    case 'tanh':
    case 'sign':
    case 'gelu':
    case 'clamp':
    case 'contiguous':
    case 'softmax':
      return first ? { shape: [...first.shape], dtype: first.dtype } : undefined
    case 'cast':
      return first && options?.dtype ? { shape: [...first.shape], dtype: options.dtype } : undefined
    case 'reshape':
      return first && Array.isArray(options?.shape) ? { shape: [...options.shape], dtype: first.dtype } : undefined
    case 'squeeze': {
      if (!first) return undefined
      const axis = options?.axis
      let shape: number[]
      if (axis == null) {
        shape = first.shape.filter((dim) => dim !== 1)
      } else if (axis >= 0 && axis < first.shape.length && first.shape[axis] === 1) {
        shape = first.shape.filter((_, index) => index !== axis)
      } else {
        shape = [...first.shape]
      }
      return { shape: shape.length === 0 ? [1] : shape, dtype: first.dtype }
    }
    case 'unsqueeze': {
      const axis = options?.axis
      if (!first || !Number.isInteger(axis) || axis < 0 || axis > first.shape.length) return undefined
      const shape = first.shape.slice()
      shape.splice(axis, 0, 1)
      return { shape, dtype: first.dtype }
    }
    case 'transpose':
      return first && first.shape.length === 2 ? { shape: [first.shape[1]!, first.shape[0]!], dtype: first.dtype } : undefined
    case 'permute':
      return first && Array.isArray(options?.axes) && options.axes.length === first.shape.length
        ? { shape: options.axes.map((axis: number) => first.shape[axis]!), dtype: first.dtype }
        : undefined
    case 'dot':
      return first && second ? { shape: [], dtype: first.dtype } : undefined
    case 'matmul': {
      if (!first || !second) return undefined
      const shape = inferMatmulShape(first.shape, second.shape)
      return shape ? { shape, dtype: first.dtype } : undefined
    }
    case 'where': {
      const third = inputs[2]
      if (!first || !second || !third) return undefined
      const condTrue = broadcastShapeRightAligned(first.shape, second.shape)
      const shape = condTrue ? broadcastShapeRightAligned(condTrue, third.shape) : null
      return shape ? { shape, dtype: second.dtype } : undefined
    }
    case 'masked_fill': {
      if (!first || !second) return undefined
      const shape = broadcastShapeRightAligned(first.shape, second.shape)
      return shape ? { shape, dtype: first.dtype } : undefined
    }
    case 'cross_entropy_indexed':
      return { shape: [1], dtype: 'f32' }
    case 'one_hot':
      return first && Number.isInteger(options?.numClasses) ? { shape: [...first.shape, options.numClasses], dtype: 'f32' } : undefined
    case 'index_select': {
      const dim = options?.dim
      if (!first || !second || !Number.isInteger(dim) || dim < 0 || dim >= first.shape.length) return undefined
      return { shape: first.shape.slice(0, dim).concat(second.shape, first.shape.slice(dim + 1)), dtype: first.dtype }
    }
    case 'gather':
      return first && second ? { shape: [...second.shape], dtype: first.dtype } : undefined
    case 'add':
    case 'sub':
    case 'mul':
    case 'div':
    case 'gt': {
      if (!first || !second) return undefined
      const shape = broadcastShapeRightAligned(first.shape, second.shape)
      return shape ? { shape, dtype: kind === 'gt' ? 'i64' : first.dtype } : undefined
    }
    case 'sum':
    case 'mean':
    case 'std':
    case 'variance':
    case 'min':
    case 'max': {
      if (!first) return undefined
      const shape = reduceShape(first.shape, options?.axis, options?.keepdim)
      return shape ? { shape, dtype: first.dtype } : undefined
    }
    case 'argmin':
    case 'argmax': {
      if (!first) return undefined
      const shape = reduceShape(first.shape, options?.axis, options?.keepdim)
      return shape ? { shape, dtype: 'i64' } : undefined
    }
    default:
      return undefined
  }
}
