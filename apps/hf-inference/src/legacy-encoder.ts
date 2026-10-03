import type { Device } from 'affon:compute'

export interface AuditTensor {
  readonly shape: readonly number[]
  readonly device: Device
  to_array(): unknown
}
export interface AffineWeights { weight: AuditTensor; bias: AuditTensor }

const flatten = (value: unknown): number[] => (value as any[]).flat(Infinity).map(Number)
const size = (shape: readonly number[]) => shape.reduce((product, value) => product * value, 1)
function nested(values: readonly number[], shape: readonly number[]): unknown {
  if (shape.length === 0) return values[0]
  if (shape.length === 1) return values.slice(0, shape[0])
  const stride = size(shape.slice(1))
  return Array.from({ length: shape[0] }, (_, index) => nested(values.slice(index * stride, (index + 1) * stride), shape.slice(1)))
}
function result(values: readonly number[], shape: readonly number[], device: Device): AuditTensor {
  return { shape: [...shape], device, to_array: () => nested(values, shape) }
}

/** Host reference operators used only by parity-audit scripts. */
export function encoder_ops(device: Device) {
  const scalar = (value: number) => result([value], [], device)
  const dense = (input: AuditTensor, params: AffineWeights) => {
    const x = flatten(input.to_array()), weight = flatten(params.weight.to_array()), bias = flatten(params.bias.to_array())
    const width = input.shape.at(-1)!, output = params.weight.shape.at(-1)!
    const rows = x.length / width, values = Array(rows * output).fill(0)
    for (let row = 0; row < rows; row++) for (let column = 0; column < output; column++) {
      let value = bias[column]
      for (let inner = 0; inner < width; inner++) value += x[row * width + inner] * weight[inner * output + column]
      values[row * output + column] = value
    }
    return result(values, [...input.shape.slice(0, -1), output], device)
  }
  const norm = (input: AuditTensor, params: AffineWeights, eps: number) => {
    const x = flatten(input.to_array()), weight = flatten(params.weight.to_array()), bias = flatten(params.bias.to_array())
    const width = input.shape.at(-1)!, values = Array(x.length)
    for (let start = 0; start < x.length; start += width) {
      const mean = x.slice(start, start + width).reduce((sum, value) => sum + value, 0) / width
      const variance = x.slice(start, start + width).reduce((sum, value) => sum + (value - mean) ** 2, 0) / width
      for (let index = 0; index < width; index++) values[start + index] = (x[start + index] - mean) / Math.sqrt(variance + eps) * weight[index] + bias[index]
    }
    return result(values, input.shape, device)
  }
  const attention = (q: AuditTensor, k: AuditTensor, v: AuditTensor, heads: number, mask?: AuditTensor) => {
    const [batch, length, dimension] = q.shape, width = dimension / heads
    const qv = flatten(q.to_array()), kv = flatten(k.to_array()), vv = flatten(v.to_array()), maskValues = mask ? flatten(mask.to_array()) : null
    const output = Array(batch * length * dimension).fill(0)
    const at = (values: number[], b: number, token: number, head: number, inner: number) => values[(b * length + token) * dimension + head * width + inner]
    for (let b = 0; b < batch; b++) for (let h = 0; h < heads; h++) for (let row = 0; row < length; row++) {
      const scores = Array(length).fill(0)
      for (let column = 0; column < length; column++) {
        for (let inner = 0; inner < width; inner++) scores[column] += at(qv, b, row, h, inner) * at(kv, b, column, h, inner)
        scores[column] /= Math.sqrt(width)
        if (maskValues?.[row * length + column]) scores[column] = -Infinity
      }
      const maximum = Math.max(...scores), weights = scores.map(value => Math.exp(value - maximum)), total = weights.reduce((sum, value) => sum + value, 0)
      for (let inner = 0; inner < width; inner++) for (let column = 0; column < length; column++) output[(b * length + row) * dimension + h * width + inner] += weights[column] / total * at(vv, b, column, h, inner)
    }
    return result(output, q.shape, device)
  }
  return { scalar, dense, norm, attention }
}

export function erf_gelu(input: AuditTensor): AuditTensor {
  const values = flatten(input.to_array()).map(x => {
    const z = x / Math.sqrt(2), t = 1 / (1 + 0.3275911 * Math.abs(z))
    let polynomial = 1.061405429 * t - 1.453152027
    polynomial = polynomial * t + 1.421413741
    polynomial = polynomial * t - 0.284496736
    polynomial = polynomial * t + 0.254829592
    const erf = Math.sign(z) * (1 - polynomial * t * Math.exp(-(z * z)))
    return 0.5 * x * (1 + erf)
  })
  return result(values, input.shape, input.device)
}
