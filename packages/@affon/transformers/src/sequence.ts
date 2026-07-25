import { axes, masked_fill, tensor } from 'affon:compute'
import type { Device, DType, Tensor } from 'affon:compute'

type FloatDType = Extract<DType, 'f32' | 'f64'>

export interface SequenceTensorOptions<D extends FloatDType = 'f32'> {
  dtype?: D
  device?: Device
}

export interface SinusoidalEncodingOptions<D extends FloatDType = 'f32'> {
  dtype?: D
  device?: Device
}

function graph_safe_device(value: { device: Device }): Device | undefined {
  try {
    return value.device
  } catch {
    return undefined
  }
}

export function causal_mask<D extends FloatDType = 'f32'>(
  length: number,
  opts?: SequenceTensorOptions<D>,
): Tensor<[number, number], D> {
  if (!Number.isInteger(length) || length <= 0) {
    throw new AffonError('invalid_arg', 'causal_mask length must be a positive integer')
  }
  const rows: number[][] = []
  for (let row = 0; row < length; row++) {
    const cols: number[] = []
    for (let col = 0; col < length; col++) cols.push(col > row ? 1 : 0)
    rows.push(cols)
  }
  return tensor(rows, { dtype: opts?.dtype ?? 'f32' as D, device: opts?.device, axes: [axes.token, axes.token] }) as Tensor<[number, number], D>
}

export function apply_causal_mask<
  S extends readonly number[] = number[],
  D extends FloatDType = 'f32',
>(
  scores: Tensor<S, D>,
  value?: number,
): Tensor<number[], D> {
  if (scores.ndim < 2) {
    throw new AffonError('invalid_shape', 'apply_causal_mask expects a tensor with at least 2 dimensions')
  }
  const seqLen = scores.shape[scores.ndim - 1] as number
  const device = graph_safe_device(scores)
  const mask = causal_mask(seqLen, device
    ? { dtype: scores.dtype as D, device }
    : { dtype: scores.dtype as D })
	  return masked_fill(scores as unknown as Tensor<number[], D>, mask, value ?? -1e9) as Tensor<number[], D>
}

export function sinusoidal_encoding<D extends FloatDType = 'f32'>(
  length: number,
  dim: number,
  opts?: SinusoidalEncodingOptions<D>,
): Tensor<[number, number], D> {
  if (!Number.isInteger(length) || length <= 0) {
    throw new AffonError('invalid_arg', 'sinusoidal_encoding length must be a positive integer')
  }
  if (!Number.isInteger(dim) || dim <= 0) {
    throw new AffonError('invalid_arg', 'sinusoidal_encoding dim must be a positive integer')
  }
  const rows: number[][] = []
  for (let pos = 0; pos < length; pos++) {
    const row: number[] = []
    for (let i = 0; i < dim; i++) {
      const exponent = (2 * Math.floor(i / 2)) / dim
      const angle = pos / Math.pow(10000, exponent)
      row.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle))
    }
    rows.push(row)
  }
  return tensor(rows, { dtype: opts?.dtype ?? 'f32' as D, device: opts?.device, axes: [axes.token, axes.feature] }) as Tensor<[number, number], D>
}

export function position_ids(length: number): Tensor<[number], 'f32'> {
  if (!Number.isInteger(length) || length <= 0) {
    throw new AffonError('invalid_arg', 'position_ids length must be a positive integer')
  }
  const ids: number[] = []
  for (let i = 0; i < length; i++) ids.push(i)
  return tensor(ids, { dtype: 'f32', axes: [axes.token] }) as Tensor<[number], 'f32'>
}
