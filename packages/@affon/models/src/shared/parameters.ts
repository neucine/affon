import { Tensor, type FormalTensor, type ProgramDType, type TensorData } from 'affon:compute'
import { cast } from 'affon:ops'

/** Minimal checkpoint/runtime value contract accepted at model boundaries. */
export interface ModelTensor {
  readonly shape: readonly number[]
  readonly dtype: ProgramDType
  to_array(): TensorData
}

export interface ModelAffineWeights { weight: ModelTensor; bias: ModelTensor }

/** Keep BF16 parameters in native storage while computing legacy model graphs in f32. */
export function f32_parameter(p: any, name: string, shape: readonly number[], value: ModelTensor): FormalTensor {
  const raw = p.parameter(name, value.dtype === 'bf16' ? Tensor.bf16(shape) : Tensor.f32(shape)) as FormalTensor
  return value.dtype === 'bf16' ? cast(raw, 'f32') : raw
}

export function positive_dimensions(...values: number[]) {
  if (values.some(value => !Number.isInteger(value) || value <= 0)) throw new Error('Expected positive integer model dimensions')
}

export function parameter_checks() {
  const tensor = (value: ModelTensor, shape: number[]) => {
    if (!value || (value.dtype !== 'f32' && value.dtype !== 'bf16') || JSON.stringify(value.shape) !== JSON.stringify(shape)) {
      throw new Error(`Expected f32 or BF16 model tensor with shape ${JSON.stringify(shape)}`)
    }
  }
  const affine = (value: ModelAffineWeights, shape: number[]) => {
    if (!value) throw new Error('Missing affine model parameters')
    tensor(value.weight, shape)
    tensor(value.bias, [shape[shape.length - 1]])
  }
  return { tensor, affine }
}
