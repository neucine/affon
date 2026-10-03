import type { Device, ProgramDType, TensorData } from 'affon:compute'

/** Minimal checkpoint/runtime value contract accepted at model boundaries. */
export interface ModelTensor {
  readonly shape: readonly number[]
  readonly dtype: ProgramDType
  readonly device: Device
  to_array(): TensorData
}

export interface ModelAffineWeights { weight: ModelTensor; bias: ModelTensor }

export function positive_dimensions(...values: number[]) {
  if (values.some(value => !Number.isInteger(value) || value <= 0)) throw new Error('Expected positive integer model dimensions')
}

export function parameter_checks(device: Device) {
  const tensor = (value: ModelTensor, shape: number[]) => {
    if (!value || value.dtype !== 'f32' || JSON.stringify(value.shape) !== JSON.stringify(shape)) {
      throw new Error(`Expected f32 model tensor loadable by ${device} with shape ${JSON.stringify(shape)}`)
    }
  }
  const affine = (value: ModelAffineWeights, shape: number[]) => {
    if (!value) throw new Error('Missing affine model parameters')
    tensor(value.weight, shape)
    tensor(value.bias, [shape[shape.length - 1]])
  }
  return { tensor, affine }
}
