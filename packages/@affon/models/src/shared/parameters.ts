import type { Device, Tensor } from 'affon:compute'
import type { AffineWeights } from './encoder.ts'

export function positive_dimensions(...values: number[]) {
  if (values.some(value => !Number.isInteger(value) || value <= 0)) throw new Error('Expected positive integer model dimensions')
}

export function parameter_checks(device: Device) {
  const tensor = (value: Tensor, shape: number[]) => {
    if (!value || value.dtype !== 'f32' || value.device !== device || JSON.stringify(value.shape) !== JSON.stringify(shape)) {
      throw new Error(`Expected f32 model tensor on ${device} with shape ${JSON.stringify(shape)}`)
    }
  }
  const affine = (value: AffineWeights, shape: number[]) => {
    if (!value) throw new Error('Missing affine model parameters')
    tensor(value.weight, shape)
    tensor(value.bias, [shape[shape.length - 1]])
  }
  return { tensor, affine }
}
