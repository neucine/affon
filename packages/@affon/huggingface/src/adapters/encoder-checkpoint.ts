import { open_checkpoint } from './checkpoint.ts'
import { contiguous, transpose } from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'

export function prepare_encoder_checkpoint(directory: string, device: Device) {
  const source = open_checkpoint(directory)
  const weights: Record<string, Tensor> = Object.create(null)
  const expected = new Set<string>()
  function require_weight(name: string, shape: number[], linear = false) {
    const info = source.catalog[name]
    if (!info || !['F32', 'BF16'].includes(info.dtype) || JSON.stringify(info.shape) !== JSON.stringify(shape)) {
      throw new Error(`Expected f32 ${name} with shape ${JSON.stringify(shape)}`)
    }
    const value = source.read(name)
    expected.add(name)
    const placed = value.device === device ? value : value.to(device)
    weights[name] = linear ? contiguous(transpose(placed, 0, 1)) : placed
  }
  function finish() {
    for (const name of Object.keys(source.catalog)) {
      if (!expected.has(name)) throw new Error(`Unexpected checkpoint tensor: ${name}`)
    }
  }
  const affine = (prefix: string) => ({ weight: weights[`${prefix}.weight`], bias: weights[`${prefix}.bias`] })
  return { weights, require_weight, finish, affine }
}
