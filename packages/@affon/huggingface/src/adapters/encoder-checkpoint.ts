import checkpoint from 'affon:checkpoint'
import { contiguous, transpose } from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'

export function prepare_encoder_checkpoint(directory: string, device: Device) {
  const weights = checkpoint.load(`${directory}/model.safetensors`) as Record<string, Tensor>
  const expected = new Set<string>()
  function require_weight(name: string, shape: number[], linear = false) {
    const value = weights[name]
    if (!value || value.dtype !== 'f32' || JSON.stringify(value.shape) !== JSON.stringify(shape)) {
      throw new Error(`Expected f32 ${name} with shape ${JSON.stringify(shape)}`)
    }
    expected.add(name)
    const placed = value.device === device ? value : value.to(device)
    weights[name] = linear ? contiguous(transpose(placed, 0, 1)) : placed
  }
  function finish() {
    for (const name of Object.keys(weights)) {
      if (!expected.has(name)) throw new Error(`Unexpected checkpoint tensor: ${name}`)
    }
  }
  const affine = (prefix: string) => ({ weight: weights[`${prefix}.weight`], bias: weights[`${prefix}.bias`] })
  return { weights, require_weight, finish, affine }
}
