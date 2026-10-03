import { open_checkpoint } from './checkpoint.ts'
import type { ModelTensor } from '../../../models/src/shared/parameters.ts'

export function prepare_encoder_checkpoint(directory: string) {
  const source = open_checkpoint(directory)
  const weights: Record<string, ModelTensor> = Object.create(null)
  const expected = new Set<string>()
  function require_weight(name: string, shape: number[], linear = false) {
    const info = source.catalog[name]
    if (!info || !['F32', 'BF16'].includes(info.dtype) || JSON.stringify(info.shape) !== JSON.stringify(shape)) {
      throw new Error(`Expected f32 ${name} with shape ${JSON.stringify(shape)}`)
    }
    const value = source.read(name)
    expected.add(name)
    if (!linear) {
      weights[name] = value
      return
    }
    const rows = value.to_array() as number[][]
    const transposed = Array.from({ length: shape[1] }, (_, row) => Array.from({ length: shape[0] }, (_, column) => rows[column][row]))
    weights[name] = { shape: [shape[1], shape[0]], dtype: 'f32', to_array: () => transposed }
  }
  function finish() {
    for (const name of Object.keys(source.catalog)) {
      if (!expected.has(name)) throw new Error(`Unexpected checkpoint tensor: ${name}`)
    }
  }
  const affine = (prefix: string) => ({ weight: weights[`${prefix}.weight`], bias: weights[`${prefix}.bias`] })
  return { weights, require_weight, finish, affine }
}
