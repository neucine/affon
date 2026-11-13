import { open_checkpoint } from './checkpoint.ts'
import type { Tensor } from 'affon:core'
import type { ModelTensor } from '../../../models/src/shared/parameters.ts'

export function prepare_encoder_checkpoint(directory: string) {
  const source = open_checkpoint(directory)
  const weights: Record<string, ModelTensor> = Object.create(null)
  const expected = new Set<string>()
  function require_weight(name: string, shape: number[], linear = false) {
    const info = source.catalog[name]
    if (!info || !['F32', 'BF16'].includes(info.dtype) || JSON.stringify(info.shape) !== JSON.stringify(shape)) {
      throw new Error(`Expected f32 or BF16 ${name} with shape ${JSON.stringify(shape)}`)
    }
    const value = source.read(name)
    expected.add(name)
    if (!linear) {
      weights[name] = value
      return
    }
    weights[name] = { shape: [shape[1], shape[0]], dtype: value.dtype,
      to_bytes: () => {
        const source = new Uint8Array((value as Tensor).to_bytes())
        const size = value.dtype === 'bf16' ? 2 : 4
        const result = new Uint8Array(source.length)
        for (let row = 0; row < shape[0]; row++) for (let column = 0; column < shape[1]; column++) {
          const from = (row * shape[1] + column) * size
          result.set(source.subarray(from, from + size), (column * shape[0] + row) * size)
        }
        return result.buffer
      },
      to_array: () => {
        const rows = value.to_array() as number[][]
        return Array.from({ length: shape[1] }, (_, row) => Array.from({ length: shape[0] }, (_, column) => rows[column][row]))
      },
    }
  }
  function finish() {
    for (const name of Object.keys(source.catalog)) {
      if (!expected.has(name)) throw new Error(`Unexpected checkpoint tensor: ${name}`)
    }
  }
  const affine = (prefix: string) => ({ weight: weights[`${prefix}.weight`], bias: weights[`${prefix}.bias`] })
  return { weights, require_weight, finish, affine }
}
