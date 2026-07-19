import { describe, expect, test } from 'std:test'
import {
  add,
  axes,
  matmul,
  masked_fill,
  mul,
  parameter,
  permute,
  reshape,
  softmax,
  sum,
  tensor,
} from 'affon:compute'

function maxAbs(value: unknown): number {
  if (Array.isArray(value)) {
    return value.reduce((max, entry) => Math.max(max, maxAbs(entry)), 0)
  }
  return Math.abs(value as number)
}

describe('compute reshape backward', () => {
  test('preserves logical order for non-contiguous gradient inputs', () => {
    setDevice('cpu')

    const batch = 1
    const seq = 3
    const dModel = 4
    const numHeads = 2
    const headDim = 2
    const qProj = tensor([[
      [0.2, -0.4, 0.1, 0.3],
      [-0.5, 0.7, 0.6, 0.1],
      [-0.2, 0.5, 0.3, -0.8],
    ]], { dtype: 'f32' })
    const kLinear = tensor([[
      [0.6, 0.1, -0.2, 0.5],
      [0.3, -0.8, 0.2, -0.4],
      [0.1, 0.3, -0.5, 0.7],
    ]], { dtype: 'f32' })
    const keyBias = parameter([1, dModel], { dtype: 'f32' })
    const upstream = tensor([[
      [[0.1, -0.2, 0.3], [0.4, -0.5, 0.6], [-0.7, 0.8, -0.9]],
      [[-0.3, 0.2, 0.1], [0.6, 0.4, -0.5], [0.9, -0.8, 0.7]],
    ]], { dtype: 'f32' })
    const mask = tensor([
      [0, 1, 1],
      [0, 0, 1],
      [0, 0, 0],
    ], { dtype: 'f32', axes: [axes.token, axes.token] })

    const q = permute(reshape(qProj, [batch, seq, numHeads, headDim]), [0, 2, 1, 3])
    const k = permute(reshape(add(kLinear, keyBias), [batch, seq, numHeads, headDim]), [0, 2, 1, 3])
    const scores = matmul(q, permute(k, [0, 1, 3, 2]))
    const weights = softmax(masked_fill(scores, mask, -1e9), 3)
    sum(mul(weights, upstream)).backward()

    expect(maxAbs(keyBias.grad!.to_array()) < 1e-6).toBeTruthy()
  })
})
