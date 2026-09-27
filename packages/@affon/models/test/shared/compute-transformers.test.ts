import { describe, expect, test, values } from 'std:test'
import { axes, tensor } from 'affon:compute'
import type { Tensor } from 'affon:compute'

import {
  apply_causal_mask,
  DecoderInputEmbedding,
} from '../../src/index.ts'

describe('@affon/models compute', () => {
  test('masking and decoder layers work with compute-native values', () => {
    const masked = apply_causal_mask(tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ], { dtype: 'f32' }))
    expect(values(masked)).toEqual([
      [1, -1000000000, -1000000000],
      [4, 5, -1000000000],
      [7, 8, 9],
    ])

    const embed = DecoderInputEmbedding(32, 6, { positional: 'learned', maxSeqLen: 16 })
    const tokenIds = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ], { dtype: 'f32', axes: [axes.batch, axes.token] })
    const embedded = embed(tokenIds as Tensor<[number, number], 'f32'>)
    expect(embedded.shape).toEqual([2, 3, 6])
  })
})
