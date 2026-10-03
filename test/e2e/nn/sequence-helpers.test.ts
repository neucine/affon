import { describe, test, expect, values } from 'std:test'
import { tensor } from 'affon:compute/legacy'
import nn from 'affon:nn/legacy'
import { captureError } from '../../support/errors.ts'

describe('nn sequence helpers', () => {
  test('causal_mask returns an upper-triangular visibility mask and caches by placement', () => {
    const mask = nn.causal_mask(4, { dtype: 'f32' })

    expect(mask.shape).toEqual([4, 4])
    expect(mask.dtype).toBe('f32')
    expect(values(mask)).toEqual([
      [0, 1, 1, 1],
      [0, 0, 1, 1],
      [0, 0, 0, 1],
      [0, 0, 0, 0],
    ])
    expect(nn.causal_mask(4, { dtype: 'f32' }) === mask).toBe(true)
    expect(nn.causal_mask(4, { dtype: 'f64' }) === mask).toBe(false)
  })

  test('apply_causal_mask fills future positions across leading dimensions', () => {
    const scores = tensor([
      [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],
      [
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
      ],
    ], { dtype: 'f32' })

    const masked = nn.apply_causal_mask(scores, -99)

    expect(masked.shape).toEqual([2, 3, 3])
    expect(values(masked)).toEqual([
      [
        [1, -99, -99],
        [4, 5, -99],
        [7, 8, 9],
      ],
      [
        [10, -99, -99],
        [13, 14, -99],
        [16, 17, 18],
      ],
    ])
  })

  test('sinusoidal_encoding and position_ids expose deterministic table values', () => {
    const encoding = nn.sinusoidal_encoding(2, 4, { dtype: 'f32' })
    const expected = [
      [0, 1, 0, 1],
      [Math.sin(1), Math.cos(1), Math.sin(0.01), Math.cos(0.01)],
    ]

    expect(encoding.shape).toEqual([2, 4])
    expect(encoding.dtype).toBe('f32')
    expect(values(encoding)).toBeAllClose(expected, { rtol: 1e-6, atol: 1e-6 })
    expect(values(nn.position_ids(4))).toEqual([0, 1, 2, 3])
  })

  test('sequence helpers reject invalid shapes and lengths', () => {
    for (const fn of [
      () => nn.causal_mask(0),
      () => nn.position_ids(0),
      () => nn.sinusoidal_encoding(0, 4),
      () => nn.sinusoidal_encoding(4, 0),
    ]) {
      const err = captureError(fn)
      expect(err).toBeInstanceOf(AffonError)
      expect(err.code).toBe('invalid_arg')
    }

    const maskErr = captureError(() => nn.apply_causal_mask(tensor([1, 2, 3])))
    expect(maskErr).toBeInstanceOf(AffonError)
    expect(maskErr.code).toBe('invalid_shape')
    expect(maskErr.message).toContain('at least 2 dimensions')
  })
})
