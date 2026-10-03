import { describe, test, expect, values } from 'std:test'
import { copy, empty, parameter, seed, tensor } from 'affon:compute/legacy'
import nn from 'affon:nn/legacy'
import { captureError } from '../../support/errors.ts'

function flatten(value: any): number[] {
  if (Array.isArray(value)) return value.flatMap(flatten)
  return [value]
}

describe('nn init', () => {
  test('zeros and ones fill tensors in place', () => {
    const zeros = copy(parameter([2, 2], { dtype: 'f32' }), tensor([[5, 6], [7, 8]], { dtype: 'f32' }))
    nn.init.zeros(zeros)
    expect(values(zeros)).toEqual([[0, 0], [0, 0]])

    const ones = copy(parameter([2, 2], { dtype: 'f32' }), tensor([[0, 0], [0, 0]], { dtype: 'f32' }))
    nn.init.ones(ones)
    expect(values(ones)).toEqual([[1, 1], [1, 1]])
  })

  test('empty tensors preserve shape and remain writable by init', () => {
    const out = empty([2, 3])
    expect(out.shape).toEqual([2, 3])
    nn.init.zeros(out)
    expect(values(out)).toEqual([[0, 0, 0], [0, 0, 0]])
  })

  test('xavier_uniform and kaiming_uniform stay within expected bounds', () => {
    const xu = copy(parameter([2, 3], { dtype: 'f32' }), tensor([[0, 0, 0], [0, 0, 0]], { dtype: 'f32' }))
    nn.init.xavier_uniform(xu)
    const xuBound = Math.sqrt(6 / (2 + 3))
    for (const value of flatten(values(xu))) {
      expect(value >= -xuBound && value <= xuBound).toBe(true)
    }

    const ku = copy(parameter([2, 3], { dtype: 'f32' }), tensor([[0, 0, 0], [0, 0, 0]], { dtype: 'f32' }))
    nn.init.kaiming_uniform(ku)
    const kuBound = Math.sqrt(6 / 2)
    for (const value of flatten(values(ku))) {
      expect(value >= -kuBound && value <= kuBound).toBe(true)
    }
  })

  test('normal initializers write finite values and preserve dtype', () => {
    seed(11)

    const xn = copy(parameter([4, 3], { dtype: 'f32' }), tensor([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ], { dtype: 'f32' }))
    const kn = copy(parameter([4, 3], { dtype: 'f32' }), tensor([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ], { dtype: 'f32' }))

    nn.init.xavier_normal(xn)
    nn.init.kaiming_normal(kn)

    const xVals = flatten(values(xn))
    const kVals = flatten(values(kn))
    expect(xn.dtype).toBe('f32')
    expect(kn.dtype).toBe('f32')
    expect(xVals).toBeAllFinite()
    expect(kVals).toBeAllFinite()
    expect(xVals.some((value) => value !== 0)).toBe(true)
    expect(kVals.some((value) => value !== 0)).toBe(true)
  })

  test('fan-based initializers reject rank-1 tensors and handle higher-rank weights', () => {
    const rankOne = parameter([3], { dtype: 'f32' })
    const err = captureError(() => nn.init.xavier_uniform(rankOne))
    expect(err).toBeInstanceOf(AffonError)
    expect(err.code).toBe('invalid_shape')
    expect(err.message).toContain('at least 2 dimensions')

    const convLike = copy(parameter([2, 3, 2], { dtype: 'f32' }), tensor([
      [[0, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0]],
    ], { dtype: 'f32' }))
    nn.init.kaiming_uniform(convLike)
    const bound = Math.sqrt(6 / (3 * 2))
    for (const value of flatten(values(convLike))) {
      expect(value >= -bound && value <= bound).toBe(true)
    }
  })

  test('Linear uses kaiming_uniform weights and zero bias by default', () => {
    const linear = nn.Linear(2, 3)
    const kuBound = Math.sqrt(6 / 2)

    expect(values(linear.bias)).toEqual([[0, 0, 0]])
    for (const value of flatten(values(linear.weight))) {
      expect(value >= -kuBound && value <= kuBound).toBe(true)
    }
  })
})
