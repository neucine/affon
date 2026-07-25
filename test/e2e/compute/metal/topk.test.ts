import { describe, test, expect, values } from 'std:test'
import { sum, tensor, topk } from 'affon:compute'
import { internal_tensor } from '../../../support/compute.ts'

describe('compute metal topk', () => {
  test('runs 2d axis-1 topk on metal and keeps outputs on metal', () => {
    const input = tensor([[10, 30, 20], [60, 40, 50]], { dtype: 'f32' }).to('metal')

    const result = topk(input, 2, 1)

    expect(result.values.device).toBe('metal')
    expect(result.indices.device).toBe('metal')
    expect(values(result.values.to('cpu'))).toEqual([[30, 20], [60, 50]])
    expect(values(result.indices.to('cpu'))).toEqual([[1, 2], [0, 2]])
  })

  test('runs 1d topk on metal and keeps outputs on metal', () => {
    const input = tensor([10, 30, 20, 40], { dtype: 'f32' }).to('metal')

    const result = topk(input, 3, 0)

    expect(result.values.device).toBe('metal')
    expect(result.indices.device).toBe('metal')
    expect(values(result.values.to('cpu'))).toEqual([40, 30, 20])
    expect(values(result.indices.to('cpu'))).toEqual([3, 1, 2])
  })

  test('runs 3d topk on a non-last axis on metal and keeps outputs on metal', () => {
    const input = tensor([
      [[1, 10], [5, 8], [3, 6]],
      [[9, 2], [4, 7], [8, 0]],
    ], { dtype: 'f32' }).to('metal')

    const result = topk(input, 2, 1)

    expect(result.values.device).toBe('metal')
    expect(result.indices.device).toBe('metal')
    expect(values(result.values.to('cpu'))).toEqual([
      [[5, 10], [3, 8]],
      [[9, 7], [8, 2]],
    ])
    expect(values(result.indices.to('cpu'))).toEqual([
      [[1, 0], [2, 1]],
      [[0, 1], [2, 0]],
    ])
  })

  test('keeps topk backward on metal', () => {
    const input = internal_tensor([[10, 30, 20], [60, 40, 50]], { dtype: 'f32', device: 'metal' })

    const result = topk(input, 2, 1)
    sum(result.values).backward()

    expect(input.grad_device).toBe('metal')
    expect(values(input.grad?.to('cpu'))).toEqual([
      [0, 1, 1],
      [1, 0, 1],
    ])
  })

  test('keeps topk finite on metal for large finite inputs', () => {
    const input = tensor([
      [-1e20, 0, 1e20],
      [1e10, -1e10, 5],
    ], { dtype: 'f32' }).to('metal')

    const result = topk(input, 2, 1)
    expect(values(result.values.to('cpu'))).toBeAllFinite()
    expect(values(result.indices.to('cpu'))).toEqual([[2, 1], [0, 2]])
  })
})
