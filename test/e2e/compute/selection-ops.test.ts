import { describe, test, expect, values } from 'std:test'
import { clamp, gather, grad, index_select, masked_fill, one_hot, sum, tensor, topk, where } from 'affon:compute/legacy'
import { internal_tensor } from '../../support/compute.ts'

describe('compute selection ops', () => {
  test('supports clamp where masked_fill one_hot gather index_select and topk', () => {
    expect(values(clamp(tensor([-1, 0.5, 2]), 0, 1))).toEqual([0, 0.5, 1])
    expect(
      values(where(tensor([[1], [0]]), tensor([[10, 20], [30, 40]]), tensor([[50, 60], [70, 80]]))),
    ).toEqual([[10, 20], [70, 80]])
    expect(
      values(masked_fill(tensor([[10, 20, 30], [40, 50, 60]]), tensor([[1, 0, 1], [0, 1, 0]]), -999)),
    ).toEqual([[-999, 20, -999], [40, -999, 60]])
    expect(values(one_hot(tensor([0, 2, 1]), 3))).toEqual([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    expect(
      values(gather(tensor([[10, 20, 30], [40, 50, 60]]), 1, tensor([[2, 0], [1, 1]]))),
    ).toEqual([[30, 10], [50, 50]])
    expect(values(index_select(tensor([[10, 20, 30], [40, 50, 60]]), 0, tensor([1, 0])))).toEqual([
      [40, 50, 60],
      [10, 20, 30],
    ])
    const result = topk(tensor([[10, 30, 20], [60, 40, 50]]), 2, 1)
    expect(values(result.values)).toEqual([[30, 20], [60, 50]])
    expect(values(result.indices)).toEqual([[1, 2], [0, 2]])
  })

  test('accumulates gradients back to gathered source positions', () => {
    const x = internal_tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32' })
    const y = gather(x, 1, tensor([[2, 0], [1, 1]]))
    grad(sum(y), [x as any])
    expect(values(x.grad)).toEqual([[1, 0, 1], [0, 2, 0]])
  })

  test('accumulates gradients back to selected source positions', () => {
    const x = internal_tensor([[10, 20], [30, 40], [50, 60]], { dtype: 'f32' })
    const y = index_select(x, 0, tensor([2, 1, 2]))
    grad(sum(y), [x as any])
    expect(values(x.grad)).toEqual([[0, 0], [1, 1], [2, 2]])
  })

  test('propagates gradients through topk values', () => {
    const x = internal_tensor([[10, 30, 20], [60, 40, 50]], { dtype: 'f32' })
    const result = topk(x, 2, 1)
    grad(sum(result.values), [x as any])
    expect(values(x.grad)).toEqual([[0, 1, 1], [1, 0, 1]])
  })

  test('propagates gradients only through unmasked positions', () => {
    const x = internal_tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32' })
    const y = masked_fill(x, tensor([[1, 0, 1], [0, 1, 0]]), -999)
    grad(sum(y), [x as any])
    expect(values(x.grad)).toEqual([[0, 1, 0], [1, 0, 1]])
  })

  test('keeps selection-family outputs finite for large finite inputs', () => {
    const large = tensor([
      [-1e20, 0, 1e20],
      [1e10, -1e10, 5],
    ], { dtype: 'f32' })

    expect(values(clamp(large, -1e5, 1e5))).toBeAllFinite()
    expect(values(where(tensor([[1, 0, 1], [0, 1, 0]]), large, tensor([[7, 8, 9], [10, 11, 12]], { dtype: 'f32' })))).toBeAllFinite()
    expect(values(masked_fill(large, tensor([[1, 0, 1], [0, 1, 0]]), -1e9))).toBeAllFinite()
    expect(values(gather(large, 1, tensor([[2, 1], [0, 2]])))).toBeAllFinite()
    expect(values(index_select(large, 0, tensor([1, 0, 1])))).toBeAllFinite()

    const top = topk(large, 2, 1)
    expect(values(top.values)).toBeAllFinite()
    expect(values(top.indices)).toEqual([[2, 1], [0, 2]])
  })
})
