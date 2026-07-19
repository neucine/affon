import { describe, expect, test, values } from 'std:test'
import {
  all,
  at,
  cat,
  contiguous,
  gather,
  index_select,
  masked_fill,
  one_hot,
  permute,
  range,
  reshape,
  slice,
  squeeze,
  stack,
  topk,
  transpose,
  unsqueeze,
  tensor,
  where,
} from 'affon:compute'
import { internal_tensor } from '../../support/compute.ts'

describe('compute selection and shape ops', () => {
  test('supports selection-family operations', () => {
    const source = tensor([[10, 20, 30], [40, 50, 60]])
    expect(at(source, 1, 2).item()).toBe(60)
    expect(values(slice(source, all, range(0, 2)))).toEqual([[10, 20], [40, 50]])
    expect(values(where(tensor([[1], [0]], { dtype: 'i64' }), tensor([[10, 20], [30, 40]]), tensor([[50, 60], [70, 80]])))).toEqual([[10, 20], [70, 80]])
    expect(values(masked_fill(source, tensor([[1, 0, 1], [0, 1, 0]], { dtype: 'i64' }), -999))).toEqual([[-999, 20, -999], [40, -999, 60]])
    expect(values(one_hot(tensor([0, 2, 1], { dtype: 'i64' }), 3))).toEqual([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    expect(values(gather(source, 1, tensor([[2, 0], [1, 1]], { dtype: 'i64' })))).toEqual([[30, 10], [50, 50]])
    expect(values(index_select(source, 0, tensor([1, 0], { dtype: 'i64' })))).toEqual([[40, 50, 60], [10, 20, 30]])

    const ranked = topk(tensor([[10, 30, 20], [60, 40, 50]]), 2, 1)
    expect(values(ranked.values)).toEqual([[30, 20], [60, 50]])
    expect(values(ranked.indices)).toEqual([[1, 2], [0, 2]])
  })

  test('supports shape-family operations and routes gradients', () => {
    expect(values(cat([tensor([[1, 2], [3, 4]]), tensor([[5, 6]])], 0))).toEqual([[1, 2], [3, 4], [5, 6]])
    expect(values(stack([tensor([[1, 2], [3, 4]]), tensor([[1, 2], [3, 4]])], 0))).toEqual([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    expect(values(squeeze(tensor([[[1], [2]]])))).toEqual([1, 2])
    expect(values(unsqueeze(tensor([1, 2, 3]), 0))).toEqual([[1, 2, 3]])
    expect(values(reshape(tensor([[1, 2], [3, 4]]), [4]))).toEqual([1, 2, 3, 4])
    expect(values(transpose(tensor([[1, 2], [3, 4]]), 0, 1))).toEqual([[1, 3], [2, 4]])
    expect(values(permute(tensor([[[1, 2], [3, 4], [5, 6]]]), [0, 2, 1]))).toEqual([[[1, 3, 5], [2, 4, 6]]])
    expect(values(contiguous(permute(tensor([[[1, 2], [3, 4], [5, 6]]]), [0, 2, 1])))).toEqual([[[1, 3, 5], [2, 4, 6]]])

    const x = internal_tensor([[1, 2], [3, 4]])
    const y = reshape(contiguous(transpose(unsqueeze(x, 0), 1, 2)), [4])
    expect(values(y)).toEqual([1, 3, 2, 4])
  })
})
