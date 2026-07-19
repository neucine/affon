import { describe, test, expect, values } from 'std:test'
import { argmax, argmin, grad, max, mean, min, mul, softmax, std, sum, tensor, variance } from 'affon:compute'
import { internal_tensor } from '../../support/compute.ts'

describe('compute reduction ops', () => {
  test('supports common reductions and gradients', () => {
    expect(values(softmax(tensor([[1, 2, 3]]), 1))).toBeAllClose([[0.09003057317038046, 0.24472847105479764, 0.6652409557748218]])
    expect(values(variance(tensor([[1, 2], [3, 4]]), 1, false))).toEqual([0.25, 0.25])
    expect(values(std(tensor([[1, 2], [3, 4]]), 1, false))).toEqual([0.5, 0.5])
    expect(values(min(tensor([[1, 3], [5, 4]]), 1))).toEqual([1, 4])
    expect(values(max(tensor([[1, 3], [5, 4]]), 1))).toEqual([3, 5])
    expect(values(argmin(tensor([[1, 3], [5, 4]]), 1))).toEqual([0, 1])
    expect(values(argmax(tensor([[1, 3], [5, 4]]), 1))).toEqual([1, 0])

    const x = internal_tensor([[1, 2, 3]], { dtype: 'f32' })
    const y = softmax(x, 1)
    const loss = sum(mul(y, x))
    grad(loss, [x as any])
    expect(values(x.grad)).toBeAllClose([[-0.05178652043943169, 0.10395811358516757, 0.9478284068542642]])

    const v = internal_tensor([[[1, 2, 3], [3, 4, 5]]], { dtype: 'f32' })
    grad(sum(variance(v, 2, true)), [v as any])
    expect(v.grad?.shape).toEqual([1, 2, 3])
  })

  test('keeps softmax finite for extreme but finite logits', () => {
    // This stays on `.backward()` because it validates direct tracked-input
    // gradient stability rather than the public parameter-oriented grad path.
    const x = internal_tensor([[1000, 0, -1000], [-1000, 1000, 0]], { dtype: 'f32' })
    const y = softmax(x, 1)
    const rows = values(y) as number[][]

    for (const row of rows) {
      expect(row.every((value) => Number.isFinite(value))).toBe(true)
      expect(Math.abs(row.reduce((acc, value) => acc + value, 0) - 1) < 1e-6).toBe(true)
    }

    sum(mul(y, x)).backward()
    const grad = values(x.grad) as number[][]
    expect(grad.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)
  })

  test('keeps safe-range reductions finite for large finite values', () => {
    const x = tensor([
      [1e10, 1e10 + 1e4, 1e10 - 1e4],
      [-1e10, -1e10 + 1e4, -1e10 - 1e4],
    ], { dtype: 'f32' })

    expect(values(sum(x, 1))).toBeAllFinite()
    expect(values(mean(x, 1))).toBeAllFinite()
    expect(values(min(x, 1))).toBeAllFinite()
    expect(values(max(x, 1))).toBeAllFinite()
    expect(values(argmin(x, 1))).toEqual([2, 2])
    expect(values(argmax(x, 1))).toEqual([1, 1])
  })
})
