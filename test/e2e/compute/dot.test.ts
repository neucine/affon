import { describe, test, expect, values } from 'std:test'
import { dot, grad, tensor } from 'affon:compute'
import { internal_tensor } from '../../support/compute.ts'

describe('compute dot', () => {
  test('computes dot products and propagates gradients', () => {
    const a = internal_tensor([1, 2, 3], { dtype: 'f32' })
    const b = internal_tensor([4, 5, 6], { dtype: 'f32' })

    const y = dot(a, b)
    expect(values(y)).toEqual(32)

    grad(y, [a as any, b as any])
    expect(values(a.grad)).toEqual([4, 5, 6])
    expect(values(b.grad)).toEqual([1, 2, 3])
  })

  test('keeps dot finite for large finite safe-range inputs', () => {
    const a = tensor([1e10, -1e10, 3], { dtype: 'f32' })
    const b = tensor([2, 3, -4], { dtype: 'f32' })
    const out = dot(a, b).item()
    expect(Number.isFinite(out)).toBe(true)
  })
})
