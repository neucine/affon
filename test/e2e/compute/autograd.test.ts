import { describe, test, expect, values } from 'std:test'
import { add, grad, mul, neg, no_grad, sum, tensor } from 'affon:compute/legacy'
import { internal_tensor } from '../../support/compute.ts'
import { captureError } from '../../support/errors.ts'

describe('compute autograd basics', () => {
  test('square sum produces the expected gradient', () => {
    const x = internal_tensor([2.0, 3.0], { dtype: 'f32' })
    const y = mul(x, x)
    grad(sum(y), [x as any])

    expect(values(x.grad!)).toEqual([4, 6])
  })

  test('a * b + a produces the expected gradient for a', () => {
    const a = internal_tensor([1.0, 2.0], { dtype: 'f32' })
    const b = tensor([3.0, 4.0], { dtype: 'f32' })
    const d = add(mul(a, b), a)
    grad(sum(d), [a as any])

    expect(values(a.grad!)).toEqual([4, 5])
  })

  test('neg contributes a gradient of -1', () => {
    const x = internal_tensor([5.0], { dtype: 'f32' })
    const y = neg(x)
    grad(y, [x as any])

    expect(values(x.grad!)).toEqual([-1])
  })

  test('no_grad disables tracking for enclosed computations', () => {
    const x = internal_tensor([1.0], { dtype: 'f32' })
    const y = no_grad(() => mul(x, x))
    const err = captureError(() => grad(sum(y), [x as any]))
    expect(err).toBeInstanceOf(Error)
  })
})
