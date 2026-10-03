import { describe, test, expect, values } from 'std:test'
import { add, sum } from 'affon:compute/legacy'
import { internal_tensor } from '../../../support/compute.ts'

describe('compute metal broadcast', () => {
  test('runs broadcast add on metal and keeps grads on metal', () => {
    const a = internal_tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32', device: 'metal' })
    const b = internal_tensor([[10, 20, 30]], { dtype: 'f32', device: 'metal' })
    const y = add(a, b)

    expect(y.device).toBe('metal')
    expect(values(y.to('cpu'))).toEqual([[11, 22, 33], [14, 25, 36]])

    sum(y).backward()
    expect(a.grad_device).toBe('metal')
    expect(b.grad_device).toBe('metal')
    expect(values(a.grad?.to('cpu'))).toEqual([[1, 1, 1], [1, 1, 1]])
    expect(values(b.grad?.to('cpu'))).toEqual([[2, 2, 2]])
  })
})
