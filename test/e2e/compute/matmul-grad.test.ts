import { describe, test, expect, values } from 'std:test'
import { grad, matmul, sum, tensor } from 'affon:compute/legacy'
import { internal_tensor } from '../../support/compute.ts'

describe('compute matmul grad', () => {
  test('matmul backward matches the simple hand-derived case', () => {
    const W = internal_tensor([[1.0, 2.0], [3.0, 4.0]], { dtype: 'f32' })
    const x = internal_tensor([[5.0], [6.0]], { dtype: 'f32' })

    const y = matmul(W, x)
    const loss = sum(y)
    grad(loss, [W as any, x as any])

    expect(values(W.grad!)).toEqual([[5, 6], [5, 6]])
    expect(values(x.grad!)).toEqual([[4], [6]])
  })
})
