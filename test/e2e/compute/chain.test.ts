import { describe, test, expect, values } from 'std:test'
import { add, exp, grad, matmul, sum, tensor } from 'affon:compute/legacy'
import { internal_tensor } from '../../support/compute.ts'

describe('compute chain', () => {
  test('a small exp(W @ x + b) chain backpropagates into tracked values only', () => {
    const W = internal_tensor([[0.1, 0.2], [0.3, 0.4]], { dtype: 'f32' })
    const x = tensor([[1.0], [2.0]], { dtype: 'f32' })
    const b = internal_tensor([[0.0], [0.0]], { dtype: 'f32' })

    const y = exp(add(matmul(W, x), b))
    const loss = sum(y)
    grad(loss, [W as any, b as any])

    expect(W.grad !== null).toBe(true)
    expect(b.grad !== null).toBe(true)
    expect(values(x)).toEqual([[1], [2]])

    const wGrad = values(W.grad!)
    const bGrad = values(b.grad!)
    expect(Number.isFinite((wGrad as any)[0][0])).toBe(true)
    expect(Number.isFinite((bGrad as any)[0][0])).toBe(true)
  })
})
