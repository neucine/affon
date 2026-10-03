import { describe, expect, test, values } from 'std:test'
import { clear_grad, clip_grad_norm, grad, mul, parameter, sum, tensor } from 'affon:compute/legacy'

describe('compute metal clip_grad_norm', () => {
  test('reduces and clips f32 gradients without moving them to cpu', () => {
    const weight = parameter([2], { dtype: 'f32', device: 'metal' }).ones()
    clear_grad([weight])

    const loss = sum(mul(weight, tensor([10, 0], { dtype: 'f32', device: 'metal' })))
    grad(loss, [weight])

    const totalNorm = clip_grad_norm([weight], 5)

    expect(totalNorm > 5).toBe(true)
    expect(weight.grad_device).toBe('metal')
    const clipped = values(weight.grad?.to('cpu')) as number[]
    expect(Math.abs(clipped[0] - 5) < 1e-5).toBe(true)
    expect(Math.abs(clipped[1]) < 1e-5).toBe(true)
  })
})
