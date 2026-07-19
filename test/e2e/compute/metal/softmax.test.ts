import { describe, test, expect, values } from 'std:test'
import { mul, softmax, sum } from 'affon:compute'
import { internal_tensor } from '../../../support/compute.ts'

// Metal tests intentionally check grad placement (`grad_device`) and direct
// tracked-input backward behavior, so they keep using the lower-level path.
describe('compute metal softmax', () => {
  test('runs softmax on metal and keeps forward output on device', () => {
    const x = internal_tensor([[1, 2, 3]], { dtype: 'f32', device: 'metal' })
    const y = softmax(x, 1)

    expect(y.device).toBe('metal')
    expect(values(y.to('cpu'))).toBeAllClose([[0.09003057, 0.24472847, 0.66524096]])
  })

  test('supports backward through metal softmax with metal grads', () => {
    const x = internal_tensor([[1, 2, 3]], { dtype: 'f32', device: 'metal' })
    const y = softmax(x, 1)
    const loss = sum(mul(y, x))
    loss.backward()

    expect(y.device).toBe('metal')
    expect(x.grad_device).toBe('metal')
    expect(values(x.grad?.to('cpu'))).toBeAllClose([[-0.05178652, 0.10395811, 0.9478284]])
  })

  test('keeps softmax finite on metal for extreme finite logits', () => {
    const x = internal_tensor([[1e20, 0, -1e20], [-1e20, 1e20, 0]], { dtype: 'f32', device: 'metal' })
    const y = softmax(x, 1)
    const rows = values(y.to('cpu')) as number[][]

    expect(rows).toBeAllFinite()
    expect(rows).toHaveRowSumsCloseTo(1)

    const loss = sum(mul(y, x))
    loss.backward()
    const grad = values(x.grad?.to('cpu')) as number[][]
    expect(grad).toBeAllFinite()
  })
})
