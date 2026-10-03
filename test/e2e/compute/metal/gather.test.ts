import { describe, test, expect, values } from 'std:test'
import { gather, sum, tensor } from 'affon:compute/legacy'
import { captureError } from '../../../support/errors.ts'
import { internal_tensor } from '../../../support/compute.ts'

describe('compute metal gather', () => {
  test('runs gather on metal and keeps output on metal', () => {
    const input = tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32' }).to('metal')
    const index = tensor([[2, 0], [1, 1]], { dtype: 'i64' }).to('metal')

    const y = gather(input, 1, index)

    expect(y.device).toBe('metal')
    expect(values(y.to('cpu'))).toEqual([[30, 10], [50, 50]])
  })

  test('keeps gather backward on metal and accumulates duplicate indices', () => {
    const input = internal_tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32', device: 'metal' })
    const index = tensor([[2, 0], [1, 1]], { dtype: 'i64' }).to('metal')

    sum(gather(input, 1, index)).backward()

    expect(input.grad_device).toBe('metal')
    expect(values(input.grad?.to('cpu'))).toEqual([
      [1, 0, 1],
      [0, 2, 0],
    ])
  })

  test('rejects out-of-bounds metal gather indices', () => {
    const input = tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32' }).to('metal')
    const index = tensor([[2, 3], [1, 1]], { dtype: 'i64' }).to('metal')

    const err = captureError(() => gather(input, 1, index))
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('IndexOutOfBounds')
  })

  test('keeps metal gather finite for large finite inputs', () => {
    const input = tensor([
      [-1e20, 0, 1e20],
      [1e10, -1e10, 5],
    ], { dtype: 'f32' }).to('metal')
    const index = tensor([[2, 1], [0, 2]], { dtype: 'i64' }).to('metal')

    expect(values(gather(input, 1, index).to('cpu'))).toBeAllFinite()
  })
})
