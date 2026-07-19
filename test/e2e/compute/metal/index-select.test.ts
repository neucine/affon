import { describe, test, expect, values } from 'std:test'
import { index_select, sum, tensor } from 'affon:compute'
import { internal_tensor } from '../../../support/compute.ts'

describe('compute metal index_select', () => {
  test('runs index_select on metal and keeps output on metal', () => {
    const input = tensor([[10, 20], [30, 40], [50, 60]], { dtype: 'f32' }).to('metal')
    const index = tensor([2, 0], { dtype: 'i64' }).to('metal')

    const y = index_select(input, 0, index)

    expect(y.device).toBe('metal')
    expect(values(y.to('cpu'))).toEqual([[50, 60], [10, 20]])
  })

  test('keeps index_select backward on metal and accumulates duplicate indices', () => {
    const input = internal_tensor([[10, 20], [30, 40], [50, 60]], { dtype: 'f32', device: 'metal' })
    const index = tensor([2, 1, 2], { dtype: 'i64' }).to('metal')

    sum(index_select(input, 0, index)).backward()

    expect(input.grad_device).toBe('metal')
    expect(values(input.grad?.to('cpu'))).toEqual([
      [0, 0],
      [1, 1],
      [2, 2],
    ])
  })
})
