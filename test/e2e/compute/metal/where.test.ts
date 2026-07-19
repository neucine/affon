import { describe, test, expect, values } from 'std:test'
import { masked_fill, tensor, where } from 'affon:compute'

describe('compute metal where', () => {
  test('runs same-shape where on metal', () => {
    const cond = tensor([[1, 0], [0, 1]], { dtype: 'f32' }).to('metal')
    const a = tensor([[10, 20], [30, 40]], { dtype: 'f32' }).to('metal')
    const b = tensor([[50, 60], [70, 80]], { dtype: 'f32' }).to('metal')

    const y = where(cond, a, b)
    expect(y.device).toBe('metal')
    expect(values(y.to('cpu'))).toEqual([[10, 60], [70, 40]])
  })

  test('runs masked_fill on metal', () => {
    const x = tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32' }).to('metal')
    const mask = tensor([[1, 0, 1], [0, 1, 0]], { dtype: 'f32' }).to('metal')

    const y = masked_fill(x, mask, -999)
    expect(y.device).toBe('metal')
    expect(values(y.to('cpu'))).toEqual([[-999, 20, -999], [40, -999, 60]])
  })

  test('runs broadcasted where on metal', () => {
    const cond = tensor([[1], [0]], { dtype: 'f32' }).to('metal')
    const a = tensor([[10, 20], [30, 40]], { dtype: 'f32' }).to('metal')
    const b = tensor([[50, 60], [70, 80]], { dtype: 'f32' }).to('metal')

    const y = where(cond, a, b)
    expect(y.device).toBe('metal')
    expect(values(y.to('cpu'))).toEqual([[10, 20], [70, 80]])
  })

  test('keeps where and masked_fill finite on metal for large finite inputs', () => {
    const cond = tensor([[1, 0, 1], [0, 1, 0]], { dtype: 'f32' }).to('metal')
    const a = tensor([
      [-1e20, 0, 1e20],
      [1e10, -1e10, 5],
    ], { dtype: 'f32' }).to('metal')
    const b = tensor([
      [7, 8, 9],
      [10, 11, 12],
    ], { dtype: 'f32' }).to('metal')

    expect(values(where(cond, a, b).to('cpu'))).toBeAllFinite()
    expect(values(masked_fill(a, cond, -1e9).to('cpu'))).toBeAllFinite()
  })
})
