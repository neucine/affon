import { describe, test, expect, values } from 'std:test'
import { std, tensor, variance } from 'affon:compute/legacy'

describe('compute metal variance', () => {
  test('runs variance and std on metal', () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32' }).to('metal')

    const varAll = variance(x)
    const stdAxis = std(x, 1, false)

    expect(varAll.device).toBe('metal')
    expect(values(varAll.to('cpu'))).toBeAllClose([2.9166667])
    expect(stdAxis.device).toBe('metal')
    expect(values(stdAxis.to('cpu'))).toBeAllClose([0.8164966, 0.8164966])
  })

  test('keeps variance and std finite on metal for large finite magnitudes', () => {
    const x = tensor([
      [1e20, 1e20 + 1e12, 1e20 - 1e12],
      [-1e20, -1e20 + 1e12, -1e20 - 1e12],
    ], { dtype: 'f32' }).to('metal')

    const varAxis = variance(x, 1, false)
    const stdAxis = std(x, 1, false)
    const varRows = values(varAxis.to('cpu')) as number[]
    const stdRows = values(stdAxis.to('cpu')) as number[]

    expect(varRows.every((value) => Number.isFinite(value))).toBe(true)
    expect(stdRows.every((value) => Number.isFinite(value))).toBe(true)
  })
})
