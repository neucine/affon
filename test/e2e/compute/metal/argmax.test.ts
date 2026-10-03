import { describe, test, expect, values } from 'std:test'
import { argmax, argmin, max, min, tensor } from 'affon:compute/legacy'

describe('compute metal min/max indices', () => {
  test('runs min/max and argmin/argmax on metal', () => {
    const x = tensor([[1, 9, 3], [7, 5, 6]], { dtype: 'f32' }).to('metal')

    const minAll = min(x)
    const maxAll = max(x)
    const argminAll = argmin(x)
    const all = argmax(x)
    const minAxis = min(x, 1)
    const maxAxis = max(x, 1)
    const argminAxis = argmin(x, 1)
    const axis = argmax(x, 1)

    expect(minAll.device).toBe('metal')
    expect(values(minAll.to('cpu'))).toEqual([1])
    expect(maxAll.device).toBe('metal')
    expect(values(maxAll.to('cpu'))).toEqual([9])
    expect(argminAll.device).toBe('metal')
    expect(values(argminAll.to('cpu'))).toEqual([0])
    expect(all.device).toBe('metal')
    expect(values(all.to('cpu'))).toEqual([1])
    expect(minAxis.device).toBe('metal')
    expect(values(minAxis.to('cpu'))).toEqual([1, 5])
    expect(maxAxis.device).toBe('metal')
    expect(values(maxAxis.to('cpu'))).toEqual([9, 7])
    expect(argminAxis.device).toBe('metal')
    expect(values(argminAxis.to('cpu'))).toEqual([0, 1])
    expect(axis.device).toBe('metal')
    expect(values(axis.to('cpu'))).toEqual([1, 0])
  })
})
