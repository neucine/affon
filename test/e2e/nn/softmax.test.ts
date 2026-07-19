import { describe, test, expect, values } from 'std:test'
import { softmax, tensor } from 'affon:compute'

describe('nn softmax', () => {
  test('matches the tensor softmax surface on a simple row input', () => {
    const y = softmax(tensor([[1, 2, 3]], { dtype: 'f32' }), 1)
    const actual = values(y) as number[][]

    expect(y.shape).toEqual([1, 3])
    expect(actual).toBeAllClose([[0.09003057, 0.24472847, 0.66524096]], { rtol: 1e-5, atol: 1e-6 })
    expect(Math.abs((actual[0][0] + actual[0][1] + actual[0][2]) - 1) <= 1e-6).toBe(true)
  })
})
