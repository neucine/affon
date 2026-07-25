import { describe, test, expect, values } from 'std:test'
import { clamp, div, exp, log, sigmoid, sqrt, tanh, tensor } from 'affon:compute'

describe('compute numeric contracts', () => {
  test('keeps exp finite for large negative finite inputs and overflows honestly for large positive inputs', () => {
    const negative = values(exp(tensor([-1000, -100, -20, 0], { dtype: 'f32' }))) as number[]
    expect(negative).toBeAllFinite()
    expect(negative[0]).toBe(0)
    expect(negative[1] >= 0 && negative[1] < 1e-40).toBe(true)

    const positive = values(exp(tensor([100], { dtype: 'f32' }))) as number[]
    expect(positive[0]).toBe(Infinity)
  })

  test('surfaces invalid log and sqrt domains explicitly', () => {
    const logged = values(log(tensor([1, 0, -1], { dtype: 'f32' }))) as number[]
    expect(Number.isFinite(logged[0])).toBe(true)
    expect(logged[1]).toBe(-Infinity)
    expect([logged[2]]).toContainNaN()

    const rooted = values(sqrt(tensor([4, 0, -1], { dtype: 'f32' }))) as number[]
    expect(rooted[0]).toBe(2)
    expect(rooted[1]).toBe(0)
    expect([rooted[2]]).toContainNaN()
  })

  test('surfaces division-by-zero behavior explicitly', () => {
    const quotient = values(div(
      tensor([1, -1, 0], { dtype: 'f32' }),
      tensor([0, 0, 0], { dtype: 'f32' }),
    )) as number[]

    expect(quotient[0]).toBe(Infinity)
    expect(quotient[1]).toBe(-Infinity)
    expect([quotient[2]]).toContainNaN()
  })

  test('keeps sigmoid tanh and clamp finite for extreme finite inputs', () => {
    const x = tensor([-1e20, -100, 0, 100, 1e20], { dtype: 'f32' })

    const sig = values(sigmoid(x)) as number[]
    expect(sig).toBeAllFinite()
    expect(sig[0]).toBe(0)
    expect(Math.abs(sig[2] - 0.5) < 1e-6).toBe(true)
    expect(sig[4]).toBe(1)

    const th = values(tanh(x)) as number[]
    expect(th).toBeAllFinite()
    expect(th[0]).toBe(-1)
    expect(Math.abs(th[2]) < 1e-6).toBe(true)
    expect(th[4]).toBe(1)

    const clipped = values(clamp(x, -7, 7)) as number[]
    expect(clipped).toBeAllFinite()
    expect(clipped).toEqual([-7, -7, 0, 7, 7])
  })
})
