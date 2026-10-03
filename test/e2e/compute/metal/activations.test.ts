import { describe, test, expect, values } from 'std:test'
import { abs, clamp, gelu, mean, relu, sigmoid, sum, tanh, tensor } from 'affon:compute/legacy'
import { internal_tensor } from '../../../support/compute.ts'

describe('compute metal activations', () => {
  test('runs relu sigmoid tanh and clamp on metal', () => {
    const x = tensor([-1, 0, 1], { dtype: 'f32' }).to('metal')

    expect(relu(x).device).toBe('metal')
    expect(values(relu(x).to('cpu'))).toEqual([0, 0, 1])

    const sig = values(sigmoid(x).to('cpu')) as number[]
    expect(sig[0] < 0.5).toBe(true)
    expect(Math.abs(sig[1] - 0.5) < 1e-6).toBe(true)
    expect(sig[2] > 0.5).toBe(true)

    const th = values(tanh(x).to('cpu')) as number[]
    expect(th[0] < 0).toBe(true)
    expect(Math.abs(th[1]) < 1e-6).toBe(true)
    expect(th[2] > 0).toBe(true)

    expect(clamp(x, 0, 1).device).toBe('metal')
    expect(values(clamp(x, 0, 1).to('cpu'))).toEqual([0, 0, 1])
  })

  test('keeps abs and relu backward on metal', () => {
    const x = internal_tensor([-2, -0.5, 0, 3], { dtype: 'f32', device: 'metal' })

    sum(abs(x)).backward()
    expect(x.grad_device).toBe('metal')
    expect(values(x.grad?.to('cpu'))).toEqual([-1, -1, 0, 1])

    const y = internal_tensor([-2, -0.5, 0, 3], { dtype: 'f32', device: 'metal' })
    sum(relu(y)).backward()
    expect(y.grad_device).toBe('metal')
    expect(values(y.grad?.to('cpu'))).toEqual([0, 0, 0, 1])
  })

  test('keeps axis reduction backward on metal', () => {
    const x = internal_tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32', device: 'metal' })

    sum(mean(x, 1)).backward()
    expect(x.grad_device).toBe('metal')
    const grad = values(x.grad?.to('cpu')) as number[][]
    for (const row of grad) {
      for (const value of row) expect(Math.abs(value - 1 / 3) < 1e-6).toBe(true)
    }
  })

  test('keeps gelu and clamp backward on metal', () => {
    const x = internal_tensor([-2, -0.5, 0, 3], { dtype: 'f32', device: 'metal' })

    sum(gelu(x)).backward()
    expect(x.grad_device).toBe('metal')

    const y = internal_tensor([-2, -0.5, 0, 3], { dtype: 'f32', device: 'metal' })
    sum(clamp(y, -0.5, 2)).backward()
    expect(y.grad_device).toBe('metal')
    expect(values(y.grad?.to('cpu'))).toEqual([0, 0, 1, 0])
  })

  test('keeps gelu finite for extreme finite metal inputs', () => {
    const x = internal_tensor([-1e20, -20, -10, 0, 10, 20, 1e20], { dtype: 'f32', device: 'metal' })
    const y = gelu(x)
    const yValues = values(y.to('cpu')) as number[]

    expect(yValues.every((value) => Number.isFinite(value))).toBe(true)
    expect(Math.abs(yValues[0]) < 1e-6).toBe(true)
    expect(Math.abs(yValues[1]) < 1e-6).toBe(true)
    expect(Math.abs(yValues[3]) < 1e-6).toBe(true)
    expect(Math.abs(yValues[4] - 10) < 1e-4).toBe(true)
    expect(Math.abs(yValues[5] - 20) < 1e-4).toBe(true)
    expect(Math.abs(yValues[6] - 1e20) / 1e20 < 1e-6).toBe(true)

    sum(y).backward()
    expect(x.grad_device).toBe('metal')
    const grad = values(x.grad?.to('cpu')) as number[]
    expect(grad.every((value) => Number.isFinite(value))).toBe(true)
    expect(Math.abs(grad[0]) < 1e-6).toBe(true)
    expect(Math.abs(grad[4] - 1) < 1e-6).toBe(true)
    expect(Math.abs(grad[6] - 1) < 1e-6).toBe(true)
  })

  test('keeps sigmoid tanh and clamp finite on metal for extreme finite inputs', () => {
    const x = internal_tensor([-1e20, -100, 0, 100, 1e20], { dtype: 'f32', device: 'metal' })

    const sig = values(sigmoid(x).to('cpu')) as number[]
    expect(sig).toBeAllFinite()
    expect(sig[0]).toBe(0)
    expect(Math.abs(sig[2] - 0.5) < 1e-6).toBe(true)
    expect(sig[4]).toBe(1)

    const th = values(tanh(x).to('cpu')) as number[]
    expect(th).toBeAllFinite()
    expect(th[0]).toBe(-1)
    expect(Math.abs(th[2]) < 1e-6).toBe(true)
    expect(th[4]).toBe(1)

    const clipped = clamp(x, -7, 7)
    expect(values(clipped.to('cpu'))).toEqual([-7, -7, 0, 7, 7])

    sum(sigmoid(x)).backward()
    const sigGrad = values(x.grad?.to('cpu')) as number[]
    expect(sigGrad).toBeAllFinite()
  })
})
