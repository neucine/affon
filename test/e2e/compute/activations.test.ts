import { describe, test, expect, values } from 'std:test'
import { grad, relu, sigmoid, sum, tanh } from 'affon:compute/legacy'
import { internal_tensor } from '../../support/compute.ts'

describe('compute activations', () => {
  test('relu forward and backward behave as expected on a simple vector', () => {
    const x = internal_tensor([-1.0, 0.0, 1.0, 2.0], { dtype: 'f32' })
    const y = relu(x)
    grad(sum(y), [x as any])

    expect(values(y)).toEqual([0, 0, 1, 2])
    expect(values(x.grad!)).toEqual([0, 0, 1, 1])
  })

  test('sigmoid forward and backward match the known x=0 value', () => {
    const x = internal_tensor([0.0], { dtype: 'f32' })
    const y = sigmoid(x)
    grad(sum(y), [x as any])

    expect(values(y)).toBeAllClose([0.5], { rtol: 1e-6, atol: 1e-6 })
    expect(values(x.grad!)).toBeAllClose([0.25], { rtol: 1e-6, atol: 1e-6 })
  })

  test('tanh forward and backward match the known x=0 value', () => {
    const x = internal_tensor([0.0], { dtype: 'f32' })
    const y = tanh(x)
    grad(sum(y), [x as any])

    expect(values(y)).toBeAllClose([0], { rtol: 1e-6, atol: 1e-6 })
    expect(values(x.grad!)).toBeAllClose([1], { rtol: 1e-6, atol: 1e-6 })
  })
})
