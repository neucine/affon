// Gradient reference cases adapted from affon-legacy/test/parity/compute.
import { add, matmul, mean, mul, reshape, sum, transpose } from 'affon:ops'
import { Session, Tensor, gradient, program } from 'affon:compute'
import { describe, expect, test } from 'std:test'

function close(actual: unknown, expected: unknown): void {
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual)).toBe(true)
    expect((actual as unknown[]).length).toBe(expected.length)
    expected.forEach((entry, index) => close((actual as unknown[])[index], entry))
  } else expect(Math.abs((actual as number) - (expected as number)) < 1e-5).toBe(true)
}

describe('Gradient reference values through Program transforms', () => {
  test('broadcast add reduces the repeated operand gradient', () => {
    const objective = program('legacy_broadcast_add', p => sum(add(
      p.argument('x', Tensor.f32([2, 3])),
      p.parameter('bias', Tensor.f32([1, 3]), { initializer: { kind: 'ones' } }),
    )))
    const derivative = gradient(objective, 'bias')
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(derivative)
    const x = session.tensor([[1, 2, 3], [4, 5, 6]])
    const result = session.compile(derivative).run({ x }, state)
    close(result.to_array(), [[2, 2, 2]])
    result.dispose(); x.dispose(); state.dispose(); session.dispose()
  })

  test('matmul gradient matches the transposed left operand', () => {
    const objective = program('legacy_matmul_gradient', p => sum(matmul(
      p.argument('x', Tensor.f32([2, 3])),
      p.parameter('weight', Tensor.f32([3, 2]), { initializer: { kind: 'ones' } }),
    )))
    const derivative = gradient(objective, 'weight')
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(derivative)
    const x = session.tensor([[1, 2, 3], [4, 5, 6]])
    const result = session.compile(derivative).run({ x }, state)
    close(result.to_array(), [[5, 5], [7, 7], [9, 9]])
    result.dispose(); x.dispose(); state.dispose(); session.dispose()
  })

  test('mean of a product scales the parameter gradient', () => {
    const objective = program('legacy_mean_product', p => mean(mul(
      p.argument('x', Tensor.f32([2, 2])),
      p.parameter('scale', Tensor.f32([2, 2]), { initializer: { kind: 'ones' } }),
    )))
    const derivative = gradient(objective, 'scale')
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(derivative)
    const x = session.tensor([[1, 2], [3, 4]])
    const result = session.compile(derivative).run({ x }, state)
    close(result.to_array(), [[0.25, 0.5], [0.75, 1]])
    result.dispose(); x.dispose(); state.dispose(); session.dispose()
  })

  test('reshape and transpose route gradients back to original parameter shape', () => {
    const objective = program('view_gradient', p => {
      const weight = p.parameter('weight', Tensor.f32([2, 3]), { initializer: { kind: 'ones' } })
      const coefficients = p.argument('coefficients', Tensor.f32([3, 2]))
      return sum(mul(transpose(reshape(weight, [2, 3]), [1, 0]), coefficients))
    })
    const derivative = gradient(objective, 'weight')
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(derivative)
    const coefficients = session.tensor([[1, 2], [3, 4], [5, 6]])
    const result = session.compile(derivative).run({ coefficients }, state)
    close(result.to_array(), [[1, 3, 5], [2, 4, 6]])
    result.dispose(); coefficients.dispose(); state.dispose(); session.dispose()
  })

  test('multiple uses of one parameter accumulate through separate branches', () => {
    const objective = program('branch_gradient', p => {
      const x = p.argument('x', Tensor.f32([2, 2]))
      const weight = p.parameter('weight', Tensor.f32([2, 2]), { initializer: { kind: 'ones' } })
      return sum(add(mul(x, weight), mul(weight, weight)))
    })
    const derivative = gradient(objective, 'weight')
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(derivative)
    const x = session.tensor([[1, 2], [3, 4]])
    const result = session.compile(derivative).run({ x }, state)
    close(result.to_array(), [[3, 4], [5, 6]])
    result.dispose(); x.dispose(); state.dispose(); session.dispose()
  })
})
