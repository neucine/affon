// Fixed reference cases adapted from affon-legacy/test/parity/nn.
import { Session, Tensor, gradient, program } from 'affon:compute'
import { embedding, layer_norm, linear } from 'affon:nn'
import { sum } from 'affon:ops'
import { describe, expect, test } from 'std:test'

function close(actual: unknown, expected: unknown, tolerance = 1e-5): void {
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual)).toBe(true)
    expect((actual as unknown[]).length).toBe(expected.length)
    expected.forEach((item, index) => close((actual as unknown[])[index], item, tolerance))
  } else expect(Math.abs((actual as number) - (expected as number)) <= tolerance).toBe(true)
}

describe('Neural-network reference values', () => {
  test('linear forward and weight gradient on batched input', () => {
    const layer = linear({ out_features: 2 })
    const model = program('linear_reference', p => layer({ x: p.argument('x', Tensor.f32([2, 2])) }, 'head'))
    const derivative = gradient(program('linear_gradient_reference', p => sum(layer({ x: p.argument('x', Tensor.f32([2, 2])) }, 'head'))), 'head.weight')
    const session = new Session({ device: 'cpu' })
    const parameters = { 'head.weight': [[1, 2], [3, 4]], 'head.bias': [5, 6] }
    const state = session.initialize(model, { parameters })
    const gradState = session.initialize(derivative, { parameters })
    const x = session.tensor([[1, 2], [3, 4]])
    const output = session.compile(model).run({ x }, state)
    const gradWeight = session.compile(derivative).run({ x }, gradState)
    close(output.to_array(), [[12, 16], [20, 28]])
    close(gradWeight.to_array(), [[4, 4], [6, 6]])
    output.dispose(); gradWeight.dispose(); x.dispose(); state.dispose(); gradState.dispose(); session.dispose()
  })

  test('layer normalization uses population variance and affine parameters', () => {
    const norm = layer_norm({ epsilon: 1e-5 })
    const model = program('norm_reference', p => norm({ x: p.argument('x', Tensor.f32([1, 3])) }, 'norm'))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(model, { parameters: { 'norm.weight': [2, 2, 2], 'norm.bias': [1, 1, 1] } })
    const x = session.tensor([[1, 2, 3]])
    const output = session.compile(model).run({ x }, state)
    const scale = 2 / Math.sqrt(2 / 3 + 1e-5)
    close(output.to_array(), [[1 - scale, 1, 1 + scale]])
    output.dispose(); x.dispose(); state.dispose(); session.dispose()
  })

  test('embedding lookup preserves repeated indices', () => {
    const lookup = embedding({ num_embeddings: 3, embedding_dim: 2 })
    const model = program('embedding_reference', p => lookup({ indices: p.argument('indices', Tensor.i64([3])) }, 'table'))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(model, { parameters: { 'table.weight': [[1, 2], [3, 4], [5, 6]] } })
    const indices = session.tensor([2, 0, 2], { dtype: 'i64' })
    const output = session.compile(model).run({ indices }, state)
    close(output.to_array(), [[5, 6], [1, 2], [5, 6]])
    output.dispose(); indices.dispose(); state.dispose(); session.dispose()
  })
})
