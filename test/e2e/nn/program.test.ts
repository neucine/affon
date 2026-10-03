import { describe, expect, test } from 'std:test'
import { Session, Tensor, program, type FormalTensor, type ProgramBuilder } from 'affon:compute'
import { cross_entropy } from 'affon:ops'

describe('nn Program API', () => {
  test('authors reusable linear, embedding, normalization, and ordinary function composition', () => {
    const encoder = (p: ProgramBuilder, value: FormalTensor) => {
      value = p.nn.linear(value, { name: 'input', out_features: 4 })
      value = p.nn.layer_norm(value, { name: 'norm' })
      return p.nn.linear(value, { name: 'output', out_features: 2, bias: false })
    }
    const model = program('canonical_nn', p => encoder(p, p.argument('x', Tensor.f32([2, 3]))))
    expect(model.inspect().parameters.map(value => value.name)).toEqual([
      'input_weight', 'input_bias', 'norm_weight', 'norm_bias', 'output_weight',
    ])
    const modelInspection = model.inspect()
    expect(modelInspection.nodes[modelInspection.outputs[0]].spec.shape).toEqual([2, 2])

    const lookup = program('lookup', p => p.nn.embedding(p.argument('indices', Tensor.i64([2])), {
      name: 'tokens', num_embeddings: 8, embedding_dim: 3,
    }))
    expect(lookup.inspect().parameters[0].name).toBe('tokens_weight')
    const lookupInspection = lookup.inspect()
    expect(lookupInspection.nodes[lookupInspection.outputs[0]].spec.shape).toEqual([2, 3])
  })

  test('authors cross entropy as an ordinary operation and validates label specs', () => {
    const loss = program('classification_loss', p => {
      expect((p.nn as any).cross_entropy).toBeUndefined()
      const logits = p.argument('logits', Tensor.f32([2, 3]))
      const labels = p.argument('labels', Tensor.i64([2]))
      return cross_entropy(logits, labels)
    })
    const inspection = loss.inspect()
    expect(inspection.nodes[inspection.outputs[0]].spec.shape).toEqual([1])

    expect(() => program('bad_loss_dtype', p => cross_entropy(
      p.argument('logits', Tensor.f32([2, 3])),
      p.argument('labels', Tensor.f32([2])),
    ))).toThrow('i64')
    expect(() => program('bad_loss_shape', p => cross_entropy(
      p.argument('logits', Tensor.f32([2, 3])),
      p.argument('labels', Tensor.i64([3])),
    ))).toThrow('without its class dimension')
  })

  test('executes canonical layers with Session-owned values', () => {
    const model = program('linear_runtime', p => p.nn.linear(p.argument('x', Tensor.f32([2, 2])), {
      name: 'head', out_features: 2,
    }))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(model, { seed: 7 })
    const x = session.tensor([[1, 0], [0, 1]])
    const result = session.compile(model).run({ x }, state) as any
    expect(result.shape).toEqual([2, 2])
    expect((result.to_array() as number[][]).flat().every(Number.isFinite)).toBe(true)
    result.dispose()
    x.dispose()
    state.dispose()
    session.dispose()
  })
})
