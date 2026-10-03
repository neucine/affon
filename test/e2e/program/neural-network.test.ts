import { describe, expect, test } from 'std:test'
import { Session, Tensor, program, type FormalTensor } from 'affon:compute'
import { embedding, layer_norm, linear } from 'affon:nn'
import { cross_entropy } from 'affon:ops'

describe('Program neural-network authoring', () => {
  test('authors reusable linear, embedding, normalization, and ordinary function composition', () => {
    const input = linear({ out_features: 4 })
    const norm = layer_norm()
    const output = linear({ out_features: 2, bias: false })
    const encoder = ({ x }: { x: FormalTensor }, name = 'encoder') => {
      let value = input({ x }, `${name}.input`)
      value = norm({ x: value }, `${name}.norm`)
      return output({ x: value }, `${name}.output`)
    }
    const model = program('canonical_nn', p => encoder({ x: p.argument('x', Tensor.f32([2, 3])) }, 'encoder'))
    expect(model.inspect().parameters.map(value => value.name)).toEqual([
      'encoder.input.weight', 'encoder.input.bias', 'encoder.norm.weight', 'encoder.norm.bias', 'encoder.output.weight',
    ])
    const modelInspection = model.inspect()
    expect(modelInspection.nodes[modelInspection.outputs[0]].spec.shape).toEqual([2, 2])

    const tokens = embedding({ num_embeddings: 8, embedding_dim: 3 })
    const lookup = program('lookup', p => tokens({ indices: p.argument('indices', Tensor.i64([2])) }, 'tokens'))
    expect(lookup.inspect().parameters[0].name).toBe('tokens.weight')
    const lookupInspection = lookup.inspect()
    expect(lookupInspection.nodes[lookupInspection.outputs[0]].spec.shape).toEqual([2, 3])
  })

  test('authors cross entropy as an ordinary operation and validates label specs', () => {
    const loss = program('classification_loss', p => {
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
    const head = linear({ out_features: 2 })
    const model = program('linear_runtime', p => head({ x: p.argument('x', Tensor.f32([2, 2])) }, 'head'))
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
