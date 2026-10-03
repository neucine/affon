import { describe, expect, test } from 'std:test'
import { Session, Tensor, program } from 'affon:compute'
import { add, cat, masked_fill, matmul, mean, mul, neg, reshape, softmax, transpose } from 'affon:ops'

describe('affon:ops', () => {
  test('uses one operation vocabulary for formal and evaluated tensors', () => {
    const source = program('ops_formal', p => {
      const x = p.argument('x', Tensor.f32([2, 2]))
      const weight = p.argument('weight', Tensor.f32([2, 2]))
      return mean(softmax(add(matmul(x, weight), x), 1))
    })
    expect(source.inspect().nodes.filter(node => node.kind === 'operation').map(node => node.op)).toEqual([
      'matmul', 'add', 'softmax', 'mean',
    ])

    const session = new Session()
    const x = session.tensor([[1, 2], [3, 4]])
    const weight = session.tensor([[1, 0], [0, 1]])
    const product = matmul(x, weight)
    const shifted = add(product, x)
    const probabilities = softmax(shifted, 1)
    const average = mean(probabilities)
    expect(Math.abs(average.item() - 0.5) < 1e-6).toBe(true)
    average.dispose(); probabilities.dispose(); shifted.dispose(); product.dispose()
    x.dispose(); weight.dispose(); session.dispose()
  })

  test('supports variadic and shape operations in both modes', () => {
    const source = program('ops_shapes', p => {
      const x = p.argument('x', Tensor.f32([1, 2]))
      return transpose(reshape(cat([x, x], 0), [2, 2]), [1, 0])
    })
    expect(source.inspect().nodes.at(-1)?.spec.shape).toEqual([2, 2])

    const session = new Session()
    const x = session.tensor([[1, 2]])
    const joined = cat([x, x], 0)
    expect(transpose(joined, [1, 0]).to_array()).toEqual([[1, 1], [2, 2]])
    session.dispose()
  })

  test('rejects mixed representations, Sessions, and removed builder operations', () => {
    const left = new Session(), right = new Session()
    const a = left.tensor([1]), b = right.tensor([2])
    expect(() => add(a, b)).toThrow('same Session')
    program('no_builder_ops', p => {
      const x = p.argument('x', Tensor.f32([1]))
      expect(() => add(x, a as any)).toThrow('cannot mix')
      expect((p as any).cat).toBeUndefined()
      expect((p as any).operation).toBeUndefined()
      return x
    })
    left.dispose(); right.dispose()
  })

  test('infers broadcasts and rejects invalid formal specs during authoring', () => {
    const broadcast = program('broadcast_specs', p => {
      const matrix = p.argument('matrix', Tensor.f32([2, 3]))
      const bias = p.argument('bias', Tensor.f32([3]))
      const left = p.argument('left', Tensor.f32([4, 2, 3]))
      const right = p.argument('right', Tensor.f32([1, 3, 5]))
      return [add(matrix, bias), matmul(left, right)]
    }).inspect()
    expect(broadcast.nodes[broadcast.outputs[0]].spec.shape).toEqual([2, 3])
    expect(broadcast.nodes[broadcast.outputs[1]].spec.shape).toEqual([4, 2, 5])

    expect(() => program('bad_broadcast', p => mul(
      p.argument('left', Tensor.f32([2, 3])),
      p.argument('right', Tensor.f32([4])),
    ))).toThrow('cannot broadcast')
    expect(() => program('bad_dtype', p => add(
      p.argument('left', Tensor.f32([2])),
      p.argument('right', Tensor.f64([2])),
    ))).toThrow('dtype mismatch')
    expect(() => program('bad_reshape', p => reshape(
      p.argument('value', Tensor.f32([2, 3])), [5],
    ))).toThrow('preserve element count')
    expect(() => program('bad_mask', p => masked_fill(
      p.argument('value', Tensor.f32([2, 3])),
      p.argument('mask', Tensor.i64([4])), 0,
    ))).toThrow('cannot broadcast')
  })

  test('does not expose fluent operation methods on formal tensors', () => {
    program('formal_runtime_surface', p => {
      const value = p.argument('value', Tensor.f32([2]))
      expect((value as any).add).toBe(undefined)
      expect((value as any).$add).toBe(undefined)
      expect((value as any).matmul).toBe(undefined)
      expect((value as any).$matmul).toBe(undefined)
      expect((value as any).reshape).toBe(undefined)
      return neg(value)
    })
  })
})
