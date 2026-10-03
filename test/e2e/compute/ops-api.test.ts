import { describe, expect, test } from 'std:test'
import { Session, Tensor, metrics, program } from 'affon:compute'
import { add, argmax, binary_cross_entropy_with_logits, cat, clamp, cross_entropy, gather, gt_scalar, masked_fill, matmul, max, mean, mean_absolute_error, mean_squared_error, min, mul, neg, one_hot, reshape, sign, softmax, stack, std, transpose, variance, where } from 'affon:ops'

describe('affon:ops', () => {
  test('uses one lazy default Session for evaluated Tensor constructors', () => {
    const x = Tensor.from([[1, 2], [3, 4]])
    const zeros = Tensor.zeros([2, 2])
    const ones = Tensor.ones([2, 2], { dtype: 'f64' })
    const empty = Tensor.full([2, 0, 3], 7, { axes: ['batch', 'empty', 'feature'] })

    expect(add(x, zeros).to_array()).toEqual([[1, 2], [3, 4]])
    expect(ones.to_array()).toEqual([[1, 1], [1, 1]])
    expect(ones.dtype).toBe('f64')
    expect(empty.shape).toEqual([2, 0, 3])
    expect(empty.axes).toEqual(['batch', 'empty', 'feature'])
    expect((Session as any).default).toBeUndefined()
    expect((Tensor as any).setDefaultSession).toBeUndefined()
  })

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

  test('runs cross entropy as the same formal and evaluated operation', () => {
    const loss = program('ops_cross_entropy', p => cross_entropy(
      p.argument('logits', Tensor.f32([2, 2])),
      p.argument('labels', Tensor.i64([2])),
    ))
    expect(loss.inspect().nodes.at(-1)?.op).toBe('cross_entropy')

    const session = new Session()
    const logits = session.tensor([[2, 0], [0, 2]])
    const labels = session.tensor([0, 1], { dtype: 'i64' })
    const result = cross_entropy(logits, labels)
    expect(Number.isFinite(result.item())).toBe(true)
    result.dispose(); logits.dispose(); labels.dispose(); session.dispose()
  })

  test('covers reductions, selection, loss primitives, constructors, and metrics', () => {
    const source = program('expanded_ops', p => {
      const values = p.argument('values', Tensor.f32([2, 3]))
      const indices = p.argument('indices', Tensor.i64([2, 2]))
      const condition = p.argument('condition', Tensor.i64([2, 3]))
      return [
        min(values), max(values, 1), variance(values), std(values), argmax(values, 1),
        gather(values, 1, indices), one_hot(indices, 3), stack([values, values]),
        where(condition, values, sign(values)), clamp(values, -1, 1),
        mean_squared_error(values, values), mean_absolute_error(values, values),
        binary_cross_entropy_with_logits(values, values), gt_scalar(values, 0),
      ]
    })
    expect(source.inspect().nodes[source.inspect().outputs[4]].spec.dtype).toBe('i64')

    const session = new Session()
    const values = session.tensor([[1, 3, 2], [-2, 0, 4]])
    const indices = session.tensor([[2, 0], [0, 2]], { dtype: 'i64' })
    const condition = session.tensor([[1, 0, 1], [0, 1, 0]], { dtype: 'i64' })
    const outputs = session.compile(source).run({ values, indices, condition }) as readonly any[]
    expect(outputs[0].item()).toBe(-2)
    expect(outputs[4].to_array()).toEqual([1, 2])
    expect(outputs[5].to_array()).toEqual([[2, 1], [-2, 4]])
    expect(outputs[6].shape).toEqual([2, 2, 3])
    expect(outputs[7].shape).toEqual([2, 2, 3])
    expect(outputs[10].item()).toBe(0)
    expect(outputs[11].item()).toBe(0)
    expect(outputs[13].to_array()).toEqual([[1, 1, 1], [0, 0, 1]])

    const logits = session.tensor([[4, 1], [1, 4]])
    const labels = session.tensor([0, 1], { dtype: 'i64' })
    expect(metrics.accuracy(logits, labels)).toBe(1)
    expect(metrics.mean_squared_error(labels, labels)).toBe(0)
    expect(Tensor.arange(1, 5, 2).to_array()).toEqual([1, 3])
    expect(Tensor.linspace(0, 1, 3).to_array()).toEqual([0, 0.5, 1])
    expect(Tensor.rand([2], { seed: 7 }).to_array()).toEqual(Tensor.rand([2], { seed: 7 }).to_array())

    for (const output of outputs) output.dispose()
    values.dispose(); indices.dispose(); condition.dispose(); logits.dispose(); labels.dispose(); session.dispose()
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
