import { add, cat, cross_entropy, matmul, mean, reshape, softmax, transpose } from 'affon:ops'
import { Session, Tensor, program } from 'affon:compute'
import { describe, expect, test } from 'std:test'

describe('Formal and evaluated operation parity', () => {
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
    average.dispose()
    probabilities.dispose()
    shifted.dispose()
    product.dispose()
    x.dispose()
    weight.dispose()
    session.dispose()
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
    result.dispose()
    logits.dispose()
    labels.dispose()
    session.dispose()
  })
})
