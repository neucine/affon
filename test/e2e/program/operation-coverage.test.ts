import { argmax, binary_cross_entropy_with_logits, clamp, gather, gt_scalar, max, mean_absolute_error, mean_squared_error, min, one_hot, sign, stack, std, variance, where } from 'affon:ops'
import { Session, Tensor, metrics, program } from 'affon:compute'
import { describe, expect, test } from 'std:test'

describe('Operation family coverage', () => {
  test('covers reductions, selection, and loss primitives', () => {
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

    for (const output of outputs) output.dispose()
    values.dispose()
    indices.dispose()
    condition.dispose()
    session.dispose()
  })

  test('covers evaluated metrics and Tensor constructors', () => {
    const session = new Session()
    const logits = session.tensor([[4, 1], [1, 4]])
    const labels = session.tensor([0, 1], { dtype: 'i64' })
    const range = Tensor.arange(1, 5, 2)
    const linear = Tensor.linspace(0, 1, 3)
    const firstRandom = Tensor.rand([2], { seed: 7 })
    const secondRandom = Tensor.rand([2], { seed: 7 })

    expect(metrics.accuracy(logits, labels)).toBe(1)
    expect(metrics.mean_squared_error(labels, labels)).toBe(0)
    expect(range.to_array()).toEqual([1, 3])
    expect(linear.to_array()).toEqual([0, 0.5, 1])
    expect(firstRandom.to_array()).toEqual(secondRandom.to_array())

    range.dispose()
    linear.dispose()
    firstRandom.dispose()
    secondRandom.dispose()
    logits.dispose()
    labels.dispose()
    session.dispose()
  })
})
