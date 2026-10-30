import { add } from 'affon:ops'
import { Session, Tensor } from 'affon:compute'
import { describe, expect, test } from 'std:test'

describe('Evaluated Tensor construction', () => {
  test('uses one lazy default Session for evaluated Tensor constructors', () => {
    const x = Tensor.from([[1, 2], [3, 4]])
    const zeros = Tensor.zeros([2, 2])
    const ones = Tensor.ones([2, 2], { dtype: 'f64' })
    const empty = Tensor.full([2, 0, 3], 7, { axes: ['batch', 'empty', 'feature'] })

    const sum = add(x, zeros)

    expect(sum.to_array()).toEqual([[1, 2], [3, 4]])
    expect(ones.to_array()).toEqual([[1, 1], [1, 1]])
    expect(ones.dtype).toBe('f64')
    expect(empty.shape).toEqual([2, 0, 3])
    expect(empty.axes).toEqual(['batch', 'empty', 'feature'])
    expect((Session as any).default).toBeUndefined()
    expect((Tensor as any).setDefaultSession).toBeUndefined()

    sum.dispose()
    x.dispose()
    zeros.dispose()
    ones.dispose()
    empty.dispose()
  })
})
