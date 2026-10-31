import { add, slice } from 'affon:ops'
import { Session, Tensor } from 'affon:compute'
import { describe, expect, test } from 'std:test'

describe('Evaluated Tensor construction', () => {
  test('reads dense and strided tensors as row major bytes', () => {
    const session = new Session({ device: 'cpu' })
    const source = session.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    const view = slice(source, [{ start: 0, stop: 2 }, { start: 1, stop: 4, step: 2 }])
    const denseBytes = source.to_bytes()
    const viewBytes = view.to_bytes()
    expect(Array.from(new Float32Array(denseBytes))).toEqual([1, 2, 3, 4, 5, 6, 7, 8])
    expect(Array.from(new Float32Array(viewBytes))).toEqual([2, 4, 6, 8])
    const copied = session.fromBytes(viewBytes, Tensor.f32([2, 2]))
    expect(copied.to_array()).toEqual([[2, 4], [6, 8]])
    const bf16 = session.tensor([1, 2], { dtype: 'bf16' })
    const bf16Copy = session.fromBytes(bf16.to_bytes(), Tensor.bf16([2]))
    expect(bf16Copy.to_array()).toEqual([1, 2])
    bf16Copy.dispose()
    bf16.dispose()
    copied.dispose()
    view.dispose()
    source.dispose()
    session.dispose()
  })

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
