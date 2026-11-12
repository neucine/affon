import { testDevices } from '../../devices.ts'
import { Session, Tensor, program } from 'affon:core'
import { add, cat, contiguous, matmul, permute, reshape, slice, squeeze, stack, transpose, unsqueeze } from 'affon:ops'
import { describe, expect, test } from 'std:test'

describe('Layout and shape operation contracts', () => {
  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) {
    test(`sliced offset and transposed views feed arithmetic on ${device}`, () => {
      const session = new Session({ device })
      const base = session.tensor([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]])
      const view = slice(base, [{ start: 1, stop: 3 }, { start: 1, stop: 4 }])
      const transposed = transpose(view, [1, 0])
      const copy = contiguous(transposed)
      const bias = session.tensor([[1, 2]])
      const shifted = add(copy, bias)
      expect(view.to_array()).toEqual([[11, 12, 13], [21, 22, 23]])
      expect(transposed.to_array()).toEqual([[11, 21], [12, 22], [13, 23]])
      expect(copy.to_array()).toEqual([[11, 21], [12, 22], [13, 23]])
      expect(shifted.to_array()).toEqual([[12, 23], [13, 24], [14, 25]])
      shifted.dispose(); bias.dispose(); copy.dispose(); transposed.dispose(); view.dispose(); base.dispose(); session.dispose()
    })

    test(`compiled layout pipeline reads different inputs on ${device}`, () => {
      const source = program(`layout_pipeline_${device}`, p => {
        const input = p.argument('input', Tensor.f32([2, 3]))
        const cut = slice(input, [{ start: 0, stop: 2 }, { start: 1, stop: 3 }])
        return matmul(contiguous(transpose(cut, [1, 0])), cut)
      })
      const session = new Session({ device })
      const executable = session.compile(source)
      const firstInput = session.tensor([[1, 2, 3], [4, 5, 6]])
      const first = executable.run({ input: firstInput })
      expect(first.to_array()).toEqual([[29, 36], [36, 45]])
      const secondInput = session.tensor([[1, 0, 1], [2, 1, 0]])
      const second = executable.run({ input: secondInput })
      expect(second.to_array()).toEqual([[1, 0], [0, 1]])
      first.dispose(); second.dispose(); firstInput.dispose(); secondInput.dispose(); session.dispose()
    })
  }

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`rank-three permutation maps every coordinate without flattening on ${device}`, () => {
    const session = new Session({ device })
    const input = session.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    const result = permute(input, [2, 0, 1])
    expect(result.shape).toEqual([2, 2, 3])
    expect(result.to_array()).toEqual([[[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]])
    result.dispose(); input.dispose(); session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`cat, stack, squeeze, and unsqueeze preserve order and rank on ${device}`, () => {
    const session = new Session({ device })
    const a = session.tensor([[1, 2]])
    const b = session.tensor([[3, 4]])
    const joined = cat([a, b], 0)
    const grouped = stack([a, b], 0)
    const expanded = unsqueeze(joined, -1)
    const restored = squeeze(expanded, -1)
    expect(joined.to_array()).toEqual([[1, 2], [3, 4]])
    expect(grouped.to_array()).toEqual([[[1, 2]], [[3, 4]]])
    expect(expanded.to_array()).toEqual([[[1], [2]], [[3], [4]]])
    expect(restored.to_array()).toEqual(joined.to_array())
    restored.dispose(); expanded.dispose(); grouped.dispose(); joined.dispose(); a.dispose(); b.dispose(); session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`invalid shape requests do not poison later view operations on ${device}`, () => {
    const session = new Session({ device })
    const input = session.tensor([[1, 2, 3], [4, 5, 6]])
    expect(() => reshape(input, [5])).toThrow('preserve element count')
    expect(() => transpose(input, [0, 0])).toThrow()
    expect(() => squeeze(input, 0)).toThrow()
    const good = reshape(input, [3, 2])
    expect(good.to_array()).toEqual([[1, 2], [3, 4], [5, 6]])
    good.dispose(); input.dispose(); session.dispose()
  })
})
