import { testDevices } from '../../devices.ts'
import { Session, Tensor, program } from 'affon:core'
import { dot, matmul, transpose } from 'affon:ops'
import { describe, expect, test } from 'std:test'

describe('Matrix and dot operation contracts', () => {
  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) {
    test(`batched matrix multiplication keeps batches independent on ${device}`, () => {
      const source = program(`batched_matmul_${device}`, p => matmul(
        p.argument('left', Tensor.f32([2, 2, 3])),
        p.argument('right', Tensor.f32([2, 3, 2])),
      ))
      const session = new Session({ device })
      const left = session.tensor([[[1, 2, 3], [4, 5, 6]], [[2, 0, 1], [0, 1, 2]]])
      const right = session.tensor([[[1, 0], [0, 1], [1, 1]], [[3, 1], [2, 4], [1, 5]]])
      const expected = [[[4, 5], [10, 11]], [[7, 7], [4, 14]]]
      const immediate = matmul(left, right)
      const compiled = session.compile(source).run({ left, right })
      expect(immediate.to_array()).toEqual(expected)
      expect(compiled.to_array()).toEqual(expected)
      immediate.dispose(); compiled.dispose(); left.dispose(); right.dispose(); session.dispose()
    })

    test(`transposed inputs and dot use actual strides on ${device}`, () => {
      const session = new Session({ device })
      const base = session.tensor([[1, 2, 3], [4, 5, 6]])
      const transposed = transpose(base, [1, 0])
      const product = matmul(transposed, base)
      expect(product.to_array()).toEqual([[17, 22, 27], [22, 29, 36], [27, 36, 45]])
      const a = session.tensor([2, -1, 3])
      const b = session.tensor([4, 5, -2])
      const scalar = dot(a, b)
      expect(scalar.item()).toBe(-3)
      scalar.dispose(); a.dispose(); b.dispose(); product.dispose(); transposed.dispose(); base.dispose(); session.dispose()
    })
  }

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`invalid inner dimensions reject before a valid multiply in the same Session on ${device}`, () => {
    const session = new Session({ device })
    const left = session.tensor([[1, 2, 3]])
    const wrong = session.tensor([[1, 2]])
    expect(() => matmul(left, wrong)).toThrow()
    const right = session.tensor([[1], [2], [3]])
    const result = matmul(left, right)
    expect(result.to_array()).toEqual([[14]])
    result.dispose(); right.dispose(); wrong.dispose(); left.dispose(); session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`dot rejects length mismatch but keeps the vectors usable on ${device}`, () => {
    const session = new Session({ device })
    const left = session.tensor([1, 2, 3])
    const wrong = session.tensor([1, 2])
    expect(() => dot(left, wrong)).toThrow()
    const valid = session.tensor([3, 2, 1])
    const result = dot(left, valid)
    expect(result.item()).toBe(10)
    result.dispose(); valid.dispose(); wrong.dispose(); left.dispose(); session.dispose()
  })
})
