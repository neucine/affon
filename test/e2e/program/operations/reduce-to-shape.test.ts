import { describe, expect, test } from 'std:test'
import { Session, Tensor, program } from 'affon:compute'
import { reduce_to_shape } from 'affon:ops'

for (const device of ['cpu', 'metal'] as const) describe(`reduce_to_shape ${device}`, () => {
  test('sums axes through immediate and Program calls', () => {
    const session = new Session({ device })
    const input = session.tensor([[1, 2, 3], [4, 5, 6]])
    const immediate = reduce_to_shape(input, [1, 3])
    expect(immediate.to_array()).toEqual([[5, 7, 9]])
    const source = program('reduce_shape_contract', p => reduce_to_shape(
      p.argument('input', Tensor.f32([2, 3])), [1, 3]))
    const executable = session.compile(source, { explanationLevel: 'detailed' })
    const output = executable.run({ input })
    expect(output.to_array()).toEqual([[5, 7, 9]])
    output.dispose(); executable.dispose(); immediate.dispose(); input.dispose(); session.dispose()
  })

  test('rejects incompatible target dimensions', () => {
    expect(() => program('invalid_reduce_shape', p => reduce_to_shape(
      p.argument('input', Tensor.f32([2, 3])), [2, 2]))).toThrow()
  })

  test('reduces a matrix to a scalar output', () => {
    const session = new Session({ device })
    const input = session.tensor([[1, 2, 3], [4, 5, 6]])
    const source = program('reduce_shape_scalar', p => reduce_to_shape(
      p.argument('input', Tensor.f32([2, 3])), []))
    const executable = session.compile(source)
    const output = executable.run({ input })
    expect(output.shape).toEqual([])
    expect(output.to_array()).toBe(21)
    output.dispose(); executable.dispose(); input.dispose(); session.dispose()
  })
})
