import { testDevices, testOn } from '../../devices.ts'
import { describe, expect, test } from 'std:test'
import { Session, Tensor, program } from 'affon:core'
import { add, reduce_to_shape } from 'affon:ops'

for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) describe(`reduce_to_shape ${device}`, () => {
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

testOn('metal')('metal keeps a scalar reduction between GPU producer and consumer', () => {
  const source = program('gpu_resident_scalar_reduction', p => {
    const input = p.argument('input', Tensor.f32([33, 5]))
    const reduced = reduce_to_shape(add(input, input), [1, 1])
    return add(reduced, reduced)
  })
  const session = new Session({ device: 'metal' })
  const executable = session.compile(source, { explanationLevel: 'detailed' })
  const input = session.tensor(Array.from({ length: 33 }, (_, row) =>
    Array.from({ length: 5 }, (_, col) => row * 5 + col)))
  const output = executable.run({ input })
  expect(output.to_array()).toEqual([[54120]])
  expect(JSON.stringify(executable.explain())).toContain('metal.reduce_to_shape.scalar_parallel')
  output.dispose(); input.dispose(); executable.dispose(); session.dispose()
})
