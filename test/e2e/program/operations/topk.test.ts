import { describe, expect, test } from 'std:test'
import { Session, Tensor, program } from 'affon:compute'
import { topk } from 'affon:ops'

for (const device of ['cpu', 'metal'] as const) describe(`topk ${device}`, () => {
  test('returns values and indices in immediate and Program calls', () => {
    const session = new Session({ device })
    const input = session.tensor([[1, 4, 3, 2], [8, 5, 7, 6]])
    const immediate = topk(input, 2, 1)
    expect(immediate[0].to_array()).toEqual([[4, 3], [8, 7]])
    expect(immediate[1].to_array()).toEqual([[1, 2], [0, 2]])
    const source = program('topk_contract', p => topk(p.argument('input', Tensor.f32([2, 4])), 2, 1))
    const executable = session.compile(source)
    const output = executable.run({ input })
    expect(output[0].to_array()).toEqual([[4, 3], [8, 7]])
    expect(output[1].to_array()).toEqual([[1, 2], [0, 2]])
    output.forEach(value => value.dispose()); executable.dispose()
    immediate.forEach(value => value.dispose()); input.dispose(); session.dispose()
  })

  test('rejects k beyond the selected axis', () => {
    expect(() => program('invalid_topk', p => topk(p.argument('x', Tensor.f32([2, 4])), 5, 1))).toThrow()
  })

  test('supports BF16 and the outer axis', () => {
    const session = new Session({ device })
    const input = session.tensor([[1, 4, 3], [8, 5, 7], [2, 9, 6]], { dtype: 'bf16' })
    const source = program('topk_bf16_outer', p => topk(p.argument('input', Tensor.bf16([3, 3])), 2, 0))
    const executable = session.compile(source)
    const [values, indices] = executable.run({ input })
    expect(values.dtype).toBe('bf16')
    expect(indices.dtype).toBe('i64')
    expect(values.to_array()).toEqual([[8, 9, 7], [2, 5, 6]])
    expect(indices.to_array()).toEqual([[1, 2, 1], [2, 1, 2]])
    values.dispose(); indices.dispose(); executable.dispose(); input.dispose(); session.dispose()
  })

  test('selects within rank-three batches', () => {
    const session = new Session({ device })
    const input = session.tensor([[[1, 4, 3], [8, 5, 7]], [[2, 9, 6], [4, 3, 5]]])
    const source = program('topk_rank_three', p => topk(p.argument('input', Tensor.f32([2, 2, 3])), 2, 2))
    const executable = session.compile(source)
    const [values, indices] = executable.run({ input })
    expect(values.to_array()).toEqual([[[4, 3], [8, 7]], [[9, 6], [5, 4]]])
    expect(indices.to_array()).toEqual([[[1, 2], [0, 2]], [[1, 2], [2, 0]]])
    values.dispose(); indices.dispose(); executable.dispose(); input.dispose(); session.dispose()
  })
})
