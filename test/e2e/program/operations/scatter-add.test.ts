import { describe, expect, test } from 'std:test'
import { Session, Tensor, program } from 'affon:compute'
import { scatter_add } from 'affon:ops'

for (const device of ['cpu', 'metal'] as const) describe(`scatter_add ${device}`, () => {
  test('accumulates colliding indices through immediate and Program calls', () => {
    const session = new Session({ device })
    const base = session.tensor([[1, 2, 3], [4, 5, 6]])
    const indices = session.tensor([[0, 1, 1], [1, 0, 1]], { dtype: 'i64' })
    const updates = session.tensor([[1, 1, 1], [2, 2, 2]])
    const immediate = scatter_add(base, indices, updates, 0)
    expect(immediate.to_array()).toEqual([[2, 4, 3], [6, 6, 9]])
    const source = program('scatter_add_contract', p => scatter_add(
      p.argument('base', Tensor.f32([2, 3])), p.argument('indices', Tensor.i64([2, 3])),
      p.argument('updates', Tensor.f32([2, 3])), 0))
    const executable = session.compile(source, { explanationLevel: 'detailed' })
    const output = executable.run({ base, indices, updates })
    expect(output.to_array()).toEqual([[2, 4, 3], [6, 6, 9]])
    expect(JSON.stringify(executable.explain()).includes('metal_primitive')).toBe(device === 'metal')
    output.dispose(); executable.dispose(); immediate.dispose(); updates.dispose(); indices.dispose(); base.dispose(); session.dispose()
  })

  test('rejects an index shape that differs outside the scatter axis', () => {
    const source = () => program('invalid_scatter', p => scatter_add(
      p.argument('base', Tensor.f32([2, 3])), p.argument('indices', Tensor.i64([1, 2])),
      p.argument('updates', Tensor.f32([1, 2])), 0))
    expect(source).toThrow()
  })

  test('accumulates independently within rank-three batches', () => {
    const session = new Session({ device })
    const base = session.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    const indices = session.tensor([[[1, 0]], [[0, 1]]], { dtype: 'i64' })
    const updates = session.tensor([[[2, 3]], [[4, 5]]])
    const source = program('scatter_add_batched', p => scatter_add(
      p.argument('base', Tensor.f32([2, 2, 2])),
      p.argument('indices', Tensor.i64([2, 1, 2])),
      p.argument('updates', Tensor.f32([2, 1, 2])), 1))
    const executable = session.compile(source)
    const output = executable.run({ base, indices, updates })
    expect(output.to_array()).toEqual([[[1, 5], [5, 4]], [[9, 6], [7, 13]]])
    output.dispose(); executable.dispose(); updates.dispose(); indices.dispose(); base.dispose(); session.dispose()
  })
})
