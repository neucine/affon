import { describe, expect, test } from 'std:test'
import { Session, Tensor, program } from 'affon:compute'
import { index_select } from 'affon:ops'

for (const device of ['cpu', 'metal'] as const) describe(`batched index_select ${device}`, () => {
  test('selects an inner axis within each batch', () => {
    const session = new Session({ device })
    const input = session.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    const indices = session.tensor([2, 0], { dtype: 'i64' })
    const source = program('index_select_batched', p => index_select(
      p.argument('input', Tensor.f32([2, 2, 3])), 2,
      p.argument('indices', Tensor.i64([2]))))
    const executable = session.compile(source)
    const output = executable.run({ input, indices })
    expect(output.to_array()).toEqual([[[3, 1], [6, 4]], [[9, 7], [12, 10]]])
    output.dispose(); executable.dispose(); indices.dispose(); input.dispose(); session.dispose()
  })
})
