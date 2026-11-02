import { Session, Tensor, program } from 'affon:compute'
import { add, argmax, embedding, gather, index_select, masked_fill, one_hot, reshape, where } from 'affon:ops'
import { describe, expect, test } from 'std:test'

describe('Selection operation contracts', () => {
  for (const device of ['cpu', 'metal'] as const) {
    test(`gather uses per-row indices and repeated compiled runs on ${device}`, () => {
      const source = program(`gather_recovery_${device}`, p => gather(
        p.argument('input', Tensor.f32([2, 3])), -1,
        p.argument('indices', Tensor.i64([2, 2])),
      ))
      const session = new Session({ device })
      const executable = session.compile(source)
      const input = session.tensor([[10, 20, 30], [40, 50, 60]])
      const indices = session.tensor([[2, 0], [1, 1]], { dtype: 'i64' })
      const first = executable.run({ input, indices })
      expect(first.to_array()).toEqual([[30, 10], [50, 50]])
      const second = executable.run({ input, indices })
      expect(second.to_array()).toEqual(first.to_array())
      first.dispose(); second.dispose(); input.dispose(); indices.dispose(); session.dispose()
    })

    test(`index_select repeats positions along an interior axis on ${device}`, () => {
      const session = new Session({ device })
      const input = session.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
      const indices = session.tensor([2, 0, 2], { dtype: 'i64' })
      const selected = index_select(input, 1, indices)
      expect(selected.to_array()).toEqual([[[5, 6], [1, 2], [5, 6]], [[11, 12], [7, 8], [11, 12]]])
      selected.dispose(); indices.dispose(); input.dispose(); session.dispose()
    })

    test(`embedding reads repeated rows without changing their order on ${device}`, () => {
      const source = program(`embedding_rows_${device}`, p => embedding(
        p.argument('table', Tensor.f32([4, 2])),
        p.argument('indices', Tensor.i64([2, 3])),
      ))
      const session = new Session({ device })
      const table = session.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
      const indices = session.tensor([[3, 0, 3], [1, 2, 0]], { dtype: 'i64' })
      const expected = [[[7, 8], [1, 2], [7, 8]], [[3, 4], [5, 6], [1, 2]]]
      const direct = embedding(table, indices)
      const compiled = session.compile(source).run({ table, indices })
      expect(direct.to_array()).toEqual(expected)
      expect(compiled.to_array()).toEqual(expected)
      direct.dispose(); compiled.dispose(); table.dispose(); indices.dispose(); session.dispose()
    })
  }

  for (const device of ['cpu', 'metal'] as const) test(`${device} gather rejects an invalid index and reuses the compiled Program`, () => {
    const source = program(`${device}_gather_bounds`, p => gather(
      p.argument('input', Tensor.f32([2, 3])), 1,
      p.argument('indices', Tensor.i64([2, 2])),
    ))
    const session = new Session({ device })
    const executable = session.compile(source)
    const input = session.tensor([[10, 20, 30], [40, 50, 60]])
    const invalid = session.tensor([[2, 3], [1, 0]], { dtype: 'i64' })
    const valid = session.tensor([[2, 0], [1, 1]], { dtype: 'i64' })
    expect(() => executable.run({ input, indices: invalid })).toThrow('InvalidCommand')
    const recovered = executable.run({ input, indices: valid })
    expect(recovered.to_array()).toEqual([[30, 10], [50, 50]])
    recovered.dispose(); valid.dispose(); invalid.dispose(); input.dispose(); session.dispose()
  })

  for (const device of ['cpu', 'metal'] as const) test(`${device} embedding rejects an invalid row and still reads a valid row`, () => {
    const session = new Session({ device })
    const table = session.tensor([[1, 2], [3, 4]])
    const invalid = session.tensor([2], { dtype: 'i64' })
    const valid = session.tensor([1], { dtype: 'i64' })
    expect(() => embedding(table, invalid)).toThrow('InvalidCommand')
    const result = embedding(table, valid)
    expect(result.to_array()).toEqual([[3, 4]])
    result.dispose(); valid.dispose(); invalid.dispose(); table.dispose(); session.dispose()
  })

  for (const device of ['cpu', 'metal'] as const) test(`${device} index_select rejects an invalid position and reuses its input`, () => {
    const session = new Session({ device })
    const input = session.tensor([[1, 2], [3, 4]])
    const invalid = session.tensor([2], { dtype: 'i64' })
    const valid = session.tensor([1], { dtype: 'i64' })
    expect(() => index_select(input, 0, invalid)).toThrow('InvalidCommand')
    const result = index_select(input, 0, valid)
    expect(result.to_array()).toEqual([[3, 4]])
    result.dispose(); valid.dispose(); invalid.dispose(); input.dispose(); session.dispose()
  })

  test('metal reports invalid GPU-produced embedding indices and recovers', () => {
    const source = program('gpu_index_bounds', p => {
      const selected = embedding(
        p.argument('table', Tensor.f32([4, 2])),
        add(argmax(p.argument('scores', Tensor.f32([2, 4])), 1), p.argument('offset', Tensor.i64([2]))),
      )
      return add(selected, selected)
    })
    const session = new Session({ device: 'metal' })
    const executable = session.compile(source)
    const table = session.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    const scores = session.tensor([[1, 2, 3, 9], [1, 8, 3, 2]])
    const invalidOffset = session.tensor([1, 4], { dtype: 'i64' })
    const validOffset = session.tensor([0, 0], { dtype: 'i64' })
    expect(() => executable.run({ table, scores, offset: invalidOffset })).toThrow('InvalidCommand')
    const recovered = executable.run({ table, scores, offset: validOffset })
    expect(recovered.to_array()).toEqual([[14, 16], [6, 8]])
    recovered.dispose(); validOffset.dispose(); invalidOffset.dispose(); scores.dispose(); table.dispose(); executable.dispose(); session.dispose()
  })

  test('metal reports invalid GPU-produced index_select indices and recovers', () => {
    const source = program('gpu_index_select_bounds', p => index_select(
      p.argument('table', Tensor.f32([4, 2])), 0,
      add(argmax(p.argument('scores', Tensor.f32([2, 4])), 1), p.argument('offset', Tensor.i64([2]))),
    ))
    const session = new Session({ device: 'metal' })
    const executable = session.compile(source)
    const table = session.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    const scores = session.tensor([[1, 2, 3, 9], [1, 8, 3, 2]])
    const invalidOffset = session.tensor([1, 4], { dtype: 'i64' })
    const validOffset = session.tensor([0, 0], { dtype: 'i64' })
    expect(() => executable.run({ table, scores, offset: invalidOffset })).toThrow('InvalidCommand')
    const recovered = executable.run({ table, scores, offset: validOffset })
    expect(recovered.to_array()).toEqual([[7, 8], [3, 4]])
    recovered.dispose(); validOffset.dispose(); invalidOffset.dispose(); scores.dispose(); table.dispose(); executable.dispose(); session.dispose()
  })

  test('metal reports invalid GPU-produced gather indices and recovers', () => {
    const source = program('gpu_gather_bounds', p => gather(
      p.argument('table', Tensor.f32([2, 4])), 1,
      reshape(add(argmax(p.argument('scores', Tensor.f32([2, 4])), 1), p.argument('offset', Tensor.i64([2]))), [2, 1]),
    ))
    const session = new Session({ device: 'metal' })
    const executable = session.compile(source)
    const table = session.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    const scores = session.tensor([[1, 2, 3, 9], [1, 8, 3, 2]])
    const invalidOffset = session.tensor([1, 4], { dtype: 'i64' })
    const validOffset = session.tensor([0, 0], { dtype: 'i64' })
    expect(() => executable.run({ table, scores, offset: invalidOffset })).toThrow('InvalidCommand')
    const recovered = executable.run({ table, scores, offset: validOffset })
    expect(recovered.to_array()).toEqual([[4], [6]])
    recovered.dispose(); validOffset.dispose(); invalidOffset.dispose(); scores.dispose(); table.dispose(); executable.dispose(); session.dispose()
  })

  test('one_hot retains repeated labels and rejects out-of-range labels', () => {
    const session = new Session({ device: 'cpu' })
    const indices = session.tensor([[2, 0, 2], [1, 1, 0]], { dtype: 'i64' })
    const encoded = one_hot(indices, 3)
    expect(encoded.to_array()).toEqual([
      [[0, 0, 1], [1, 0, 0], [0, 0, 1]],
      [[0, 1, 0], [0, 1, 0], [1, 0, 0]],
    ])
    const invalid = session.tensor([3], { dtype: 'i64' })
    expect(() => one_hot(invalid, 3)).toThrow()
    const again = one_hot(indices, 3)
    expect(again.to_array()).toEqual(encoded.to_array())
    again.dispose(); invalid.dispose(); encoded.dispose(); indices.dispose(); session.dispose()
  })

  test('where and masked_fill broadcast a mask over feature values', () => {
    const session = new Session({ device: 'cpu' })
    const data = session.tensor([[1, 2, 3], [4, 5, 6]])
    const alternative = session.tensor([[-1, -2, -3]])
    const mask = session.tensor([[1], [0]], { dtype: 'i64' })
    const chosen = where(mask, data, alternative)
    const filled = masked_fill(data, mask, 99)
    expect(chosen.to_array()).toEqual([[1, 2, 3], [-1, -2, -3]])
    expect(filled.to_array()).toEqual([[99, 99, 99], [4, 5, 6]])
    chosen.dispose(); filled.dispose(); mask.dispose(); alternative.dispose(); data.dispose(); session.dispose()
  })

  test('invalid mask broadcasting leaves operands usable', () => {
    const session = new Session({ device: 'cpu' })
    const data = session.tensor([[1, 2, 3], [4, 5, 6]])
    const invalid = session.tensor([1, 0, 1, 0], { dtype: 'i64' })
    const valid = session.tensor([0, 1, 0], { dtype: 'i64' })
    expect(() => masked_fill(data, invalid, 0)).toThrow('cannot broadcast')
    const result = masked_fill(data, valid, 0)
    expect(result.to_array()).toEqual([[1, 0, 3], [4, 0, 6]])
    result.dispose(); valid.dispose(); invalid.dispose(); data.dispose(); session.dispose()
  })
})
