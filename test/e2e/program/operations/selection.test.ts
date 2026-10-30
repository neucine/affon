// Cases adapted from affon-legacy/test/parity/compute selection families.
import { gather, index_select, masked_fill, one_hot, where } from 'affon:ops'
import { Session } from 'affon:compute'
import { describe, expect, test } from 'std:test'

describe('Selection reference values', () => {
  test('gather and index selection preserve repeated indices and order', () => {
    const session = new Session({ device: 'cpu' })
    const values = session.tensor([[10, 20, 30], [40, 50, 60]])
    const gatherIndices = session.tensor([[2, 0, 2], [1, 1, 0]], { dtype: 'i64' })
    const selectIndices = session.tensor([2, 0, 2], { dtype: 'i64' })
    const gathered = gather(values, 1, gatherIndices)
    const selected = index_select(values, 1, selectIndices)
    expect(gathered.to_array()).toEqual([[30, 10, 30], [50, 50, 40]])
    expect(selected.to_array()).toEqual([[30, 10, 30], [60, 40, 60]])
    gathered.dispose(); selected.dispose(); values.dispose(); gatherIndices.dispose(); selectIndices.dispose(); session.dispose()
  })

  test('masking, conditional selection, and one-hot match reference values', () => {
    const session = new Session({ device: 'cpu' })
    const values = session.tensor([[1, 2, 3], [4, 5, 6]])
    const mask = session.tensor([[1, 0, 1], [0, 1, 0]], { dtype: 'i64' })
    const other = session.tensor([[9, 9, 9], [9, 9, 9]])
    const indices = session.tensor([2, 0, 1], { dtype: 'i64' })
    const filled = masked_fill(values, mask, -1)
    const chosen = where(mask, values, other)
    const encoded = one_hot(indices, 3)
    expect(filled.to_array()).toEqual([[-1, 2, -1], [4, -1, 6]])
    expect(chosen.to_array()).toEqual([[1, 9, 3], [9, 5, 9]])
    expect(encoded.to_array()).toEqual([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    filled.dispose(); chosen.dispose(); encoded.dispose(); values.dispose(); mask.dispose(); other.dispose(); indices.dispose(); session.dispose()
  })
})
