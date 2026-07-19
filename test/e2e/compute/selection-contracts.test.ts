import { describe, test, expect } from 'std:test'
import { gather, index_select, one_hot, tensor, topk } from 'affon:compute'
import { captureError } from '../../support/errors.ts'

describe('compute selection contracts', () => {
  test('gather and index_select reject out-of-bounds indices as invalid_arg', () => {
    const x = tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32' })

    const gatherErr = captureError(() => gather(x, 1, tensor([[2, 3], [1, 1]])))
    expect(gatherErr).toBeInstanceOf(Error)
    expect(gatherErr.message).toContain('IndexOutOfBounds')

    const selectErr = captureError(() => index_select(x, 0, tensor([0, 2])))
    expect(selectErr).toBeInstanceOf(Error)
    expect(selectErr.message).toContain('IndexOutOfBounds')
  })

  test('gather, index_select, and one_hot reject non-integral or non-finite indices as invalid_arg', () => {
    const x = tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32' })

    for (const err of [
      captureError(() => gather(x, 1, tensor([[1.5, 0], [1, 1]]))),
      captureError(() => gather(x, 1, tensor([[Number.NaN, 0], [1, 1]]))),
      captureError(() => index_select(x, 0, tensor([0.5, 1]))),
      captureError(() => index_select(x, 0, tensor([Number.POSITIVE_INFINITY, 1]))),
      captureError(() => one_hot(tensor([0.5, 1]), 3)),
      captureError(() => one_hot(tensor([Number.NaN, 1]), 3)),
    ]) {
      expect(err).toBeInstanceOf(Error)
      expect(err.message).toContain('InvalidArgument')
    }
  })

  test('topk rejects invalid axis and invalid k as invalid_arg', () => {
    const x = tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32' })

    const axisErr = captureError(() => topk(x, 1, 3))
    expect(axisErr).toBeInstanceOf(Error)
    expect(axisErr.message).toContain('InvalidAxis')

    const kErr = captureError(() => topk(x, 4, 1))
    expect(kErr).toBeInstanceOf(Error)
    expect(kErr.message).toContain('InvalidTopK')
  })
})
