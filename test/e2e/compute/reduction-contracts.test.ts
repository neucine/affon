import { describe, test, expect } from 'std:test'
import { mean, min, tensor } from 'affon:compute/legacy'
import { captureError } from '../../support/errors.ts'

describe('compute reduction contracts', () => {
  test('dim reductions reject out-of-bounds dims as invalid_arg', () => {
    const x = tensor([[1, 2], [3, 4]], { dtype: 'f32' })

    const meanErr = captureError(() => mean(x, 3))
    expect(meanErr).toBeInstanceOf(Error)
    expect(meanErr.message).toContain('axis')

    const minErr = captureError(() => min(x, 3, false))
    expect(minErr).toBeInstanceOf(Error)
    expect(minErr.message).toContain('axis')
  })
})
