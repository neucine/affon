import { describe, test, expect } from 'std:test'
import { cat, stack, tensor } from 'affon:compute/legacy'
import { captureError } from '../../support/errors.ts'

describe('compute shape contracts', () => {
  test('cat and stack reject shape mismatches as shape_mismatch', () => {
    const catErr = captureError(() => cat([tensor([[1, 2]]), tensor([[3, 4, 5]])], 0))
    expect(catErr).toBeInstanceOf(Error)
    expect(catErr.message).toContain('cat() failed')

    const stackErr = captureError(() => stack([tensor([[1, 2]]), tensor([[3, 4, 5]])], 0))
    expect(stackErr).toBeInstanceOf(Error)
    expect(stackErr.message).toContain('stack() failed')
  })

  test('cat and stack reject mixed-device inputs as device_mismatch', () => {
    const cpu = tensor([[1, 2], [3, 4]], { dtype: 'f32' })
    const metal = tensor([[5, 6], [7, 8]], { dtype: 'f32' }).to('metal')

    const catErr = captureError(() => cat([cpu, metal], 0))
    expect(catErr).toBeInstanceOf(Error)
    expect(catErr.message).toContain('DeviceMismatch')

    const stackErr = captureError(() => stack([cpu, metal], 0))
    expect(stackErr).toBeInstanceOf(Error)
    expect(stackErr.message).toContain('DeviceMismatch')
  })

  test('slice rejects out-of-bounds numeric index entries as invalid_arg', () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]])

    for (const spec of [[2, ':'], [-3, ':']] as const) {
      const err = captureError(() => x.slice(spec as unknown as (number | string)[]))
      expect(err).toBeInstanceOf(Error)
      expect(err.message).toContain('index out of bounds')
    }
  })
})
