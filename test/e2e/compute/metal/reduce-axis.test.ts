import { describe, test, expect, values } from 'std:test'
import { mean, sum } from 'affon:compute/legacy'
import { internal_tensor } from '../../../support/compute.ts'

describe('compute metal reduce axis', () => {
  test('keeps axis reductions on metal with correct values', () => {
    const x = internal_tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32', device: 'metal' })

    const s0 = sum(x, 0, false)
    const m1 = mean(x, 1, false)

    expect(s0.device).toBe('metal')
    expect(m1.device).toBe('metal')
    expect(values(s0.to('cpu'))).toEqual([5, 7, 9])
    expect(values(m1.to('cpu'))).toEqual([2, 5])
  })
})
