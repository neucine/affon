import { describe, test, expect } from 'std:test'
import { sum, tensor, where } from 'affon:compute/legacy'
import { captureError } from '../../support/errors.ts'
import { internal_tensor } from '../../support/compute.ts'

// These tests intentionally exercise the internal autograd-facing tensor API.
// They stay on `.backward()` because they are validating substrate error modes,
// not the public parameter-centered `grad(loss, params)` surface.
describe('compute autograd contracts', () => {
  test('backward rejects tensors without a gradient graph as grad_error', () => {
    const err = captureError(() => tensor([1, 2, 3], { dtype: 'f32' }).backward())
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toContain('backward() failed')
  })

  test('backward rejects non-scalar losses as grad_error', () => {
    const err = captureError(() => internal_tensor([1, 2, 3], { dtype: 'f32' }).backward())
    expect(err.message).toContain('backward() failed')
  })

  test('unsupported backward paths surface as grad_error', () => {
    const x = internal_tensor([[1, 2], [3, 4]], { dtype: 'f32' })
    const y = sum(x.slice(['1:2', ':']))
    const err = captureError(() => y.backward())
    expect(err.message).toContain('backward() failed')
  })

  test('where condition gradient remains unsupported-by-design', () => {
    const cond = internal_tensor([[1, 0], [0, 1]], { dtype: 'f32' })
    const a = internal_tensor([[10, 20], [30, 40]], { dtype: 'f32' })
    const b = internal_tensor([[50, 60], [70, 80]], { dtype: 'f32' })
    const loss = sum(where(cond, a, b))
    const err = captureError(() => loss.backward())
    expect(err.message).toContain('backward() failed')
  })
})
