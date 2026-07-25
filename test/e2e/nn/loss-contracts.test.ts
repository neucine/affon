import { describe, test, expect, values } from 'std:test'
import { tensor } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import nn from 'affon:nn'
import { metalAvailable } from '../../support/metal.ts'
import { captureError } from '../../support/errors.ts'
import { internal_tensor } from '../../support/compute.ts'

// These finite-gradient checks intentionally keep direct tracked-input coverage.
// They validate loss stability for non-parameter differentiable leaves, which the
// public `grad(loss, params)` surface does not model directly today.
describe('nn loss contracts', () => {
  test('rejects non-probability inputs for BCELoss', () => {
    const prediction = tensor([[1.2], [-0.1]]) as Tensor<number[], 'f32'>
    const target = tensor([1, 0]) as Tensor<number[], 'f32'>
    const err = captureError(() => nn.BCELoss()(prediction, target))
    expect(err).toBeInstanceOf(AffonError)
    expect(err.code).toBe('invalid_arg')
    expect(err.message).toContain('expects probability inputs in the range [0, 1]')
  })

  test('rejects non-binary targets for BCE losses', () => {
    const prediction = tensor([[0.9], [0.2]]) as Tensor<number[], 'f32'>
    const logits = tensor([[2], [-1]]) as Tensor<number[], 'f32'>
    const target = tensor([1, 0.5]) as Tensor<number[], 'f32'>
    const bceErr = captureError(() => nn.BCELoss()(prediction, target))
    expect(bceErr).toBeInstanceOf(AffonError)
    expect(bceErr.code).toBe('invalid_arg')
    expect(bceErr.message).toContain('expects binary targets containing only 0 or 1')

    const logitsErr = captureError(() => nn.BCEWithLogitsLoss()(logits, target))
    expect(logitsErr).toBeInstanceOf(AffonError)
    expect(logitsErr.code).toBe('invalid_arg')
    expect(logitsErr.message).toContain('expects binary targets containing only 0 or 1')
  })

  test('rejects invalid dense targets for CrossEntropyLoss', () => {
    const logits = tensor([[2, 0, -1], [0, 1, 3]])
    const rankErr = captureError(() => nn.CrossEntropyLoss()(logits, tensor([0, 2, 1]) as Tensor<number[], 'f32'>))
    expect(rankErr).toBeInstanceOf(AffonError)
    expect(rankErr.code).toBe('invalid_shape')
    expect(rankErr.message).toContain('expects logits and targets shaped [batch, classes]')

    for (const err of [
      captureError(() => nn.CrossEntropyLoss()(logits, tensor([[1, 1, 0], [0, 0, 1]]))),
      captureError(() => nn.CrossEntropyLoss()(logits, tensor([[1, 0, 0], [0.2, 0, 0.8]]))),
    ]) {
      expect(err).toBeInstanceOf(AffonError)
      expect(err.code).toBe('invalid_arg')
      expect(err.message).toContain('expects each one-hot target row to contain exactly one 1')
    }
  })

  test('rejects BCE-family shape mismatches as shape_mismatch', () => {
    const prediction = tensor([1.2, -0.1], { dtype: 'f32' })
    const logits = tensor([2, -1], { dtype: 'f32' })
    const target = tensor([[1, 0]], { dtype: 'f32' })

    const bceErr = captureError(() => nn.BCELoss()(prediction, target))
    expect(bceErr).toBeInstanceOf(AffonError)
    expect(bceErr.code).toBe('shape_mismatch')
    expect(bceErr.message).toContain('same shape after alignment')

    const logitsErr = captureError(() => nn.BCEWithLogitsLoss()(logits, target))
    expect(logitsErr).toBeInstanceOf(AffonError)
    expect(logitsErr.code).toBe('shape_mismatch')
    expect(logitsErr.message).toContain('same shape after alignment')
  })

  test('keeps BCE finite for saturated probabilities', () => {
    const prediction = internal_tensor([[1], [0], [0.9999999], [0.0000001]], { dtype: 'f32' }) as Tensor<number[], 'f32'>
    const target = tensor([1, 0, 1, 0]) as Tensor<number[], 'f32'>
    const loss = nn.BCELoss()(prediction, target)

    expect(Number.isFinite(loss.item())).toBe(true)

    loss.backward()

    const grad = values(prediction.grad) as number[][]
    expect(grad.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)
  })

  test('keeps BCEWithLogitsLoss finite for extreme logits', () => {
    const logits = internal_tensor([[1000], [-1000], [80], [-80]], { dtype: 'f32' }) as Tensor<number[], 'f32'>
    const target = tensor([1, 0, 1, 0]) as Tensor<number[], 'f32'>
    const loss = nn.BCEWithLogitsLoss()(logits, target)

    expect(Number.isFinite(loss.item())).toBe(true)

    loss.backward()

    const grad = values(logits.grad) as number[][]
    expect(grad.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)
  })

  test('keeps CrossEntropyLoss finite for extreme logits', () => {
    const logits = internal_tensor([
      [1000, -1000, -1000],
      [-1000, 1000, -1000],
    ], { dtype: 'f32' })
    const targets = tensor([
      [1, 0, 0],
      [0, 1, 0],
    ], { dtype: 'f32' })

    const loss = nn.CrossEntropyLoss()(logits, targets)
    expect(Number.isFinite(loss.item())).toBe(true)

    loss.backward()

    const grad = values(logits.grad) as number[][]
    expect(grad.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)
  })

  test('keeps MSELoss finite for large but safe finite values', () => {
    const prediction = internal_tensor([[1e10], [-1e10], [1e8]], { dtype: 'f32' }) as Tensor<number[], 'f32'>
    const target = tensor([[1e10 - 1e5], [-1e10 + 1e5], [1e8 + 1e4]], { dtype: 'f32' }) as Tensor<number[], 'f32'>

    const loss = nn.MSELoss()(prediction, target)
    expect(Number.isFinite(loss.item())).toBe(true)

    loss.backward()
    const grad = values(prediction.grad) as number[][]
    expect(grad.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)
  })

  test.skip(() => !metalAvailable())('keeps BCEWithLogitsLoss finite on metal for extreme logits', () => {
      const logits = internal_tensor([[1000], [-1000], [80], [-80]], { dtype: 'f32', device: 'metal' }) as Tensor<number[], 'f32'>
    const target = tensor([1, 0, 1, 0], { dtype: 'f32' }).to('metal') as Tensor<number[], 'f32'>
    const loss = nn.BCEWithLogitsLoss()(logits, target)

    expect(Number.isFinite(loss.to('cpu').item())).toBe(true)

    loss.backward()

    const grad = values(logits.grad?.to('cpu')) as number[][]
    expect(grad.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)
  })

  test.skip(() => !metalAvailable())('keeps CrossEntropyLoss finite on metal for extreme logits', () => {
      const logits = internal_tensor([
        [1000, -1000, -1000],
        [-1000, 1000, -1000],
      ], { dtype: 'f32', device: 'metal' })
    const targets = tensor([
      [1, 0, 0],
      [0, 1, 0],
    ], { dtype: 'f32' }).to('metal')

    const loss = nn.CrossEntropyLoss()(logits, targets)
    expect(Number.isFinite(loss.to('cpu').item())).toBe(true)

    loss.backward()

    const grad = values(logits.grad?.to('cpu')) as number[][]
    expect(grad.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)
  })
})
