import { describe, test, expect } from 'std:test'
import nn from 'affon:nn'
import { module as computeModule, tensor } from 'affon:compute'
import { captureError } from '../../support/errors.ts'

describe('nn authoring contracts', () => {
  test('nn.module is removed from the public api', () => {
    expect((nn as any).module).toBeUndefined()
  })

  test('nn namespace no longer exposes module-level checkpoint helpers', () => {
    expect((nn as any).save).toBeUndefined()
    expect((nn as any).load).toBeUndefined()
  })

  test('nn.module_list rejects non-module entries as invalid_arg', () => {
    const listErr = captureError(() => (nn as any).module_list([{}]))
    expect(listErr).toBeInstanceOf(AffonError)
    expect(listErr.code).toBe('invalid_arg')
    expect(listErr.message).toContain('expects compute modules')

    const factoryErr = captureError(() => (nn as any).module_list(2, () => ((x: any) => x)))
    expect(factoryErr).toBeInstanceOf(AffonError)
    expect(factoryErr.code).toBe('invalid_arg')
    expect(factoryErr.message).toContain('expects compute modules')
  })

  test('nn.module_list accepts compute modules', () => {
    const mod = computeModule({ offset: tensor([1, 1], { dtype: 'f32' }) }, (state, x: any) => x)
    const list = nn.module_list([mod])
    expect(list.length).toBe(1)
  })

  test('common nn layer constructors reject invalid dimensions and options as invalid_arg', () => {
    const linearIn = captureError(() => (nn as any).Linear(0, 2))
    expect(linearIn).toBeInstanceOf(AffonError)
    expect(linearIn.code).toBe('invalid_arg')
    expect(linearIn.message).toContain('Linear in_features')

    const linearOut = captureError(() => (nn as any).Linear(2, 0))
    expect(linearOut).toBeInstanceOf(AffonError)
    expect(linearOut.code).toBe('invalid_arg')
    expect(linearOut.message).toContain('Linear out_features')

    const batchNormShape = captureError(() => (nn as any).BatchNorm(0))
    expect(batchNormShape).toBeInstanceOf(AffonError)
    expect(batchNormShape.code).toBe('invalid_arg')
    expect(batchNormShape.message).toContain('BatchNorm num_features')

    const batchNormMomentum = captureError(() => (nn as any).BatchNorm(4, { momentum: 2 }))
    expect(batchNormMomentum).toBeInstanceOf(AffonError)
    expect(batchNormMomentum.code).toBe('invalid_arg')
    expect(batchNormMomentum.message).toContain('BatchNorm momentum')

    const layerNormShape = captureError(() => (nn as any).LayerNorm(0))
    expect(layerNormShape).toBeInstanceOf(AffonError)
    expect(layerNormShape.code).toBe('invalid_arg')
    expect(layerNormShape.message).toContain('LayerNorm normalized_shape')

    const layerNormEps = captureError(() => (nn as any).LayerNorm(4, { eps: 0 }))
    expect(layerNormEps).toBeInstanceOf(AffonError)
    expect(layerNormEps.code).toBe('invalid_arg')
    expect(layerNormEps.message).toContain('LayerNorm eps')
  })

  test('built-in nn modules expose checkpoint-backed save and load methods', () => {
    const mod = nn.Linear(1, 1)
    expect(typeof mod.save).toBe('function')
    expect(typeof mod.load).toBe('function')
  })

  test('nn.diagnostics finite assertions are filterable', () => {
    try {
      nn.diagnostics.configure({ mode: 'off', include: [], exclude: [] })
      const bad = tensor([1, Number.NaN], { dtype: 'f32' })

      nn.diagnostics.assert('finite', { path: 'decoder.blocks.0.attn.q', value: bad })

      nn.diagnostics.configure({ mode: 'error', include: ['finite:decoder.blocks.*'], exclude: ['finite:decoder.blocks.0.attn.k'] })
      nn.diagnostics.assert('finite', {
        path: 'decoder.blocks.0.attn.k',
        value: bad,
      })

      const err = captureError(() => nn.diagnostics.assert('finite', {
        path: 'decoder.blocks.0.attn.q',
        value: bad,
      }))
      expect(err).toBeInstanceOf(AffonError)
      expect(err.code).toBe('grad_error')
      expect(err.message).toContain('decoder.blocks.0.attn.q produced a non-finite tensor')
      expect(err.message).toContain('first_bad_flat_index=1')
    } finally {
      nn.diagnostics.configure({ mode: 'off', include: [], exclude: [] })
    }
  })
})
