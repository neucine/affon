import { describe, test, expect, values } from 'std:test'
import checkpoint from 'affon:checkpoint'
import { tensor } from 'affon:compute/legacy'
import nn from 'affon:nn/legacy'
import { captureError } from '../../support/errors.ts'

describe('nn batchnorm', () => {
  test('updates running stats in train mode and uses them in eval mode', () => {
    const bn = nn.BatchNorm(2, { momentum: 1, eps: 1e-5 })
    const x = tensor([
      [1, 2],
      [3, 6],
    ], { dtype: 'f32' })

    const trainOut = bn(x)
    expect(values(trainOut)).toBeAllClose([
      [-0.999995, -0.99999875],
      [0.999995, 0.99999875],
    ], { rtol: 1e-5, atol: 1e-5 })

    const state = bn.state()
    expect(values(state.running_mean)).toBeAllClose([[2, 4]], { rtol: 1e-6, atol: 1e-6 })
    expect(values(state.running_var)).toBeAllClose([[1, 4]], { rtol: 1e-6, atol: 1e-6 })

    bn.eval()
    const evalOut = bn(tensor([[4, 8]], { dtype: 'f32' }))
    expect(values(evalOut)).toBeAllClose([[1.99999, 1.9999975]], { rtol: 1e-5, atol: 1e-5 })
  })

  test('save and load preserve affine parameters and running stats', () => {
    const source = nn.BatchNorm(2, { momentum: 1 })
    source(tensor([
      [1, 2],
      [3, 6],
    ], { dtype: 'f32' }))
    source.save('/tmp/affon-test-batchnorm.safetensors')

    const saved = checkpoint.load('/tmp/affon-test-batchnorm.safetensors')
    expect(Object.keys(saved).sort()).toEqual(['beta', 'gamma', 'running_mean', 'running_var'])

    const restored = nn.BatchNorm(2, { momentum: 1 })
    restored.load('/tmp/affon-test-batchnorm.safetensors')

    restored.eval()
    source.eval()
    const probe = tensor([[4, 8]], { dtype: 'f32' })
    expect(values(restored(probe))).toBeAllClose(values(source(probe)), { rtol: 1e-6, atol: 1e-6 })
  })

  test('rejects invalid eps and mismatched feature dimensions', () => {
    const epsErr = captureError(() => (nn as any).BatchNorm(2, { eps: 0 }))
    expect(epsErr).toBeInstanceOf(AffonError)
    expect(epsErr.code).toBe('invalid_arg')
    expect(epsErr.message).toContain('BatchNorm eps')

    const bn = nn.BatchNorm(2)
    const shapeErr = captureError(() => bn(tensor([[1, 2, 3]], { dtype: 'f32' })))
    expect(shapeErr).toBeInstanceOf(AffonError)
    expect(shapeErr.code).toBe('shape_mismatch')
  })
})
