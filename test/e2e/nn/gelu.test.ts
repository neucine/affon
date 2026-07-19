import { describe, test, expect, values } from 'std:test'
import { gelu, grad, tensor } from 'affon:compute'
import nn from 'affon:nn'
import { internal_tensor } from '../../support/compute.ts'

function roundNested(value: any): any {
  if (Array.isArray(value)) return value.map(roundNested)
  return Math.round(value * 1e6) / 1e6
}

describe('nn gelu', () => {
  test('applies the tanh-approximate GELU elementwise', () => {
    const x = tensor([[-1, 0, 1]])
    const y = gelu(x)

    expect(y.shape).toEqual([1, 3])
    expect(roundNested(values(y))).toEqual([[-0.158808, 0, 0.841192]])
  })

  test('supports autograd through the activation', () => {
    const x = internal_tensor([[-1, 0, 1]], { dtype: 'f32' })
    const loss = gelu(x).sum()
    grad(loss, [x as any])

    expect(roundNested(values(x.grad))).toEqual([[-0.082964, 0.5, 1.082964]])
  })
})
