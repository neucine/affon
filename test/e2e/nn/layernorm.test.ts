import { describe, test, expect, values } from 'std:test'
import checkpoint from 'affon:checkpoint'
import { grad, mean, tensor, variance } from 'affon:compute'
import nn from 'affon:nn'
import { metalAvailable } from '../../support/metal.ts'
import { internal_tensor } from '../../support/compute.ts'

function roundNested(value: any): any {
  if (Array.isArray(value)) return value.map(roundNested)
  return Math.round(value * 1e6) / 1e6
}

describe('nn layernorm', () => {
  test('normalizes values and round-trips state', () => {
    const ln = nn.LayerNorm(3)

    const x2 = tensor([
      [1, 2, 3],
      [2, 4, 6],
    ])
    const y2 = ln(x2)
    expect(y2.shape).toEqual([2, 3])
    expect(roundNested(values(mean(y2, 1, false)))).toEqual([0, 0])
    expect(roundNested(values(variance(y2, 1, false)))).toEqual([0.999985, 0.999996])

    const x3 = tensor([
      [
        [1, 2, 3],
        [3, 2, 1],
      ],
      [
        [2, 4, 6],
        [6, 4, 2],
      ],
    ])
    const y3 = ln(x3)
    expect(y3.shape).toEqual([2, 2, 3])
    expect(roundNested(values(mean(y3, 2, false)))).toEqual([[0, 0], [0, 0]])

    ln.save('/tmp/affon-test-layernorm.safetensors')
    const state = checkpoint.load('/tmp/affon-test-layernorm.safetensors')
    expect(Object.keys(state).sort()).toEqual(['beta', 'gamma'])

    const ln2 = nn.LayerNorm(3)
    ln2.load('/tmp/affon-test-layernorm.safetensors')
    expect(JSON.stringify(roundNested(values(ln2(x2))))).toBe(JSON.stringify(roundNested(values(y2))))
  })

  test('supports backward on rank-3 inputs', () => {
    const ln = nn.LayerNorm(3)
    const x = internal_tensor([
      [
        [1, 2, 3],
        [3, 2, 1],
      ],
      [
        [2, 4, 6],
        [6, 4, 2],
      ],
    ], { dtype: 'f32' })

    grad(ln(x).sum(), [x as any, ...ln.parameters])
    expect(x.grad?.shape).toEqual([2, 2, 3])
    expect(ln.gamma.grad?.shape).toEqual([3])
    expect(ln.beta.grad?.shape).toEqual([3])
  })

  test.skip(() => !metalAvailable())('keeps layernorm finite on metal for extreme finite inputs', () => {
    // This stays on `.backward()` to cover tracked input gradients plus Metal
    // grad placement in one place. The public `grad(loss, params)` path is
    // already covered separately above.
    setDevice('cpu')
    setDevice('metal')
    const ln = nn.LayerNorm(3, { dtype: 'f32' })

    const x = internal_tensor([
      [1e20, 1e20 + 1e12, 1e20 - 1e12],
      [-1e20, -1e20 + 1e12, -1e20 - 1e12],
    ], { dtype: 'f32', device: 'metal' })

    const y = ln(x)
    const rows = values(y.to('cpu')) as number[][]
    expect(rows.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)

    y.sum().backward()
    const grad = values(x.grad?.to('cpu')) as number[][]
    expect(grad.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)
    setDevice('cpu')
  })
})
