import { describe, test, expect } from 'std:test'
import { grad, tensor } from 'affon:compute/legacy'
import nn from 'affon:nn/legacy'
import { internal_tensor } from '../../support/compute.ts'

describe('nn cross entropy', () => {
  test('averages per-example loss rather than dividing by class count', () => {
    const logits = tensor([
      [2, 0, -1],
      [0, 1, 3],
    ])
    const targets = tensor([
      [1, 0, 0],
      [0, 0, 1],
    ])

    const loss = nn.CrossEntropyLoss()(logits, targets)
    expect(Math.abs(loss.item() - 0.16984601955628567) < 1e-6).toBe(true)
  })

  test('supports indexed class targets through the public loss option', () => {
    const logits = internal_tensor([
      [2, 0, -1],
      [0, 1, 3],
    ], { dtype: 'f32' })
    const targets = tensor([0, 2], { dtype: 'i64' })

    const explicit = nn.CrossEntropyLoss({ target: 'index' })(logits, targets)
    const inferred = nn.CrossEntropyLoss()(logits, targets)
    expect(Math.abs(explicit.item() - 0.16984601955628567) < 1e-6).toBe(true)
    expect(Math.abs(inferred.item() - explicit.item()) < 1e-6).toBe(true)

    grad(explicit, [logits as any])
    expect(logits.grad?.shape).toEqual([2, 3])
  })

  test('supports wide one-hot target matrices without JS array expansion', () => {
    const classes = 5000
    const logits = tensor([
      Array.from({ length: classes }, (_, i) => (i === 17 ? 3 : -1)),
      Array.from({ length: classes }, (_, i) => (i === 4097 ? 2 : -2)),
    ], { dtype: 'f32' })
    const targets = tensor([
      Array.from({ length: classes }, (_, i) => (i === 17 ? 1 : 0)),
      Array.from({ length: classes }, (_, i) => (i === 4097 ? 1 : 0)),
    ], { dtype: 'f32' })

    const loss = nn.CrossEntropyLoss()(logits, targets)
    expect(Number.isFinite(loss.item())).toBe(true)
    expect(loss.item() > 0).toBe(true)
  })
})
