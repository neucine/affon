import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn/legacy'
import { compile, seed, tensor } from 'affon:compute/legacy'
import { captureError } from '../../support/errors.ts'

describe('nn dropout', () => {
  test('rejects invalid probabilities', () => {
    for (const p of [-0.1, 1, 2]) {
      const err = captureError(() => nn.Dropout(p))
      expect(err instanceof AffonError).toBe(true)
      expect(err.code).toBe('invalid_arg')
      expect(err.message).toContain('range [0, 1)')
    }
  })

  test('returns input unchanged in eval mode', () => {
    const drop = nn.Dropout(0.5)
    drop.eval()
    const x = tensor([[1, 2], [3, 4]])
    expect(values(drop(x))).toBeAllClose(values(x))
  })

  test('supports compiled training-mode dropout with fresh masks per execution', () => {
    seed(7)
    const drop = nn.Dropout(0.5)
    const compiled = compile((x) => drop(x))
    const x = tensor([
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
    ])

    const first = values(compiled(x))
    const second = values(compiled(x))

    expect(JSON.stringify(first) === JSON.stringify(values(x))).toBe(false)
    expect(JSON.stringify(second) === JSON.stringify(values(x))).toBe(false)
    expect(JSON.stringify(first) === JSON.stringify(second)).toBe(false)
  })
})
