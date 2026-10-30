// Causal attention motif adapted from affon-legacy/test/parity/lm/attention-motif.test.ts.
import { Session, Tensor, gradient, program } from 'affon:compute'
import { masked_fill, matmul, softmax, sum } from 'affon:ops'
import { describe, expect, test } from 'std:test'

function close(actual: number, expected: number): void {
  expect(Math.abs(actual - expected) < 1e-5).toBe(true)
}

describe('Attention reference values', () => {
  test('causal mask zeros future probability and score gradient', () => {
    const objective = program('causal_attention_reference', p => {
      const scores = p.parameter('scores', Tensor.f32([2, 2]), { initializer: { kind: 'ones' } })
      const mask = p.argument('mask', Tensor.i64([2, 2]))
      const values = p.argument('values', Tensor.f32([2, 2]))
      return sum(matmul(softmax(masked_fill(scores, mask, -1e9), 1), values))
    })
    const derivative = gradient(objective, 'scores')
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(derivative, { parameters: { 'causal_attention_reference.scores': [[1, 2], [3, 4]] } })
    const mask = session.tensor([[0, 1], [0, 0]], { dtype: 'i64' })
    const values = session.tensor([[1, 0], [0, 2]])
    const result = session.compile(derivative).run({ mask, values }, state)
    const rows = result.to_array() as number[][]
    const p = 1 / (1 + Math.exp(1))
    close(rows[0][0], 0); close(rows[0][1], 0)
    close(rows[1][0], -p * (1 - p)); close(rows[1][1], p * (1 - p))
    result.dispose(); mask.dispose(); values.dispose(); state.dispose(); session.dispose()
  })
})
