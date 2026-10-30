// Two-step SGD momentum reference from affon-legacy/test/parity/optim/sgd.test.ts.
import { Session, Tensor, gradient, program, update_parameters } from 'affon:compute'
import { mul, sum } from 'affon:ops'
import { sgd } from 'affon:optim'
import { describe, expect, test } from 'std:test'

describe('Optimizer reference values', () => {
  test('SGD momentum follows the independent two-step vector reference', () => {
    const objective = program('sgd_reference', p => sum(mul(
      p.argument('x', Tensor.f32([2])),
      p.parameter('weight', Tensor.f32([2]), { initializer: { kind: 'ones' } }),
    )))
    const update = update_parameters(objective, gradient(objective, 'weight'), sgd({ learning_rate: 0.1, momentum: 0.8 }))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(update, { parameters: { 'sgd_reference.weight': [1, -2] } })
    const executable = session.compile(update)
    for (const [input, expected] of [
      [[3, -4], [0.7, -1.6]],
      [[-2, 5], [0.66, -1.78]],
    ] as const) {
      const x = session.tensor(input)
      const loss = executable.run({ x }, state)
      const actual = state.parameters['sgd_reference.weight'].to_array() as number[]
      expect(actual.every((value, index) => Math.abs(value - expected[index]) < 1e-5)).toBe(true)
      loss.dispose(); x.dispose()
    }
    expect(state.optimizer_state.$step).toBe(2)
    state.dispose(); session.dispose()
  })
})
