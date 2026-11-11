import { testDevices } from '../../devices.ts'
// Numerical fixtures from affon-legacy/test/parity/metrics, applied to the current API.
import { Session, metrics } from 'affon:compute'
import { describe, expect, test } from 'std:test'

function near(actual: number, expected: number): void {
  expect(Math.abs(actual - expected) < 1e-6).toBe(true)
}

describe('Metric reference values', () => {
  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`binary classification counts and threshold behavior on ${device}`, () => {
    const session = new Session({ device })
    const prediction = session.tensor([0.9, 0.8, 0.2, 0.7, 0.1, 0.9])
    const target = session.tensor([1, 0, 0, 1, 0, 1])
    near(metrics.accuracy(prediction, target), 5 / 6)
    near(metrics.precision(prediction, target), 3 / 4)
    near(metrics.recall(prediction, target), 1)
    near(metrics.f1(prediction, target), 6 / 7)
    near(metrics.precision(prediction, target, { threshold: 0.95 }), 0)
    prediction.dispose(); target.dispose(); session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`multiclass accuracy uses the maximum score on ${device}`, () => {
    const session = new Session({ device })
    const prediction = session.tensor([[2, 5, 1], [3, 1, 0], [0, 1, 4], [1, 0, 3]])
    const target = session.tensor([1, 0, 2, 2], { dtype: 'i64' })
    near(metrics.accuracy(prediction, target), 1)
    prediction.dispose(); target.dispose(); session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`regression metrics flatten multidimensional values on ${device}`, () => {
    const session = new Session({ device })
    const prediction = session.tensor([[2.5, 0], [2, 8]])
    const target = session.tensor([[3, -0.5], [2, 7]])
    near(metrics.mean_squared_error(prediction, target), 0.375)
    near(metrics.mean_absolute_error(prediction, target), 0.5)
    near(metrics.r2_score(prediction, target), 1 - 1.5 / 29.1875)
    prediction.dispose(); target.dispose(); session.dispose()
  })
})
