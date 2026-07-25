import { describe, test, expect, values } from 'std:test'
import { clear_grad, grad, seed, sgd, tensor } from 'affon:compute'
import nn from 'affon:nn'

describe('nn training', () => {
  test('a single Linear layer can learn y = 2x + 1', () => {
    seed(0)
    const x = tensor([[1.0], [2.0], [3.0], [4.0]], { dtype: 'f32' })
    const yTarget = tensor([[3.0], [5.0], [7.0], [9.0]], { dtype: 'f32' })

    const model = nn.Linear(1, 1)
    const optimizer = sgd({ lr: 0.01 })
    const criterion = nn.MSELoss()
    const params = model.parameters

    const initialLoss = criterion(model(x), yTarget).item()
    expect(Number.isFinite(initialLoss)).toBe(true)

    for (let epoch = 0; epoch < 100; epoch++) {
      clear_grad(params)
      const pred = model(x)
      const loss = criterion(pred, yTarget)
      grad(loss, params)
      optimizer(params)
    }

    const finalPred = model(x)
    const finalLoss = criterion(finalPred, yTarget).item()

    expect(Number.isFinite(finalLoss)).toBe(true)
    expect(finalLoss < initialLoss).toBe(true)
    expect(values(finalPred)).toBeAllClose(values(yTarget), { rtol: 1e-1, atol: 1e-1 })
  })
})
