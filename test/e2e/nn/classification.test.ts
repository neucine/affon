import { describe, test, expect } from 'std:test'
import { adam, clear_grad, grad, relu, tensor } from 'affon:compute/legacy'
import nn from 'affon:nn/legacy'

describe('nn classification', () => {
  test('a small classifier with batchnorm and dropout learns a 3-class split', () => {
    const x = tensor([
      [1.0, 1.0],
      [0.5, 0.8],
      [0.8, 0.5],
      [-1.0, 0.5],
      [-0.8, -0.3],
      [-0.5, 0.2],
      [1.0, -1.0],
      [0.5, -0.8],
    ], { dtype: 'f32' })

    const yTarget = tensor([
      [1, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 1, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 0, 1],
    ], { dtype: 'f32' })

    const model = nn.Sequential(
      nn.Linear(2, 8),
      nn.BatchNorm(8),
      relu,
      nn.Dropout(0.1),
      nn.Linear(8, 3),
    )

    const optimizer = adam({ lr: 0.05 })
    const criterion = nn.CrossEntropyLoss()
    const params = model.parameters

    model.train()
    const initialLoss = criterion(model(x), yTarget).item()
    expect(Number.isFinite(initialLoss)).toBe(true)

    for (let epoch = 0; epoch < 200; epoch++) {
      clear_grad(params)
      const loss = criterion(model(x), yTarget)
      grad(loss, params)
      optimizer(params)
    }

    const finalTrainLoss = criterion(model(x), yTarget).item()
    expect(Number.isFinite(finalTrainLoss)).toBe(true)
    expect(finalTrainLoss < initialLoss).toBe(true)

    model.eval()
    const evalLoss = criterion(model(x), yTarget).item()
    expect(Number.isFinite(evalLoss)).toBe(true)
  })
})
