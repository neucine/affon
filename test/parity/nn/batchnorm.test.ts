import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn/legacy'
import { sum } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { trackedF32 } from './helpers.ts'

describe('nn parity batchnorm', () => {
  test('matches torch batch norm for 2D training-mode forward values and gradients', async () => {
    if (!(await python.torch.available())) return

    const layer = nn.BatchNorm(3)
    const x = trackedF32([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ])

    const y = layer(x)
    sum(y).backward()
    const gamma = values(layer.gamma) as number[][]
    const beta = values(layer.beta) as number[][]

    const peer = await python.torch.json<{
      value: number[][]
      gradX: number[][]
      gradGamma: number[][]
      gradBeta: number[][]
    }>(`
      x = torch.tensor([
      [1., 2., 3.],
      [4., 5., 6.],
      [7., 8., 9.],
      [10., 11., 12.],
      ], dtype=torch.float32, requires_grad=True)
      gamma = torch.tensor(${JSON.stringify(gamma)}, dtype=torch.float32, requires_grad=True)
      beta = torch.tensor(${JSON.stringify(beta)}, dtype=torch.float32, requires_grad=True)
      y = torch.nn.functional.batch_norm(x, None, None, gamma.view(-1), beta.view(-1), True, 0.1, 1e-5)
      y.sum().backward()
      print(json.dumps({
      "value": y.detach().cpu().tolist(),
      "gradX": x.grad.detach().cpu().tolist(),
      "gradGamma": gamma.grad.detach().cpu().view(1, -1).tolist(),
      "gradBeta": beta.grad.detach().cpu().view(1, -1).tolist(),
      }))
    `)

    expect(values(y)).toBeAllClose(peer.value)
    expect(values(x.grad)).toBeAllClose(peer.gradX)
    expect(values(layer.gamma.grad)).toBeAllClose(peer.gradGamma)
    expect(values(layer.beta.grad)).toBeAllClose(peer.gradBeta)
  })
})
