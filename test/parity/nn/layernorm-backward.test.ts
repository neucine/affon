import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn/legacy'
import { mul } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { trackedF32 } from './helpers.ts'

describe('nn parity layernorm backward', () => {
  test('matches torch layer norm under non-uniform upstream gradients', async () => {
    if (!(await python.torch.available())) return

    const layer = nn.LayerNorm(3)
    const x = trackedF32([
      [[1, -2, 3], [4, 0.5, -6]],
      [[-7, 8, 1], [10, -11, 12]],
    ])
    const upstream = trackedF32([
      [[1, 2, -1], [0.5, -3, 4]],
      [[-2, 1, 3], [4, -1, 0.25]],
    ])

    const y = layer(x)
    mul(y, upstream).sum().backward()
    const gamma = values(layer.gamma) as number[]
    const beta = values(layer.beta) as number[]

    const peer = await python.torch.json<{
      value: number[][][]
      gradX: number[][][]
      gradGamma: number[]
      gradBeta: number[]
    }>(`
      x = torch.tensor([
      [[1., -2., 3.], [4., 0.5, -6.]],
      [[-7., 8., 1.], [10., -11., 12.]],
      ], dtype=torch.float32, requires_grad=True)
      upstream = torch.tensor([
      [[1., 2., -1.], [0.5, -3., 4.]],
      [[-2., 1., 3.], [4., -1., 0.25]],
      ], dtype=torch.float32)
      gamma = torch.tensor(${JSON.stringify(gamma)}, dtype=torch.float32, requires_grad=True)
      beta = torch.tensor(${JSON.stringify(beta)}, dtype=torch.float32, requires_grad=True)
      y = torch.nn.functional.layer_norm(x, (3,), gamma, beta, 1e-5)
      (y * upstream).sum().backward()
      print(json.dumps({
      "value": y.detach().cpu().tolist(),
      "gradX": x.grad.detach().cpu().tolist(),
      "gradGamma": gamma.grad.detach().cpu().tolist(),
      "gradBeta": beta.grad.detach().cpu().tolist(),
      }))
    `)

    expect(values(y)).toBeAllClose(peer.value)
    expect(values(x.grad)).toBeAllClose(peer.gradX)
    expect(values(layer.gamma.grad)).toBeAllClose(peer.gradGamma)
    expect(values(layer.beta.grad)).toBeAllClose(peer.gradBeta)
  })
})
