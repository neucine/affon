import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn'
import { sum } from 'affon:compute'
import { python } from '../../support/python.ts'
import { trackedF32 } from './helpers.ts'

describe('nn parity linear', () => {
  test('matches torch linear on batched inputs for forward values and gradients', async () => {
    if (!(await python.torch.available())) return

    const layer = nn.Linear(2, 3)
    const x = trackedF32([
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
    ])

    const y = layer(x)
    sum(y).backward()
    const weight = values(layer.weight) as number[][]
    const bias = values(layer.bias) as number[][]

    const peer = await python.torch.json<{
      value: number[][][]
      gradX: number[][][]
      gradW: number[][]
      gradB: number[][]
    }>(`
      x = torch.tensor([
      [[1., 2.], [3., 4.]],
      [[5., 6.], [7., 8.]],
      ], dtype=torch.float32, requires_grad=True)
      weight = torch.tensor(${JSON.stringify(weight)}, dtype=torch.float32, requires_grad=True)
      bias = torch.tensor(${JSON.stringify(bias)}, dtype=torch.float32, requires_grad=True)
      y = x @ weight + bias
      y.sum().backward()
      print(json.dumps({
      "value": y.detach().cpu().tolist(),
      "gradX": x.grad.detach().cpu().tolist(),
      "gradW": weight.grad.detach().cpu().tolist(),
      "gradB": bias.grad.detach().cpu().tolist(),
      }))
    `)

    expect(values(y)).toBeAllClose(peer.value)
    expect(values(x.grad)).toBeAllClose(peer.gradX)
    expect(values(layer.weight.grad)).toBeAllClose(peer.gradW)
    expect(values(layer.bias.grad)).toBeAllClose(peer.gradB)
  })
})
