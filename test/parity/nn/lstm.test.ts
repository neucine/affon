import { describe, test, expect } from 'std:test'
import nn from 'affon:nn'
import { python } from '../../support/python.ts'
import { hostValues, trackedF32 } from './helpers.ts'

describe('nn parity lstm', () => {
  test('matches torch LSTM on sequence-first batched input', async () => {
    if (!(await python.torch.available())) return

    const lstm = nn.LSTM(2, 3)
    const x = trackedF32([
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
      [[9, 10], [11, 12]],
    ])

    const y = lstm(x)
    const h = lstm.hidden_state()
    const c = lstm.cell_state()
    const state = lstm.state()
    const weightIH = hostValues((state as any).layers[0].weight_ih, 'cpu') as number[][]
    const weightHH = hostValues((state as any).layers[0].weight_hh, 'cpu') as number[][]
    const bias = hostValues((state as any).layers[0].bias, 'cpu') as number[][]

    const peer = await python.torch.json<{ value: number[][][]; hidden: number[][][]; cell: number[][][] }>(`
      x = torch.tensor([
        [[1., 2.], [3., 4.]],
        [[5., 6.], [7., 8.]],
        [[9., 10.], [11., 12.]],
      ], dtype=torch.float32)
      lstm = torch.nn.LSTM(2, 3, num_layers=1, bias=True, batch_first=False, dtype=torch.float32)
      with torch.no_grad():
        lstm.weight_ih_l0.copy_(torch.tensor(${JSON.stringify(weightIH)}, dtype=torch.float32).transpose(0, 1))
        lstm.weight_hh_l0.copy_(torch.tensor(${JSON.stringify(weightHH)}, dtype=torch.float32).transpose(0, 1))
        lstm.bias_ih_l0.copy_(torch.tensor(${JSON.stringify(bias[0])}, dtype=torch.float32))
        lstm.bias_hh_l0.zero_()
      y, (h, c) = lstm(x)
      print(json.dumps({
        "value": y.detach().cpu().tolist(),
        "hidden": h.detach().cpu().tolist(),
        "cell": c.detach().cpu().tolist(),
      }))
    `)

    const tol = { rtol: 1e-3, atol: 1e-7 }
    expect(hostValues(y, 'cpu')).toBeAllClose(peer.value, tol)
    expect(hostValues(h, 'cpu')).toBeAllClose(peer.hidden, tol)
    expect(hostValues(c, 'cpu')).toBeAllClose(peer.cell, tol)
  })
})
