import { describe, test, expect } from 'std:test'
import nn from 'affon:nn'
import { python } from '../../support/python.ts'
import { hostValues, trackedF32 } from './helpers.ts'

describe('nn parity rnn', () => {
  test('matches torch RNN on sequence-first batched input', async () => {
    if (!(await python.torch.available())) return

    const rnn = nn.RNN(2, 3)
    const x = trackedF32([
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
      [[9, 10], [11, 12]],
    ])

    const y = rnn(x)
    const h = rnn.hidden_state()
    const state = rnn.state()
    const weightIH = hostValues((state as any).layers[0].weight_ih, 'cpu') as number[][]
    const weightHH = hostValues((state as any).layers[0].weight_hh, 'cpu') as number[][]
    const bias = hostValues((state as any).layers[0].bias, 'cpu') as number[][]

    const peer = await python.torch.json<{ value: number[][][]; hidden: number[][][] }>(`
      x = torch.tensor([
        [[1., 2.], [3., 4.]],
        [[5., 6.], [7., 8.]],
        [[9., 10.], [11., 12.]],
      ], dtype=torch.float32)
      rnn = torch.nn.RNN(2, 3, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dtype=torch.float32)
      with torch.no_grad():
        rnn.weight_ih_l0.copy_(torch.tensor(${JSON.stringify(weightIH)}, dtype=torch.float32).transpose(0, 1))
        rnn.weight_hh_l0.copy_(torch.tensor(${JSON.stringify(weightHH)}, dtype=torch.float32).transpose(0, 1))
        rnn.bias_ih_l0.copy_(torch.tensor(${JSON.stringify(bias[0])}, dtype=torch.float32))
        rnn.bias_hh_l0.zero_()
      y, h = rnn(x)
      print(json.dumps({
        "value": y.detach().cpu().tolist(),
        "hidden": h.detach().cpu().tolist(),
      }))
    `)

    expect(hostValues(y, 'cpu')).toBeAllClose(peer.value)
    expect(hostValues(h, 'cpu')).toBeAllClose(peer.hidden)
  })
})
