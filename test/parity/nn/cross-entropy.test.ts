import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn/legacy'
import { tensor } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { trackedF32 } from './helpers.ts'

describe('nn parity cross entropy', () => {
  test('matches torch cross entropy for one-hot targets', async () => {
    if (!(await python.torch.available())) return

    const logits = trackedF32([
      [2, 0, -1],
      [0, 1, 3],
    ])
    const targets = tensor([
      [1, 0, 0],
      [0, 0, 1],
    ], { dtype: 'f32' })

    const loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    const peer = await python.torch.json<{
      loss: number
      grad: number[][]
    }>(`
      logits = torch.tensor([[2., 0., -1.], [0., 1., 3.]], dtype=torch.float32, requires_grad=True)
      targets = torch.tensor([0, 2], dtype=torch.int64)
      loss = torch.nn.functional.cross_entropy(logits, targets)
      loss.backward()
      print(json.dumps({
      "loss": float(loss.detach().cpu().item()),
      "grad": logits.grad.detach().cpu().tolist(),
      }))
    `)

    expect(Math.abs(loss.item() - peer.loss) < 1e-9).toBe(true)
    expect(values(logits.grad)).toBeAllClose(peer.grad)
  })
})
