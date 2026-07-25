import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn'
import { tensor } from 'affon:compute'
import { python } from '../../support/python.ts'
import { trackedF32 } from './helpers.ts'

describe('nn parity bce loss', () => {
  test('matches torch binary cross entropy for [N,1] predictions and gradients', async () => {
    if (!(await python.torch.available())) return

    const prediction = trackedF32([[0.9], [0.2], [0.8], [0.1]])
    const target = tensor([1, 0, 1, 0], { dtype: 'f32' })

    const loss = nn.BCELoss()(prediction, target)
    loss.backward()

    const peer = await python.torch.json<{
      loss: number
      grad: number[][]
    }>(`
      prediction = torch.tensor([[0.9], [0.2], [0.8], [0.1]], dtype=torch.float32, requires_grad=True)
      target = torch.tensor([1., 0., 1., 0.], dtype=torch.float32)
      loss = torch.nn.functional.binary_cross_entropy(prediction.squeeze(1), target)
      loss.backward()
      print(json.dumps({
      "loss": float(loss.detach().cpu().item()),
      "grad": prediction.grad.detach().cpu().tolist(),
      }))
    `)

    expect(Math.abs(loss.item() - peer.loss) < 2e-7).toBe(true)
    expect(values(prediction.grad)).toBeAllClose(peer.grad, { rtol: 1e-6, atol: 1e-6 })
  })
})
