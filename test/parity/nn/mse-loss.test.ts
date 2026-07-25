import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn'
import { tensor } from 'affon:compute'
import { python } from '../../support/python.ts'
import { trackedF32 } from './helpers.ts'

describe('nn parity mse loss', () => {
  test('matches torch mse loss for [N,1] predictions and gradients', async () => {
    if (!(await python.torch.available())) return

    const prediction = trackedF32([[0.2], [0.7], [0.1]])
    const target = tensor([0, 1, 0], { dtype: 'f32' })

    const loss = nn.MSELoss()(prediction, target)
    loss.backward()

    const peer = await python.torch.json<{
      loss: number
      grad: number[][]
    }>(`
      prediction = torch.tensor([[0.2], [0.7], [0.1]], dtype=torch.float32, requires_grad=True)
      target = torch.tensor([0., 1., 0.], dtype=torch.float32)
      loss = torch.nn.functional.mse_loss(prediction.squeeze(1), target)
      loss.backward()
      print(json.dumps({
      "loss": float(loss.detach().cpu().item()),
      "grad": prediction.grad.detach().cpu().tolist(),
      }))
    `)

    expect(Math.abs(loss.item() - peer.loss) < 1e-12).toBe(true)
    expect(values(prediction.grad)).toBeAllClose(peer.grad)
  })
})
