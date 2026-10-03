import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn/legacy'
import { tensor } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { trackedF32 } from './helpers.ts'

describe('nn parity bce with logits loss', () => {
  test('matches torch BCEWithLogitsLoss for [N,1] logits and gradients', async () => {
    if (!(await python.torch.available())) return

    const logits = trackedF32([[2], [-1], [1.5], [-2.5]])
    const target = tensor([1, 0, 1, 0], { dtype: 'f32' })

    const loss = nn.BCEWithLogitsLoss()(logits, target)
    loss.backward()

    const peer = await python.torch.json<{
      loss: number
      grad: number[][]
    }>(`
      logits = torch.tensor([[2.], [-1.], [1.5], [-2.5]], dtype=torch.float32, requires_grad=True)
      target = torch.tensor([1., 0., 1., 0.], dtype=torch.float32)
      loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), target)
      loss.backward()
      print(json.dumps({
      "loss": float(loss.detach().cpu().item()),
      "grad": logits.grad.detach().cpu().tolist(),
      }))
    `)

    expect(Math.abs(loss.item() - peer.loss) < 1e-6).toBe(true)
    expect(values(logits.grad)).toBeAllClose(peer.grad)
  })
})
