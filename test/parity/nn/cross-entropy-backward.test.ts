import { describe, test } from 'std:test'
import nn from 'affon:nn'
import { tensor } from 'affon:compute'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { trackedF32 } from './helpers.ts'

describe('nn parity cross entropy backward', () => {
  describeParityDevices((device) => {
    test('indexed targets backward matches torch on uneven logits', async () => {
      if (!(await python.torch.available())) return

      const logits = trackedF32([
        [2.5, -1, 0.25, 3],
        [0, 1.5, -2, 4],
        [5, -3, 2, 1],
      ], { device: device.name })
      const targets = tensor([3, 1, 0], { dtype: 'i64', device: device.name })

      const loss = nn.CrossEntropyLoss({ indexed: true })(logits, targets)
      loss.backward()

      const peer = await python.torch.json<{
        value: { value: number; shape: number[]; dtype: string }
        grad: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        logits = torch.tensor([
          [2.5, -1., 0.25, 3.],
          [0., 1.5, -2., 4.],
          [5., -3., 2., 1.],
        ], dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([3, 1, 0], dtype=torch.int64)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        emit_many(value=loss.reshape(1), grad=logits.grad)
      `)

      expectBackwardParity({
        device,
        actual: loss,
        grad: logits.grad!,
        peer,
      })
    })
  })
})
