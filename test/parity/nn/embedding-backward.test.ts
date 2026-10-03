import { describe, test } from 'std:test'
import nn from 'affon:nn/legacy'
import { mul, tensor } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { hostValues, trackedF32, withModuleDevice } from './helpers.ts'

describe('nn parity embedding backward', () => {
  describeParityDevices((device) => {
    test('weighted repeated-token gradients match torch', async () => {
      if (!(await python.torch.available())) return

      const embedding = withModuleDevice(device.name, () => nn.Embedding(6, 4))
      const tokenIds = tensor([[0, 2, 2], [5, 1, 2]], { dtype: 'i64', device: device.name })
      const upstream = trackedF32([
        [[1, -1, 0.5, 2], [0, 3, -2, 1], [2, -0.5, 1, 4]],
        [[-1, 2, 3, -2], [4, 0.25, -3, 1], [1.5, -2, 2.5, 0]],
      ], { device: device.name })

      const y = embedding(tokenIds)
      mul(y, upstream).sum().backward()
      const weight = hostValues(embedding.weight, device.name) as number[][]

      const peer = await python.torch.json<{
        value: { value: number[][][]; shape: number[]; dtype: string }
        grad: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        weight = torch.tensor(${JSON.stringify(weight)}, dtype=torch.float32, requires_grad=True)
        token_ids = torch.tensor([[0, 2, 2], [5, 1, 2]], dtype=torch.int64)
        upstream = torch.tensor([
          [[1., -1., 0.5, 2.], [0., 3., -2., 1.], [2., -0.5, 1., 4.]],
          [[-1., 2., 3., -2.], [4., 0.25, -3., 1.], [1.5, -2., 2.5, 0.]],
        ], dtype=torch.float32)
        y = torch.nn.functional.embedding(token_ids, weight)
        (y * upstream).sum().backward()
        emit_many(value=y, grad=weight.grad)
      `)

      expectBackwardParity({
        device,
        actual: y,
        grad: embedding.weight.grad!,
        peer,
      })
    })
  })
})
