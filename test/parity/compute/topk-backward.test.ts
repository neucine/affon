import { describe, test } from 'std:test'
import { sum, topk } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { internal_tensor } from '../../support/compute.ts'

describe('compute parity topk backward', () => {
  describeParityDevices((device) => {
    test('rank-3 non-last-axis topk backward matches torch', async () => {
      if (!(await python.torch.available())) return

      const x = internal_tensor([[[1, 7, 3], [4, 2, 9], [6, 5, 8]]], { dtype: 'f32', device: device.name })
      const result = topk(x, 2, 1)
      sum(result.values).backward()

      const peer = await python.torch.json<{
        values: { value: number[][][]; shape: number[]; dtype: string }
        indices: { value: number[][][]; shape: number[]; dtype: string }
        grad: { value: number[][][]; shape: number[]; dtype: string }
      }>(`
        x = torch.tensor([[[1., 7., 3.], [4., 2., 9.], [6., 5., 8.]]], dtype=torch.float32, requires_grad=True)
        values, indices = torch.topk(x, 2, dim=1)
        values.sum().backward()
        emit_many(values=values, indices=indices.to(torch.float32), grad=x.grad)
      `)

      expectBackwardParity({
        device,
        actualValues: result.values,
        actualIndices: result.indices,
        grad: x.grad!,
        peer,
      })
    })

    test('rank-1 backward matches torch', async () => {
      if (!(await python.torch.available())) return

      const x = internal_tensor([5, 1, 4, 2, 3, 0], { dtype: 'f32', device: device.name })
      const result = topk(x, 3, 0)
      sum(result.values).backward()

      const peer = await python.torch.json<{
        values: { value: number[]; shape: number[]; dtype: string }
        indices: { value: number[]; shape: number[]; dtype: string }
        grad: { value: number[]; shape: number[]; dtype: string }
      }>(`
        x = torch.tensor([5., 1., 4., 2., 3., 0.], dtype=torch.float32, requires_grad=True)
        values, indices = torch.topk(x, 3, dim=0)
        values.sum().backward()
        emit_many(values=values, indices=indices.to(torch.float32), grad=x.grad)
      `)

      expectBackwardParity({
        device,
        actualValues: result.values,
        actualIndices: result.indices,
        grad: x.grad!,
        peer,
      })
    })
  })
})
