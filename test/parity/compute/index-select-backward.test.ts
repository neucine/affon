import { describe, test } from 'std:test'
import { index_select, sum, tensor } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { internal_tensor } from '../../support/compute.ts'

describe('compute parity index_select backward', () => {
  describeParityDevices((device) => {
    test('axis-0 repeated selection accumulates gradients like torch', async () => {
      if (!(await python.torch.available())) return

      const input = internal_tensor([[10, 20], [30, 40], [50, 60]], { dtype: 'f32', device: device.name })
      const index = tensor([2, 1, 2], { dtype: 'i64', device: device.name })
      const actual = index_select(input, 0, index)
      sum(actual).backward()

      const peer = await python.torch.json<{
        value: { value: number[][]; shape: number[]; dtype: string }
        grad: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        x = torch.tensor([[10., 20.], [30., 40.], [50., 60.]], dtype=torch.float32, requires_grad=True)
        index = torch.tensor([2, 1, 2], dtype=torch.int64)
        y = torch.index_select(x, 0, index)
        y.sum().backward()
        emit_many(value=y, grad=x.grad)
      `)

      expectBackwardParity({
        device,
        actual,
        grad: input.grad!,
        peer,
      })
    })

    test('axis-1 selection on rectangular input matches torch', async () => {
      if (!(await python.torch.available())) return

      const input = internal_tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32', device: device.name })
      const index = tensor([2, 0], { dtype: 'i64', device: device.name })
      const actual = index_select(input, 1, index)
      sum(actual).backward()

      const peer = await python.torch.json<{
        value: { value: number[][]; shape: number[]; dtype: string }
        grad: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float32, requires_grad=True)
        index = torch.tensor([2, 0], dtype=torch.int64)
        y = torch.index_select(x, 1, index)
        y.sum().backward()
        emit_many(value=y, grad=x.grad)
      `)

      expectBackwardParity({
        device,
        actual,
        grad: input.grad!,
        peer,
      })
    })
  })
})
