import { describe, test } from 'std:test'
import { gather, sum, tensor } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { internal_tensor } from '../../support/compute.ts'

describe('compute parity gather backward', () => {
  describeParityDevices((device) => {
    test('repeated indices accumulate gradient like torch', async () => {
      if (!(await python.torch.available())) return

      const source = internal_tensor([[5, 1, 4], [2, 3, 0]], { dtype: 'f32', device: device.name })
      const index = tensor([[2, 2], [0, 0]], { dtype: 'i64', device: device.name })
      const actual = gather(source, 1, index)
      sum(actual).backward()

      const peer = await python.torch.json<{
        value: { value: number[][]; shape: number[]; dtype: string }
        grad: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        source = torch.tensor([[5., 1., 4.], [2., 3., 0.]], dtype=torch.float32, requires_grad=True)
        index = torch.tensor([[2, 2], [0, 0]], dtype=torch.int64)
        y = torch.gather(source, 1, index)
        y.sum().backward()
        emit_many(value=y, grad=source.grad)
      `)

      expectBackwardParity({
        device,
        actual,
        grad: source.grad!,
        peer,
      })
    })

    test('axis-0 gather gradients match torch on rectangular inputs', async () => {
      if (!(await python.torch.available())) return

      const source = internal_tensor([[1, 2], [3, 4], [5, 6]], { dtype: 'f32', device: device.name })
      const index = tensor([[2, 0], [1, 1]], { dtype: 'i64', device: device.name })
      const actual = gather(source, 0, index)
      sum(actual).backward()

      const peer = await python.torch.json<{
        value: { value: number[][]; shape: number[]; dtype: string }
        grad: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        source = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float32, requires_grad=True)
        index = torch.tensor([[2, 0], [1, 1]], dtype=torch.int64)
        y = torch.gather(source, 0, index)
        y.sum().backward()
        emit_many(value=y, grad=source.grad)
      `)

      expectBackwardParity({
        device,
        actual,
        grad: source.grad!,
        peer,
      })
    })
  })
})
