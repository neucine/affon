import { describe, test } from 'std:test'
import { mul, tensor, where } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { internal_tensor } from '../../support/compute.ts'

describe('compute parity where backward', () => {
  describeParityDevices((device) => {
    test('branch and broadcast gradients match torch under non-uniform upstream', async () => {
      if (!(await python.torch.available())) return

      const cond = tensor([[1], [0]], { dtype: 'i64', device: device.name })
      const a = internal_tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32', device: device.name })
      const b = internal_tensor([[3, -4, 5]], { dtype: 'f32', device: device.name })
      const upstream = internal_tensor([[1, -2, 0.5], [-3, 4, 2]], { dtype: 'f32', device: device.name })

      const y = where(cond, a, b)
      mul(y, upstream).sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][]; shape: number[]; dtype: string }
        gradA: { value: number[][]; shape: number[]; dtype: string }
        gradB: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        cond = torch.tensor([[True], [False]])
        a = torch.tensor([[10., 20., 30.], [40., 50., 60.]], dtype=torch.float32, requires_grad=True)
        b = torch.tensor([[3., -4., 5.]], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([[1., -2., 0.5], [-3., 4., 2.]], dtype=torch.float32)
        y = torch.where(cond, a, b)
        (y * upstream).sum().backward()
        emit_many(value=y, gradA=a.grad, gradB=b.grad)
      `)

      expectBackwardParity({
        device,
        actual: y,
        gradA: a.grad!,
        gradB: b.grad!,
        peer,
      })
    })
  })
})
