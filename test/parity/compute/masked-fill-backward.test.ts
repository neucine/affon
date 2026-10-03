import { describe, test } from 'std:test'
import { masked_fill, mul, tensor } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { internal_tensor } from '../../support/compute.ts'

describe('compute parity masked-fill backward', () => {
  describeParityDevices((device) => {
    test('broadcast mask gradients match torch under non-uniform upstream', async () => {
      if (!(await python.torch.available())) return

      const mask = tensor([[1], [0]], { dtype: 'i64', device: device.name })
      const input = internal_tensor([[10, 20, 30], [40, 50, 60]], { dtype: 'f32', device: device.name })
      const upstream = internal_tensor([[1, -2, 0.5], [-3, 4, 2]], { dtype: 'f32', device: device.name })

      const y = masked_fill(input, mask, -7)
      mul(y, upstream).sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][]; shape: number[]; dtype: string }
        grad: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        mask = torch.tensor([[True], [False]])
        input = torch.tensor([[10., 20., 30.], [40., 50., 60.]], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([[1., -2., 0.5], [-3., 4., 2.]], dtype=torch.float32)
        y = input.masked_fill(mask, -7.0)
        (y * upstream).sum().backward()
        emit_many(value=y, grad=input.grad)
      `)

      expectBackwardParity({
        device,
        actual: y,
        grad: input.grad!,
        peer,
      })
    })
  })
})
