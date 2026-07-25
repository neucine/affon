import { describe, test } from 'std:test'
import { mul, reshape } from 'affon:compute'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { internal_tensor } from '../../support/compute.ts'

describe('compute parity reshape backward', () => {
  describeParityDevices((device) => {
    test('non-uniform upstream gradients route through reshape like torch', async () => {
      if (!(await python.torch.available())) return

      const x = internal_tensor([
        [[1, 2], [3, 4], [5, 6]],
        [[-1, -2], [7, 8], [9, 10]],
      ], { dtype: 'f32', device: device.name })
      const upstream = internal_tensor([
        [1, -2, 0.5, 3],
        [-1, 4, 2, -3],
        [1.5, 0.25, -0.75, 2.5],
      ], { dtype: 'f32', device: device.name })

      const y = reshape(x, [3, 4])
      mul(y, upstream).sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][]; shape: number[]; dtype: string }
        grad: { value: number[][][]; shape: number[]; dtype: string }
      }>(`
        x = torch.tensor([
          [[1., 2.], [3., 4.], [5., 6.]],
          [[-1., -2.], [7., 8.], [9., 10.]],
        ], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([
          [1., -2., 0.5, 3.],
          [-1., 4., 2., -3.],
          [1.5, 0.25, -0.75, 2.5],
        ], dtype=torch.float32)
        y = torch.reshape(x, (3, 4))
        (y * upstream).sum().backward()
        emit_many(value=y, grad=x.grad)
      `)

      expectBackwardParity({
        device,
        actual: y,
        grad: x.grad!,
        peer,
      })
    })
  })
})
