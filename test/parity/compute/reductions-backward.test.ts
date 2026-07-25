import { describe, test } from 'std:test'
import { max, min, mul, std, variance } from 'affon:compute'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { internal_tensor } from '../../support/compute.ts'

describe('compute parity reductions backward', () => {
  describeParityDevices((device) => {
    test('axis std and variance backward match torch with keepdim routing', async () => {
      if (!(await python.torch.available())) return

      const x = internal_tensor([
        [1, -2, 3],
        [4, 0.5, -6],
      ], { dtype: 'f32', device: device.name })
      const stdUpstream = internal_tensor([[1], [-0.5]], { dtype: 'f32', device: device.name })
      const varianceUpstream = internal_tensor([[2], [3]], { dtype: 'f32', device: device.name })

      const stdValue = std(x, 1, true)
      mul(stdValue, stdUpstream).sum().backward()

      const varianceX = internal_tensor([
        [1, -2, 3],
        [4, 0.5, -6],
      ], { dtype: 'f32', device: device.name })
      const varianceValue = variance(varianceX, 1, true)
      mul(varianceValue, varianceUpstream).sum().backward()

      const peer = await python.torch.json<{
        stdValue: { value: number[][]; shape: number[]; dtype: string }
        stdGrad: { value: number[][]; shape: number[]; dtype: string }
        varianceValue: { value: number[][]; shape: number[]; dtype: string }
        varianceGrad: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        x_std = torch.tensor([[1., -2., 3.], [4., 0.5, -6.]], dtype=torch.float32, requires_grad=True)
        std_upstream = torch.tensor([[1.], [-0.5]], dtype=torch.float32)
        std_value = torch.std(x_std, dim=1, keepdim=True, correction=0)
        (std_value * std_upstream).sum().backward()

        x_var = torch.tensor([[1., -2., 3.], [4., 0.5, -6.]], dtype=torch.float32, requires_grad=True)
        var_upstream = torch.tensor([[2.], [3.]], dtype=torch.float32)
        variance_value = torch.var(x_var, dim=1, keepdim=True, correction=0)
        (variance_value * var_upstream).sum().backward()

        emit_many(
          stdValue=std_value,
          stdGrad=x_std.grad,
          varianceValue=variance_value,
          varianceGrad=x_var.grad,
        )
      `)

      expectBackwardParity({
        device,
        actual: stdValue,
        grad: x.grad!,
        peer: { value: peer.stdValue, grad: peer.stdGrad },
      })
      expectBackwardParity({
        device,
        actual: varianceValue,
        grad: varianceX.grad!,
        peer: { value: peer.varianceValue, grad: peer.varianceGrad },
      })
    })

    test('axis min and max backward match torch for unique extrema', async () => {
      if (!(await python.torch.available())) return

      const minX = internal_tensor([
        [3, -2, 5],
        [7, 4, -1],
      ], { dtype: 'f32', device: device.name })
      const minUpstream = internal_tensor([1.5, -2], { dtype: 'f32', device: device.name })
      const minValue = min(minX, 1, false)
      mul(minValue, minUpstream).sum().backward()

      const maxX = internal_tensor([
        [3, -2, 5],
        [7, 4, -1],
      ], { dtype: 'f32', device: device.name })
      const maxUpstream = internal_tensor([0.25, 3], { dtype: 'f32', device: device.name })
      const maxValue = max(maxX, 1, false)
      mul(maxValue, maxUpstream).sum().backward()

      const peer = await python.torch.json<{
        minValue: { value: number[]; shape: number[]; dtype: string }
        minGrad: { value: number[][]; shape: number[]; dtype: string }
        maxValue: { value: number[]; shape: number[]; dtype: string }
        maxGrad: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        x_min = torch.tensor([[3., -2., 5.], [7., 4., -1.]], dtype=torch.float32, requires_grad=True)
        min_upstream = torch.tensor([1.5, -2.], dtype=torch.float32)
        min_value = torch.min(x_min, dim=1).values
        (min_value * min_upstream).sum().backward()

        x_max = torch.tensor([[3., -2., 5.], [7., 4., -1.]], dtype=torch.float32, requires_grad=True)
        max_upstream = torch.tensor([0.25, 3.], dtype=torch.float32)
        max_value = torch.max(x_max, dim=1).values
        (max_value * max_upstream).sum().backward()

        emit_many(
          minValue=min_value,
          minGrad=x_min.grad,
          maxValue=max_value,
          maxGrad=x_max.grad,
        )
      `)

      expectBackwardParity({
        device,
        actual: minValue,
        grad: minX.grad!,
        peer: { value: peer.minValue, grad: peer.minGrad },
      })
      expectBackwardParity({
        device,
        actual: maxValue,
        grad: maxX.grad!,
        peer: { value: peer.maxValue, grad: peer.maxGrad },
      })
    })
  })
})
