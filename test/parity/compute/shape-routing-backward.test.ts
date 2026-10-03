import { describe, test } from 'std:test'
import { add, contiguous, mul, permute, reshape, squeeze, transpose, unsqueeze } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { internal_tensor } from '../../support/compute.ts'

describe('compute parity shape routing backward', () => {
  describeParityDevices((device) => {
    test('permute backward routes non-uniform gradients like torch', async () => {
      if (!(await python.torch.available())) return

      const x = internal_tensor([
        [[1, 2], [3, 4], [5, 6]],
        [[-1, -2], [7, 8], [9, 10]],
      ], { dtype: 'f32', device: device.name })
      const upstream = internal_tensor([
        [[1, -2], [0.5, 3]],
        [[-1, 4], [2, -3]],
        [[1.5, 0.25], [-0.75, 2.5]],
      ], { dtype: 'f32', device: device.name })

      const y = permute(x, [1, 0, 2])
      mul(y, upstream).sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][][]; shape: number[]; dtype: string }
        grad: { value: number[][][]; shape: number[]; dtype: string }
      }>(`
        x = torch.tensor([
          [[1., 2.], [3., 4.], [5., 6.]],
          [[-1., -2.], [7., 8.], [9., 10.]],
        ], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([
          [[1., -2.], [0.5, 3.]],
          [[-1., 4.], [2., -3.]],
          [[1.5, 0.25], [-0.75, 2.5]],
        ], dtype=torch.float32)
        y = torch.permute(x, (1, 0, 2))
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

    test('transpose squeeze and unsqueeze backward match torch', async () => {
      if (!(await python.torch.available())) return

      const x = internal_tensor([[[1, 2, 3], [4, 5, 6]]], { dtype: 'f32', device: device.name })
      const upstream = internal_tensor([
        [1, -1],
        [2, 0.5],
        [-3, 4],
      ], { dtype: 'f32', device: device.name })

      const y = squeeze(transpose(unsqueeze(squeeze(x, 0), 0), 0, 2), 2)
      mul(y, upstream).sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][]; shape: number[]; dtype: string }
        grad: { value: number[][][]; shape: number[]; dtype: string }
      }>(`
        x = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([[1., -1.], [2., 0.5], [-3., 4.]], dtype=torch.float32)
        y = torch.squeeze(torch.transpose(torch.unsqueeze(torch.squeeze(x, 0), 0), 0, 2), 2)
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

    test('contiguous backward routes non-contiguous view gradients like torch', async () => {
      if (!(await python.torch.available())) return

      const x = internal_tensor([
        [[1, 2], [3, 4], [5, 6]],
        [[-1, -2], [7, 8], [9, 10]],
      ], { dtype: 'f32', device: device.name })
      const upstream = internal_tensor([
        [[1, -2], [0.5, 3]],
        [[-1, 4], [2, -3]],
        [[1.5, 0.25], [-0.75, 2.5]],
      ], { dtype: 'f32', device: device.name })

      const y = contiguous(permute(x, [1, 0, 2]))
      mul(y, upstream).sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][][]; shape: number[]; dtype: string }
        grad: { value: number[][][]; shape: number[]; dtype: string }
      }>(`
        x = torch.tensor([
          [[1., 2.], [3., 4.], [5., 6.]],
          [[-1., -2.], [7., 8.], [9., 10.]],
        ], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([
          [[1., -2.], [0.5, 3.]],
          [[-1., 4.], [2., -3.]],
          [[1.5, 0.25], [-0.75, 2.5]],
        ], dtype=torch.float32)
        y = torch.permute(x, (1, 0, 2)).contiguous()
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

    test('view roundtrip routes broadcast parameter reductions like torch', async () => {
      if (!(await python.torch.available())) return

      const x = internal_tensor([
        [[1, -2], [3, -4], [5, -6]],
        [[-1, 2], [-3, 4], [-5, 6]],
      ], { dtype: 'f32', device: device.name })
      const bias = internal_tensor([[
        [0.1, -0.2, 0.3],
        [-0.4, 0.5, -0.6],
      ]], { dtype: 'f32', device: device.name })
      const upstream = internal_tensor([
        [[1, -2, 0.5], [3, -1, 2]],
        [[-0.75, 1.5, -2.5], [0.25, -3, 4]],
      ], { dtype: 'f32', device: device.name })

      const y = add(contiguous(permute(reshape(x, [2, 2, 3]), [1, 0, 2])), bias)
      mul(y, upstream).sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][][]; shape: number[]; dtype: string }
        gradA: { value: number[][][]; shape: number[]; dtype: string }
        gradB: { value: number[][][]; shape: number[]; dtype: string }
      }>(`
        x = torch.tensor([
          [[1., -2.], [3., -4.], [5., -6.]],
          [[-1., 2.], [-3., 4.], [-5., 6.]],
        ], dtype=torch.float32, requires_grad=True)
        bias = torch.tensor([[
          [0.1, -0.2, 0.3],
          [-0.4, 0.5, -0.6],
        ]], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([
          [[1., -2., 0.5], [3., -1., 2.]],
          [[-0.75, 1.5, -2.5], [0.25, -3., 4.]],
        ], dtype=torch.float32)
        y = torch.reshape(x, (2, 2, 3)).permute(1, 0, 2).contiguous() + bias
        (y * upstream).sum().backward()
        emit_many(value=y, gradA=x.grad, gradB=bias.grad)
      `)

      expectBackwardParity({
        device,
        actual: y,
        gradA: x.grad!,
        gradB: bias.grad!,
        peer,
      })
    })
  })
})
