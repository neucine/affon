import { describe, test, expect } from 'std:test'
import { add, matmul, mul } from 'affon:compute'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { internal_tensor } from '../../support/compute.ts'

describe('compute parity matmul backward', () => {
  describeParityDevices((device) => {
    test('broadcasted batch matmul gradients match torch', async () => {
      if (!(await python.torch.available())) return

      const a = internal_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], { dtype: 'f32', device: device.name })
      const b = internal_tensor([[2, 1, 0], [1, 2, 3]], { dtype: 'f32', device: device.name })
      const y = matmul(a, b)
      y.sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][][]; shape: number[]; dtype: string }
        gradA: { value: number[][][]; shape: number[]; dtype: string }
        gradB: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        a = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=torch.float32, requires_grad=True)
        b = torch.tensor([[2., 1., 0.], [1., 2., 3.]], dtype=torch.float32, requires_grad=True)
        y = torch.matmul(a, b)
        y.sum().backward()
        emit_many(value=y, gradA=a.grad, gradB=b.grad)
      `)

      expect(y).toHaveShape(peer.value.shape)
      if (device.name === 'metal') expect(y.device).toBe('metal')
      expectBackwardParity({
        device,
        actual: y,
        gradA: a.grad!,
        gradB: b.grad!,
        peer,
      })
    })

    test('rank-3 projection input gradients match torch under non-uniform upstream', async () => {
      if (!(await python.torch.available())) return

      const a = internal_tensor([
        [[0.1, -0.2, 0.3, -0.4], [0.5, 0.2, -0.1, 0.7], [-0.3, 0.8, 0.4, -0.6]],
        [[0.9, -0.5, 0.6, 0.1], [-0.7, 0.3, -0.8, 0.2], [0.4, 0.6, -0.2, -0.9]],
      ], { dtype: 'f32', device: device.name })
      const b = internal_tensor([
        [0.2, -0.1, 0.05, 0.3],
        [-0.25, 0.4, 0.15, -0.2],
        [0.35, -0.3, 0.25, 0.1],
        [-0.15, 0.05, -0.4, 0.2],
      ], { dtype: 'f32', device: device.name })
      const upstream = internal_tensor([
        [[1.0, -0.5, 0.25, 0.75], [-1.25, 0.5, 1.5, -0.25], [0.3, -0.7, 0.9, -1.1]],
        [[-0.8, 1.2, -1.4, 0.6], [0.4, -0.3, 0.2, -0.1], [1.1, -0.9, 0.7, -0.5]],
      ], { dtype: 'f32', device: device.name })

      const y = matmul(a, b)
      mul(y, upstream).sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][][]; shape: number[]; dtype: string }
        gradA: { value: number[][][]; shape: number[]; dtype: string }
        gradB: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        a = torch.tensor([
          [[0.1, -0.2, 0.3, -0.4], [0.5, 0.2, -0.1, 0.7], [-0.3, 0.8, 0.4, -0.6]],
          [[0.9, -0.5, 0.6, 0.1], [-0.7, 0.3, -0.8, 0.2], [0.4, 0.6, -0.2, -0.9]],
        ], dtype=torch.float32, requires_grad=True)
        b = torch.tensor([
          [0.2, -0.1, 0.05, 0.3],
          [-0.25, 0.4, 0.15, -0.2],
          [0.35, -0.3, 0.25, 0.1],
          [-0.15, 0.05, -0.4, 0.2],
        ], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([
          [[1.0, -0.5, 0.25, 0.75], [-1.25, 0.5, 1.5, -0.25], [0.3, -0.7, 0.9, -1.1]],
          [[-0.8, 1.2, -1.4, 0.6], [0.4, -0.3, 0.2, -0.1], [1.1, -0.9, 0.7, -0.5]],
        ], dtype=torch.float32)
        y = torch.matmul(a, b)
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

    test('shared rank-3 projection input accumulates gradients from fanout like torch', async () => {
      if (!(await python.torch.available())) return

      const a = internal_tensor([
        [[0.1, -0.2, 0.3, -0.4], [0.5, 0.2, -0.1, 0.7], [-0.3, 0.8, 0.4, -0.6]],
        [[0.9, -0.5, 0.6, 0.1], [-0.7, 0.3, -0.8, 0.2], [0.4, 0.6, -0.2, -0.9]],
      ], { dtype: 'f32', device: device.name })
      const q = internal_tensor([
        [0.2, -0.1, 0.05, 0.3],
        [-0.25, 0.4, 0.15, -0.2],
        [0.35, -0.3, 0.25, 0.1],
        [-0.15, 0.05, -0.4, 0.2],
      ], { dtype: 'f32', device: device.name })
      const k = internal_tensor([
        [-0.3, 0.25, 0.1, -0.15],
        [0.2, -0.35, 0.3, 0.05],
        [0.4, 0.1, -0.2, 0.15],
        [-0.05, 0.45, -0.25, 0.35],
      ], { dtype: 'f32', device: device.name })
      const v = internal_tensor([
        [0.15, -0.2, 0.35, -0.1],
        [0.05, 0.3, -0.25, 0.4],
        [-0.35, 0.2, 0.1, -0.3],
        [0.25, -0.15, 0.45, 0.05],
      ], { dtype: 'f32', device: device.name })
      const upstream = internal_tensor([
        [[1.0, -0.5, 0.25, 0.75], [-1.25, 0.5, 1.5, -0.25], [0.3, -0.7, 0.9, -1.1]],
        [[-0.8, 1.2, -1.4, 0.6], [0.4, -0.3, 0.2, -0.1], [1.1, -0.9, 0.7, -0.5]],
      ], { dtype: 'f32', device: device.name })

      const y = add(add(matmul(a, q), matmul(a, k)), matmul(a, v))
      mul(y, upstream).sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][][]; shape: number[]; dtype: string }
        gradA: { value: number[][][]; shape: number[]; dtype: string }
      }>(`
        a = torch.tensor([
          [[0.1, -0.2, 0.3, -0.4], [0.5, 0.2, -0.1, 0.7], [-0.3, 0.8, 0.4, -0.6]],
          [[0.9, -0.5, 0.6, 0.1], [-0.7, 0.3, -0.8, 0.2], [0.4, 0.6, -0.2, -0.9]],
        ], dtype=torch.float32, requires_grad=True)
        q = torch.tensor([
          [0.2, -0.1, 0.05, 0.3],
          [-0.25, 0.4, 0.15, -0.2],
          [0.35, -0.3, 0.25, 0.1],
          [-0.15, 0.05, -0.4, 0.2],
        ], dtype=torch.float32, requires_grad=True)
        k = torch.tensor([
          [-0.3, 0.25, 0.1, -0.15],
          [0.2, -0.35, 0.3, 0.05],
          [0.4, 0.1, -0.2, 0.15],
          [-0.05, 0.45, -0.25, 0.35],
        ], dtype=torch.float32, requires_grad=True)
        v = torch.tensor([
          [0.15, -0.2, 0.35, -0.1],
          [0.05, 0.3, -0.25, 0.4],
          [-0.35, 0.2, 0.1, -0.3],
          [0.25, -0.15, 0.45, 0.05],
        ], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([
          [[1.0, -0.5, 0.25, 0.75], [-1.25, 0.5, 1.5, -0.25], [0.3, -0.7, 0.9, -1.1]],
          [[-0.8, 1.2, -1.4, 0.6], [0.4, -0.3, 0.2, -0.1], [1.1, -0.9, 0.7, -0.5]],
        ], dtype=torch.float32)
        y = torch.matmul(a, q) + torch.matmul(a, k) + torch.matmul(a, v)
        (y * upstream).sum().backward()
        emit_many(value=y, gradA=a.grad)
      `)

      expectBackwardParity({
        device,
        actual: y,
        gradA: a.grad!,
        peer,
      })
    })

    test('rank-2 matmul gradients stay stable on asymmetric shapes', async () => {
      if (!(await python.torch.available())) return

      const a = internal_tensor([[1, -2, 3], [0.5, 4, -1]], { dtype: 'f32', device: device.name })
      const b = internal_tensor([[2, 0], [1, -3], [4, 5]], { dtype: 'f32', device: device.name })
      const y = matmul(a, b)
      y.sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][]; shape: number[]; dtype: string }
        gradA: { value: number[][]; shape: number[]; dtype: string }
        gradB: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        a = torch.tensor([[1., -2., 3.], [0.5, 4., -1.]], dtype=torch.float32, requires_grad=True)
        b = torch.tensor([[2., 0.], [1., -3.], [4., 5.]], dtype=torch.float32, requires_grad=True)
        y = torch.matmul(a, b)
        y.sum().backward()
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
