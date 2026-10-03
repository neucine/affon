import { describe, test } from 'std:test'
import nn from 'affon:nn/legacy'
import { mul } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { hostValues, trackedF32, withModuleDevice } from './helpers.ts'

describe('nn parity batchnorm backward', () => {
  describeParityDevices((device) => {
    test('non-uniform upstream gradients match torch in training mode', async () => {
      if (!(await python.torch.available())) return

      const layer = withModuleDevice(device.name, () => nn.BatchNorm(3))
      const x = trackedF32([
        [1, 2, 3],
        [4, -5, 6],
        [7, 8, -9],
      ], { device: device.name })
      const upstream = trackedF32([
        [1, -2, 0.5],
        [-3, 4, 2],
        [0.25, -1, 3],
      ], { device: device.name })

      const y = layer(x)
      mul(y, upstream).sum().backward()
      const gamma = hostValues(layer.gamma, device.name) as number[][]
      const beta = hostValues(layer.beta, device.name) as number[][]

      const peer = await python.torch.json<{
        value: { value: number[][]; shape: number[]; dtype: string }
        grad: { value: number[][]; shape: number[]; dtype: string }
        gradA: { value: number[][]; shape: number[]; dtype: string }
        gradB: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        x = torch.tensor([
          [1., 2., 3.],
          [4., -5., 6.],
          [7., 8., -9.],
        ], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([
          [1., -2., 0.5],
          [-3., 4., 2.],
          [0.25, -1., 3.],
        ], dtype=torch.float32)
        gamma = torch.tensor(${JSON.stringify(gamma)}, dtype=torch.float32, requires_grad=True)
        beta = torch.tensor(${JSON.stringify(beta)}, dtype=torch.float32, requires_grad=True)
        y = torch.nn.functional.batch_norm(
          x,
          running_mean=None,
          running_var=None,
          weight=gamma.squeeze(0),
          bias=beta.squeeze(0),
          training=True,
          momentum=0.1,
          eps=1e-5,
        )
        (y * upstream).sum().backward()
        emit_many(value=y, grad=x.grad, gradA=gamma.grad.reshape(1, -1), gradB=beta.grad.reshape(1, -1))
      `)

      expectBackwardParity({
        device,
        actual: y,
        grad: x.grad!,
        gradA: layer.gamma.grad!,
        gradB: layer.beta.grad!,
        peer,
      })
    })
  })
})
