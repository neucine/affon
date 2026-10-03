import { describe, test, expect, values } from 'std:test'
import { gelu, mul, relu, sigmoid, softmax, sum, tanh } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { computeParityDTypes, torchDType, internal_tensor } from '../../support/compute.ts'
import { describeParityDevices, materializeOnDevice } from '../../support/parity.ts'

export function registerReluParity(): void {
  describe('compute parity relu', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('forward and backward match torch', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[-2, -0.5, 1.5], [3, -4, 0.75]], { dtype, device: device.name })
          const y = relu(x)
          sum(y).backward()

          const peer = await python.torch.json<{ value: { value: number[][] }; grad: { value: number[][] } }>(`
            x = torch.tensor([[-2., -0.5, 1.5], [3., -4., 0.75]], dtype=${torchDType(dtype)}, requires_grad=True)
            y = torch.relu(x)
            y.sum().backward()
            emit_many(value=y, grad=x.grad)
          `)

          expect(values(materializeOnDevice(y))).toBeAllClose(peer.value.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(x.grad!))).toBeAllClose(peer.grad.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerSoftmaxParity(): void {
  describe('compute parity softmax', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test(device.name === 'metal' ? 'rank-2 forward and backward matches torch' : 'rank-1x3 forward and backward matches torch', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor(device.name === 'metal' ? [[1, 2, 3], [0, -1, 1]] : [[1, 2, 3]], { dtype, device: device.name })
          const y = softmax(x, 1)
          sum(mul(y, x)).backward()

          const peer = await python.torch.json<{ value: { value: number[][] }; grad: { value: number[][] } }>(`
            x = torch.tensor(${device.name === 'metal' ? '[[1., 2., 3.], [0., -1., 1.]]' : '[[1., 2., 3.]]'}, dtype=${torchDType(dtype)}, requires_grad=True)
            y = torch.softmax(x, dim=1)
            (y * x).sum().backward()
            emit_many(value=y, grad=x.grad)
          `)

          expect(values(materializeOnDevice(y))).toBeAllClose(peer.value.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(x.grad!))).toBeAllClose(peer.grad.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerSigmoidParity(): void {
  describe('compute parity sigmoid', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch sigmoid', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[-2, -0.5, 1.5], [3, -4, 0.75]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[-2., -0.5, 1.5], [3., -4., 0.75]], dtype=${torchDType(dtype)})
            emit(torch.sigmoid(x))
          `)

          expect(values(materializeOnDevice(sigmoid(x)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerTanhParity(): void {
  describe('compute parity tanh', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch tanh', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[-2, -0.5, 1.5], [3, -4, 0.75]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[-2., -0.5, 1.5], [3., -4., 0.75]], dtype=${torchDType(dtype)})
            emit(torch.tanh(x))
          `)

          expect(values(materializeOnDevice(tanh(x)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerGeluParity(): void {
  describe('compute parity gelu', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch tanh-approx gelu', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[-2, -0.5, 1.5], [3, -4, 0.75]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[-2., -0.5, 1.5], [3., -4., 0.75]], dtype=${torchDType(dtype)})
            emit(torch.nn.functional.gelu(x, approximate='tanh'))
          `)

          expect(values(materializeOnDevice(gelu(x)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}
