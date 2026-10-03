import { describe, test, expect, values } from 'std:test'
import { dot, matmul, sum, tensor } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { computeParityDTypes, torchDType, internal_tensor } from '../../support/compute.ts'
import { describeParityDevices, materializeOnDevice } from '../../support/parity.ts'

export function registerMatmulParity(): void {
  describe('compute parity matmul', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test(device.name === 'metal' ? 'rank-2 forward and backward' : 'rank-3 batched forward and backward', async () => {
          if (!(await python.torch.available())) return

          const a = device.name === 'metal'
            ? internal_tensor([[1, 2], [3, 4]], { dtype, device: device.name })
            : internal_tensor([[[1, 2], [3, 4]]], { dtype, device: device.name })
          const b = device.name === 'metal'
            ? internal_tensor([[5, 6], [7, 8]], { dtype, device: device.name })
            : internal_tensor([[[5, 6, 7], [8, 9, 10]]], { dtype, device: device.name })
          const y = matmul(a, b)
          sum(y).backward()

          const peer = await python.torch.json<{ value: { value: any }; gradA: { value: any }; gradB: { value: any } }>(`
            a = torch.tensor(${device.name === 'metal' ? '[[1., 2.], [3., 4.]]' : '[[[1., 2.], [3., 4.]]]'}, dtype=${torchDType(dtype)}, requires_grad=True)
            b = torch.tensor(${device.name === 'metal' ? '[[5., 6.], [7., 8.]]' : '[[[5., 6., 7.], [8., 9., 10.]]]'}, dtype=${torchDType(dtype)}, requires_grad=True)
            y = torch.matmul(a, b)
            y.sum().backward()
            emit_many(value=y, gradA=a.grad, gradB=b.grad)
          `)

          expect(values(materializeOnDevice(y))).toBeAllClose(peer.value.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(a.grad!))).toBeAllClose(peer.gradA.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(b.grad!))).toBeAllClose(peer.gradB.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerDotParity(): void {
  describe('compute parity dot', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('forward and backward match torch for 1d dot', async () => {
          if (!(await python.torch.available())) return

          const a = internal_tensor([1, 2, 3], { dtype, device: device.name })
          const b = internal_tensor([4, 5, 6], { dtype, device: device.name })
          const y = dot(a, b)
          y.backward(tensor(1, { dtype }))

          const peer = await python.torch.json<{ value: { value: number }; gradA: { value: number[] }; gradB: { value: number[] } }>(`
            a = torch.tensor([1., 2., 3.], dtype=${torchDType(dtype)}, requires_grad=True)
            b = torch.tensor([4., 5., 6.], dtype=${torchDType(dtype)}, requires_grad=True)
            y = torch.dot(a, b)
            y.backward()
            emit_many(value=y, gradA=a.grad, gradB=b.grad)
          `)

          expect(Math.abs(materializeOnDevice(y).item() - peer.value.value) <= device.atol + device.rtol * Math.abs(peer.value.value)).toBe(true)
          expect(values(materializeOnDevice(a.grad!))).toBeAllClose(peer.gradA.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(b.grad!))).toBeAllClose(peer.gradB.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}
