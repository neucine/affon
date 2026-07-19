import { describe, test, expect, values } from 'std:test'
import { abs, add, clamp, div, exp, log, mul, neg, sign, sqrt, sub, sum } from 'affon:compute'
import { python } from '../../support/python.ts'
import { computeParityDTypes, torchDType, internal_tensor } from '../../support/compute.ts'
import { describeParityDevices, materializeOnDevice } from '../../support/parity.ts'

export function registerAddParity(): void {
  describe('compute parity add', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('forward and backward match torch', async () => {
          if (!(await python.torch.available())) return

          const a = internal_tensor([[1, 2, 3], [4, 5, 6]], { dtype, device: device.name })
          const b = internal_tensor([[10, 20, 30]], { dtype, device: device.name })
          const y = add(a, b)
          sum(y).backward()

          const peer = await python.torch.json<{ value: { value: number[][] }; gradA: { value: number[][] }; gradB: { value: number[][] } }>(`
            a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=${torchDType(dtype)}, requires_grad=True)
            b = torch.tensor([[10., 20., 30.]], dtype=${torchDType(dtype)}, requires_grad=True)
            y = a + b
            y.backward(torch.ones_like(y))
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

export function registerSubParity(): void {
  describe('compute parity sub', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('forward and backward match torch', async () => {
          if (!(await python.torch.available())) return

          const a = internal_tensor([[2, 4, 8], [16, 32, 64]], { dtype, device: device.name })
          const b = internal_tensor([[1, 2, 4]], { dtype, device: device.name })
          const y = sub(a, b)
          sum(y).backward()

          const peer = await python.torch.json<{ value: { value: number[][] }; gradA: { value: number[][] }; gradB: { value: number[][] } }>(`
            a = torch.tensor([[2., 4., 8.], [16., 32., 64.]], dtype=${torchDType(dtype)}, requires_grad=True)
            b = torch.tensor([[1., 2., 4.]], dtype=${torchDType(dtype)}, requires_grad=True)
            y = a - b
            y.backward(torch.ones_like(y))
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

export function registerMulParity(): void {
  describe('compute parity mul', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('forward and backward match torch', async () => {
          if (!(await python.torch.available())) return

          const a = internal_tensor([[2, 4, 8], [16, 32, 64]], { dtype, device: device.name })
          const b = internal_tensor([[1, 2, 4]], { dtype, device: device.name })
          const y = mul(a, b)
          sum(y).backward()

          const peer = await python.torch.json<{ value: { value: number[][] }; gradA: { value: number[][] }; gradB: { value: number[][] } }>(`
            a = torch.tensor([[2., 4., 8.], [16., 32., 64.]], dtype=${torchDType(dtype)}, requires_grad=True)
            b = torch.tensor([[1., 2., 4.]], dtype=${torchDType(dtype)}, requires_grad=True)
            y = a * b
            y.backward(torch.ones_like(y))
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

export function registerDivParity(): void {
  describe('compute parity div', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('forward and backward match torch', async () => {
          if (!(await python.torch.available())) return

          const a = internal_tensor([[2, 4, 8], [16, 32, 64]], { dtype, device: device.name })
          const b = internal_tensor([[1, 2, 4]], { dtype, device: device.name })
          const y = div(a, b)
          sum(y).backward()

          const peer = await python.torch.json<{ value: { value: number[][] }; gradA: { value: number[][] }; gradB: { value: number[][] } }>(`
            a = torch.tensor([[2., 4., 8.], [16., 32., 64.]], dtype=${torchDType(dtype)}, requires_grad=True)
            b = torch.tensor([[1., 2., 4.]], dtype=${torchDType(dtype)}, requires_grad=True)
            y = a / b
            y.backward(torch.ones_like(y))
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

export function registerAbsParity(): void {
  describe('compute parity abs', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch abs', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[-1, 2], [-3, 4]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[-1., 2.], [-3., 4.]], dtype=${torchDType(dtype)})
            emit(torch.abs(x))
          `)

          expect(values(materializeOnDevice(abs(x)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerNegParity(): void {
  describe('compute parity neg', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch neg', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[-1, 2], [-3, 4]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[-1., 2.], [-3., 4.]], dtype=${torchDType(dtype)})
            emit(torch.neg(x))
          `)

          expect(values(materializeOnDevice(neg(x)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerExpParity(): void {
  describe('compute parity exp', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch exp', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[1, 4], [9, 16]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[1., 4.], [9., 16.]], dtype=${torchDType(dtype)})
            emit(torch.exp(x))
          `)

          expect(values(materializeOnDevice(exp(x)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerLogParity(): void {
  describe('compute parity log', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch log', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[1, 4], [9, 16]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[1., 4.], [9., 16.]], dtype=${torchDType(dtype)})
            emit(torch.log(x))
          `)

          expect(values(materializeOnDevice(log(x)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerSqrtParity(): void {
  describe('compute parity sqrt', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch sqrt', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[1, 4], [9, 16]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[1., 4.], [9., 16.]], dtype=${torchDType(dtype)})
            emit(torch.sqrt(x))
          `)

          expect(values(materializeOnDevice(sqrt(x)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerSignParity(): void {
  describe('compute parity sign', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch sign', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[-2, -0.5, 0], [3, 4, 0.75]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[-2., -0.5, 0.], [3., 4., 0.75]], dtype=${torchDType(dtype)})
            emit(torch.sign(x))
          `)

          expect(values(materializeOnDevice(sign(x)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerClampParity(): void {
  describe('compute parity clamp', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch clamp', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([-1, 0.5, 2], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[]>(`
            x = torch.tensor([-1., 0.5, 2.], dtype=${torchDType(dtype)})
            emit(torch.clamp(x, min=0.0, max=1.0))
          `)

          expect(values(materializeOnDevice(clamp(x, 0, 1)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}
