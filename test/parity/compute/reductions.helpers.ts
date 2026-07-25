import { describe, test, expect, values } from 'std:test'
import { argmax, argmin, max, mean, min, std, sum, variance } from 'affon:compute'
import { python } from '../../support/python.ts'
import { computeParityDTypes, torchDType, internal_tensor } from '../../support/compute.ts'
import { describeParityDevices, materializeOnDevice } from '../../support/parity.ts'

export function registerSumParity(): void {
  describe('compute parity sum', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch sum with gradient accumulation', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[1, 2, 3], [4, 5, 6]], { dtype, device: device.name })
          const y = sum(x, 1, false)
          sum(mean(x, 0, false)).backward()
          sum(y).backward()

          const peer = await python.torch.json<{ value: { value: number[] }; grad: { value: number[][] } }>(`
            x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=${torchDType(dtype)}, requires_grad=True)
            y = torch.sum(x, dim=1)
            torch.mean(x, dim=0).backward(torch.ones(3, dtype=${torchDType(dtype)}), retain_graph=True)
            y.backward(torch.ones(2, dtype=${torchDType(dtype)}))
            emit_many(value=y, grad=x.grad)
          `)

          expect(values(materializeOnDevice(y))).toBeAllClose(peer.value.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(x.grad!))).toBeAllClose(peer.grad.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerMeanParity(): void {
  describe('compute parity mean', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch mean along an axis with gradients', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[1, 2, 3], [4, 5, 6]], { dtype, device: device.name })
          const y = mean(x, 0, false)
          sum(y).backward()

          const peer = await python.torch.json<{ value: { value: number[] }; grad: { value: number[][] } }>(`
            x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=${torchDType(dtype)}, requires_grad=True)
            y = torch.mean(x, dim=0)
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

export function registerVarianceParity(): void {
  describe('compute parity variance', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('rank-2 axis and all-elements variance', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor(device.name === 'metal' ? [[1, 2, 3], [4, 5, 6]] : [[1, 2], [3, 4]], { dtype, device: device.name })
          const peer = await python.torch.json<{ axis: { value: number[] }; all: { value: number[] } }>(`
            x = torch.tensor(${device.name === 'metal' ? '[[1., 2., 3.], [4., 5., 6.]]' : '[[1., 2.], [3., 4.]]'}, dtype=${torchDType(dtype)})
            emit_many(axis=torch.var(x, dim=1, correction=0), all=torch.var(x.reshape(-1), correction=0).reshape(1))
          `)

          expect(values(materializeOnDevice(variance(x, 1, false)))).toBeAllClose(peer.axis.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(variance(x)))).toBeAllClose(peer.all.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerStdParity(): void {
  describe('compute parity std', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('rank-2 axis std', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor(device.name === 'metal' ? [[1, 2, 3], [4, 5, 6]] : [[1, 2], [3, 4]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[]>(`
            x = torch.tensor(${device.name === 'metal' ? '[[1., 2., 3.], [4., 5., 6.]]' : '[[1., 2.], [3., 4.]]'}, dtype=${torchDType(dtype)})
            emit(torch.std(x, dim=1, correction=0))
          `)

          expect(values(materializeOnDevice(std(x, 1, false)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerMinParity(): void {
  describe('compute parity min', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('all-elements and axis forward match torch', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[3, -2, 4], [1, 5, -1]], { dtype, device: device.name })
          const all = min(x)
          const axis = min(x, 1, false)

          const peer = await python.torch.json<{ all: { value: number }; axis: { value: number[] } }>(`
            x = torch.tensor([[3., -2., 4.], [1., 5., -1.]], dtype=${torchDType(dtype)})
            emit_many(all=torch.min(x), axis=torch.min(x, dim=1).values)
          `)

          expect(values(materializeOnDevice(all))).toBeAllClose([peer.all.value], { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(axis))).toBeAllClose(peer.axis.value, { rtol: device.rtol, atol: device.atol })
        })

        test('all-elements and axis backward match torch for unique minima', async () => {
          if (!(await python.torch.available())) return

          const xAll = internal_tensor([[3, -2, 4], [1, 5, -1]], { dtype, device: device.name })
          min(xAll).backward()

          const peerAll = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[3., -2., 4.], [1., 5., -1.]], dtype=${torchDType(dtype)}, requires_grad=True)
            torch.min(x).backward()
            emit(x.grad)
          `)

          expect(values(materializeOnDevice(xAll.grad!))).toBeAllClose(peerAll.value, { rtol: device.rtol, atol: device.atol })

          const xAxis = internal_tensor([[3, -2, 4], [1, 5, -1]], { dtype, device: device.name })
          sum(min(xAxis, 1, false)).backward()

          const peerAxis = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[3., -2., 4.], [1., 5., -1.]], dtype=${torchDType(dtype)}, requires_grad=True)
            torch.min(x, dim=1).values.sum().backward()
            emit(x.grad)
          `)

          expect(values(materializeOnDevice(xAxis.grad!))).toBeAllClose(peerAxis.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerMaxParity(): void {
  describe('compute parity max', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('all-elements and axis forward match torch', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[3, -2, 4], [1, 5, -1]], { dtype, device: device.name })
          const all = max(x)
          const axis = max(x, 1, false)

          const peer = await python.torch.json<{ all: { value: number }; axis: { value: number[] } }>(`
            x = torch.tensor([[3., -2., 4.], [1., 5., -1.]], dtype=${torchDType(dtype)})
            emit_many(all=torch.max(x), axis=torch.max(x, dim=1).values)
          `)

          expect(values(materializeOnDevice(all))).toBeAllClose([peer.all.value], { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(axis))).toBeAllClose(peer.axis.value, { rtol: device.rtol, atol: device.atol })
        })

        test('all-elements and axis backward match torch for unique maxima', async () => {
          if (!(await python.torch.available())) return

          const xAll = internal_tensor([[3, -2, 4], [1, 5, -1]], { dtype, device: device.name })
          max(xAll).backward()

          const peerAll = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[3., -2., 4.], [1., 5., -1.]], dtype=${torchDType(dtype)}, requires_grad=True)
            torch.max(x).backward()
            emit(x.grad)
          `)

          expect(values(materializeOnDevice(xAll.grad!))).toBeAllClose(peerAll.value, { rtol: device.rtol, atol: device.atol })

          const xAxis = internal_tensor([[3, -2, 4], [1, 5, -1]], { dtype, device: device.name })
          sum(max(xAxis, 1, false)).backward()

          const peerAxis = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[3., -2., 4.], [1., 5., -1.]], dtype=${torchDType(dtype)}, requires_grad=True)
            torch.max(x, dim=1).values.sum().backward()
            emit(x.grad)
          `)

          expect(values(materializeOnDevice(xAxis.grad!))).toBeAllClose(peerAxis.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerArgminParity(): void {
  describe('compute parity argmin', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('all-elements and axis indices match torch', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[3, -2, 4], [1, 5, -1]], { dtype, device: device.name })
          const all = argmin(x)
          const axis = argmin(x, 1, false)

          const peer = await python.torch.json<{ all: { value: number }; axis: { value: number[] } }>(`
            x = torch.tensor([[3., -2., 4.], [1., 5., -1.]], dtype=${torchDType(dtype)})
            emit_many(all=torch.argmin(x).to(torch.float32), axis=torch.argmin(x, dim=1).to(torch.float32))
          `)

          expect(values(materializeOnDevice(all))).toBeAllClose([peer.all.value], { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(axis))).toBeAllClose(peer.axis.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerArgmaxParity(): void {
  describe('compute parity argmax', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch argmax along an axis', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[3, -2, 4], [1, 5, -1]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[]>(`
            x = torch.tensor([[3., -2., 4.], [1., 5., -1.]], dtype=${torchDType(dtype)})
            emit(torch.argmax(x, dim=1).to(torch.float32))
          `)

          expect(values(materializeOnDevice(argmax(x, 1, false)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}
