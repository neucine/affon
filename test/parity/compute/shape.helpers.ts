import { describe, test, expect, values } from 'std:test'
import { cat, permute, reshape, squeeze, stack, transpose, unsqueeze } from 'affon:compute'
import { python } from '../../support/python.ts'
import { computeParityDTypes, torchDType, internal_tensor } from '../../support/compute.ts'
import { describeParityDevices, materializeOnDevice } from '../../support/parity.ts'

export function registerCatParity(): void {
  describe('compute parity cat', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch cat', async () => {
          if (!(await python.torch.available())) return

          const a = internal_tensor([[1, 2], [3, 4]], { dtype, device: device.name })
          const b = internal_tensor([[5, 6]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            a = torch.tensor([[1., 2.], [3., 4.]], dtype=${torchDType(dtype)})
            b = torch.tensor([[5., 6.]], dtype=${torchDType(dtype)})
            emit(torch.cat((a, b), dim=0))
          `)

          expect(values(materializeOnDevice(cat([a, b], 0)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerStackParity(): void {
  describe('compute parity stack', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch stack', async () => {
          if (!(await python.torch.available())) return

          const a = internal_tensor([[1, 2], [3, 4]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][][]>(`
            a = torch.tensor([[1., 2.], [3., 4.]], dtype=${torchDType(dtype)})
            emit(torch.stack((a, a), dim=0))
          `)

          expect(values(materializeOnDevice(stack([a, a], 0)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerSqueezeParity(): void {
  describe('compute parity squeeze', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch squeeze', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[[1], [2]]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[]>(`
            x = torch.tensor([[[1.], [2.]]], dtype=${torchDType(dtype)})
            emit(torch.squeeze(x))
          `)

          expect(values(materializeOnDevice(squeeze(squeeze(x, 0), 1)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerUnsqueezeParity(): void {
  describe('compute parity unsqueeze', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch unsqueeze', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([1, 2, 3], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([1., 2., 3.], dtype=${torchDType(dtype)})
            emit(torch.unsqueeze(x, 0))
          `)

          expect(values(materializeOnDevice(unsqueeze(x, 0)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerReshapeParity(): void {
  describe('compute parity reshape', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch reshape', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[[1, 2], [3, 4], [5, 6]]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]], dtype=${torchDType(dtype)})
            emit(torch.reshape(x, (1, 6)))
          `)

          expect(values(materializeOnDevice(reshape(x, [1, 6])))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerTransposeParity(): void {
  describe('compute parity transpose', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch transpose semantics via permute', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[1, 2], [3, 4]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            x = torch.tensor([[1., 2.], [3., 4.]], dtype=${torchDType(dtype)})
            emit(torch.permute(x, (1, 0)))
          `)

          expect(values(materializeOnDevice(transpose(x, 0, 1)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerPermuteParity(): void {
  describe('compute parity permute', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch permute', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[[1, 2], [3, 4], [5, 6]]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][][]>(`
            x = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]], dtype=${torchDType(dtype)})
            emit(torch.permute(x, (0, 2, 1)))
          `)

          expect(values(materializeOnDevice(permute(x, [0, 2, 1])))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}
