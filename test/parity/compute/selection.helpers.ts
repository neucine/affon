import { describe, test, expect, values } from 'std:test'
import { gather, index_select, masked_fill, one_hot, sum, topk, tensor, where } from 'affon:compute'
import { python } from '../../support/python.ts'
import { computeParityDTypes, torchDType, internal_tensor } from '../../support/compute.ts'
import { describeParityDevices, materializeOnDevice } from '../../support/parity.ts'

export function registerWhereParity(): void {
  describe('compute parity where', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch where', async () => {
          if (!(await python.torch.available())) return

          const cond = tensor([[1], [0]], { dtype: 'i64', device: device.name })
          const a = internal_tensor([[10, 20], [30, 40]], { dtype, device: device.name })
          const b = internal_tensor([[50, 60], [70, 80]], { dtype, device: device.name })
          const peer = await python.torch.tensor<number[][]>(`
            cond = torch.tensor([[True], [False]])
            a = torch.tensor([[10., 20.], [30., 40.]], dtype=${torchDType(dtype)})
            b = torch.tensor([[50., 60.], [70., 80.]], dtype=${torchDType(dtype)})
            emit(torch.where(cond, a, b))
          `)

          expect(values(materializeOnDevice(where(cond, a, b)))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerGatherParity(): void {
  describe('compute parity gather', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('rank-2 forward and backward', async () => {
          if (!(await python.torch.available())) return

          const source = internal_tensor([[5, 1, 4], [2, 3, 0]], { dtype, device: device.name })
          const index = tensor([[2, 1], [0, 2]], { dtype: 'i64', device: device.name })
          const actual = gather(source, 1, index)
          sum(actual).backward()

          const peer = await python.torch.json<{ value: { value: number[][] }; grad: { value: number[][] } }>(`
            values = torch.tensor([[5., 1., 4.], [2., 3., 0.]], dtype=${torchDType(dtype)}, requires_grad=True)
            index = torch.tensor([[2, 1], [0, 2]], dtype=torch.int64)
            y = torch.gather(values, 1, index)
            y.sum().backward()
            emit_many(value=y, grad=values.grad)
          `)

          expect(values(materializeOnDevice(actual))).toBeAllClose(peer.value.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(source.grad!))).toBeAllClose(peer.grad.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerIndexSelectParity(): void {
  describe('compute parity index_select', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('forward and backward match torch for repeated row selection', async () => {
          if (!(await python.torch.available())) return

          const input = internal_tensor([[10, 20], [30, 40], [50, 60]], { dtype, device: device.name })
          const index = tensor([2, 1, 2], { dtype: 'i64', device: device.name })
          const actual = index_select(input, 0, index)
          sum(actual).backward()

          const peer = await python.torch.json<{ value: { value: number[][] }; grad: { value: number[][] } }>(`
            x = torch.tensor([[10., 20.], [30., 40.], [50., 60.]], dtype=${torchDType(dtype)}, requires_grad=True)
            index = torch.tensor([2, 1, 2], dtype=torch.int64)
            y = torch.index_select(x, 0, index)
            y.sum().backward()
            emit_many(value=y, grad=x.grad)
          `)

          expect(values(materializeOnDevice(actual))).toBeAllClose(peer.value.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(input.grad!))).toBeAllClose(peer.grad.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerTopKParity(): void {
  describe('compute parity topk', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('rank-2 forward and backward', async () => {
          if (!(await python.torch.available())) return

          const x = internal_tensor([[5, 1, 4], [2, 3, 0]], { dtype, device: device.name })
          const result = topk(x, 2, 1)
          sum(result.values).backward()

          const peer = await python.torch.json<{ values: { value: number[][] }; indices: { value: number[][] }; grad: { value: number[][] } }>(`
            x = torch.tensor([[5., 1., 4.], [2., 3., 0.]], dtype=${torchDType(dtype)}, requires_grad=True)
            values, indices = torch.topk(x, 2, dim=1)
            values.sum().backward()
            emit_many(values=values, indices=indices.to(torch.float32), grad=x.grad)
          `)

          expect(values(materializeOnDevice(result.values))).toBeAllClose(peer.values.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(result.indices))).toBeAllClose(peer.indices.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(x.grad!))).toBeAllClose(peer.grad.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerMaskedFillParity(): void {
  describe('compute parity masked_fill', () => {
    describeParityDevices((device) => {
      describe.each(computeParityDTypes(device.name))('$dtype', ({ dtype }) => {
        test('matches torch masked_fill for forward values and gradients', async () => {
          if (!(await python.torch.available())) return

          const mask = tensor([[1, 0, 1], [0, 1, 0]], { dtype: 'i64', device: device.name })
          const input = internal_tensor([[10, 20, 30], [40, 50, 60]], { dtype, device: device.name })
          const y = masked_fill(input, mask, -999)
          sum(y).backward()

          const peer = await python.torch.json<{ value: { value: number[][] }; grad: { value: number[][] } }>(`
            mask = torch.tensor([[True, False, True], [False, True, False]])
            input = torch.tensor([[10., 20., 30.], [40., 50., 60.]], dtype=${torchDType(dtype)}, requires_grad=True)
            y = input.masked_fill(mask, -999.0)
            y.sum().backward()
            emit_many(value=y, grad=input.grad)
          `)

          expect(values(materializeOnDevice(y))).toBeAllClose(peer.value.value, { rtol: device.rtol, atol: device.atol })
          expect(values(materializeOnDevice(input.grad!))).toBeAllClose(peer.grad.value, { rtol: device.rtol, atol: device.atol })
        })
      })
    })
  })
}

export function registerOneHotParity(): void {
  describe('compute parity one_hot', () => {
    describeParityDevices((device) => {
      test('matches torch one_hot on rank-2 indices', async () => {
        if (!(await python.torch.available())) return

        const indices = tensor([[0, 2], [1, 0]], { dtype: 'i64', device: device.name })
        const actual = one_hot(indices, 4)
        const peer = await python.torch.tensor<number[][][]>(`
          x = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
          emit(torch.nn.functional.one_hot(x, num_classes=4).to(torch.float32))
        `)

        expect(values(materializeOnDevice(actual))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
      })
    })
  })
}
