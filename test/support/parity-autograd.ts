import { expect, values } from 'std:test'
import { materializeOnDevice } from './parity.ts'

interface TorchTensorLike<T = unknown> {
  value: T
  shape: number[]
  dtype: string
}

interface BackwardParityPeer<TForward = unknown> {
  value?: TorchTensorLike<TForward>
  values?: TorchTensorLike<TForward>
  indices?: TorchTensorLike<TForward>
  grad?: TorchTensorLike<TForward>
  gradA?: TorchTensorLike<TForward>
  gradB?: TorchTensorLike<TForward>
}

interface BackwardParityOptions<TPeer extends BackwardParityPeer = BackwardParityPeer> {
  device: { rtol: number; atol: number }
  actual?: unknown
  actualValues?: unknown
  actualIndices?: unknown
  grad?: unknown
  gradA?: unknown
  gradB?: unknown
  peer: TPeer
}

function expectClose(device: { rtol: number; atol: number }, actual: unknown, expected: unknown): void {
  expect(values(materializeOnDevice(actual as any))).toBeAllClose(expected as any, {
    rtol: device.rtol,
    atol: device.atol,
  })
}

export function expectBackwardParity<TPeer extends BackwardParityPeer = BackwardParityPeer>({
  device,
  actual,
  actualValues,
  actualIndices,
  grad,
  gradA,
  gradB,
  peer,
}: BackwardParityOptions<TPeer>): void {
  if (actual !== undefined && peer.value) expectClose(device, actual, peer.value.value)
  if (actualValues !== undefined && peer.values) expectClose(device, actualValues, peer.values.value)
  if (actualIndices !== undefined && peer.indices) expectClose(device, actualIndices, peer.indices.value)
  if (grad !== undefined && peer.grad) expectClose(device, grad, peer.grad.value)
  if (gradA !== undefined && peer.gradA) expectClose(device, gradA, peer.gradA.value)
  if (gradB !== undefined && peer.gradB) expectClose(device, gradB, peer.gradB.value)
}
