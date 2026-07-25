import { describe } from 'std:test'
import { metalAvailable } from './metal.ts'

export type ParityDeviceName = 'cpu' | 'metal'

export interface ParityDeviceCase {
  name: ParityDeviceName
  rtol: number
  atol: number
  enabled(): boolean
}

const cpuCase: ParityDeviceCase = {
  name: 'cpu',
  rtol: 1e-5,
  atol: 1e-6,
  enabled: () => true,
}

const metalCase: ParityDeviceCase = {
  name: 'metal',
  rtol: 1e-4,
  atol: 1e-5,
  enabled: metalAvailable,
}

export function describeParityDevices(run: (device: ParityDeviceCase) => void): void {
  describe('cpu', () => run(cpuCase))
  describe.skip(() => !metalCase.enabled())('metal', () => run(metalCase))
}

export function isMetalDevice(value: unknown): value is { device: string } {
  return !!value && typeof value === 'object' && 'device' in (value as any) && (value as any).device === 'metal'
}

export function materializeOnDevice<T extends { to(device: 'cpu'): T }>(value: T): T {
  return isMetalDevice(value) ? value.to('cpu') : value
}

