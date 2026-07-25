import { internal_tensor } from '../../support/compute.ts'
import { values } from 'std:test'

export function trackedF32(
  data: number | number[] | number[][] | number[][][],
  opts?: { device?: 'cpu' | 'metal' },
): any {
  return internal_tensor(data as any, { dtype: 'f32', device: opts?.device })
}

export function withModuleDevice<T>(device: 'cpu' | 'metal', build: () => T): T {
  if (device !== 'metal') return build()
  const previousDevice = 'cpu'
  setDevice('metal')
  try {
    return build()
  } finally {
    setDevice(previousDevice)
  }
}

export function hostValues(x: any, device: 'cpu' | 'metal'): any {
  return values(device === 'metal' ? x.to('cpu') : x)
}
