import { getEnv } from 'std:process'
import { describe, test } from 'std:test'
import { Session } from 'affon:core'

export type TestDevice = 'cpu' | 'metal' | 'cuda'

const configured = getEnv('AFFON_TEST_DEVICES')
const requested = configured == null || configured === ''
  ? ['cpu', 'metal', 'cuda']
  : configured.split(',').map(value => value.trim())
const valid: readonly string[] = ['cpu', 'metal', 'cuda']
if (requested.some(value => !valid.includes(value)))
  throw new Error(`AFFON_TEST_DEVICES must be a comma-separated subset of ${valid.join(',')}`)

const available = new Map<TestDevice, boolean>()

export function hasDevice(device: TestDevice): boolean {
  if (device === 'cpu') return true
  if (!requested.includes(device)) return false
  if (!available.has(device)) {
    try {
      const session = new Session({ device })
      session.dispose()
      available.set(device, true)
    } catch (error) {
      if (!String(error).includes('backend unavailable')) throw error
      available.set(device, false)
    }
  }
  return available.get(device) === true
}

export function testDevices<const T extends readonly TestDevice[]>(devices: T): T[number][] {
  return devices.filter(hasDevice) as T[number][]
}

export function testOn(device: TestDevice) {
  return test.skip(() => !hasDevice(device))
}

export function describeOn(device: TestDevice) {
  return describe.skip(() => !hasDevice(device))
}
