import { Session } from 'affon:compute'

let metalAvailableCache: boolean | null = null
export function metalAvailable(): boolean {
  if (metalAvailableCache != null) return metalAvailableCache
  try {
    const session = new Session({ device: 'metal' })
    const probe = session.tensor([1], { dtype: 'f32' })
    metalAvailableCache = probe.device === 'metal'
    probe.dispose()
    session.dispose()
  } catch {
    metalAvailableCache = false
  }
  return metalAvailableCache
}
