import { tensor } from 'affon:compute/legacy'

let metalAvailableCache: boolean | null = null

export function metalAvailable(): boolean {
  if (metalAvailableCache != null) return metalAvailableCache
  try {
    const probe = tensor([1], { dtype: 'f32' }).to('metal')
    metalAvailableCache = probe.device === 'metal'
  } catch {
    metalAvailableCache = false
  }
  return metalAvailableCache
}
