import { AffonError } from 'affon:errors'
import native from 'affon:compute/native'

type DType = 'f32' | 'f64' | 'i64'
type Device = 'cpu' | 'metal'
type ComputeValue = {
  shape: number[]
  rank?: number
  ndim: number
  dtype: DType
  device: Device
  item(): number
  to(device: Device): ComputeValue
  to_array(): unknown
  slice(selectors: readonly (number | string)[]): ComputeValue
}

type ComputeState = Record<string, any> | any[] | ComputeValue | number | string | boolean | null
type ModuleRuntime = ((...args: ComputeValue[]) => ComputeValue) & {
  readonly __affon_compute_module: true
  readonly __affon_compute_state: ComputeState
}

function isTensor(value: unknown): value is ComputeValue {
  return !!value && typeof value === 'object' && Array.isArray((value as any).shape) && typeof (value as any).item === 'function'
}

function isModuleRuntime(value: unknown): value is ModuleRuntime {
  return typeof value === 'function' && !!value && (value as any).__affon_compute_module === true
}

function moduleRuntimeStateOf(value: ModuleRuntime): any {
  return (value as any).__affon_compute_state
}

function copyTensorValue(target: ComputeValue, source: ComputeValue): void {
  const adapted = source.dtype === target.dtype
    ? (source.device === target.device ? source : source.to(target.device))
    : native.cast(source, target.dtype as DType).to(target.device)
  ;(native as any).$muladd_(target, 0.0, adapted)
}

export function flattenPersistableState(state: any, out: Record<string, ComputeValue>, prefix = ''): void {
  if (isModuleRuntime(state)) {
    flattenPersistableState(moduleRuntimeStateOf(state), out, prefix)
    return
  }
  if (isTensor(state)) {
    if (!prefix) {
      throw new AffonError('invalid_arg', 'checkpoint.save(state, path) expected a structured state tree with named tensor leaves')
    }
    out[prefix] = state
    return
  }
  if (Array.isArray(state)) {
    for (let i = 0; i < state.length; i++) {
      const next = prefix ? `${prefix}.${i}` : String(i)
      flattenPersistableState(state[i], out, next)
    }
    return
  }
  if (state && typeof state === 'object') {
    const keys = Object.keys(state)
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i]
      const next = prefix ? `${prefix}.${key}` : key
      flattenPersistableState(state[key], out, next)
    }
    return
  }
  if (typeof state === 'function' || state == null) return
  if (typeof state === 'number' || typeof state === 'string' || typeof state === 'boolean') return
  throw new AffonError('invalid_arg', 'checkpoint.save(state, path) expected a persistable state tree')
}

export function saveStateTree(state: ComputeState | ModuleRuntime, path: string): void {
  const flat: Record<string, ComputeValue> = {}
  flattenPersistableState(state, flat)
  ;(native as any).saveNative(Object.entries(flat).map(([name, value]) => ({ name, value })), path)
}

export function loadStateTree(path: string): Record<string, ComputeValue> {
  return (native as any).loadNative(path) as Record<string, ComputeValue>
}

export function restorePersistedState(target: any, source: Record<string, ComputeValue>, prefix = ''): any {
  if (isModuleRuntime(target)) {
    restorePersistedState(moduleRuntimeStateOf(target), source, prefix)
    return target
  }
  if (isTensor(target)) {
    const entry = source[prefix]
    if (!entry) {
      throw new AffonError('invalid_arg', `checkpoint.restore(target, source) missing tensor state for ${prefix}`)
    }
    copyTensorValue(target, entry)
    return target
  }
  if (Array.isArray(target)) {
    for (let i = 0; i < target.length; i++) {
      const next = prefix ? `${prefix}.${i}` : String(i)
      restorePersistedState(target[i], source, next)
    }
    return target
  }
  if (target && typeof target === 'object') {
    const keys = Object.keys(target)
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i]
      const next = prefix ? `${prefix}.${key}` : key
      restorePersistedState(target[key], source, next)
    }
    return target
  }
  if (typeof target === 'function' || target == null) return target
  if (typeof target === 'number' || typeof target === 'string' || typeof target === 'boolean') return target
  throw new AffonError('invalid_arg', 'checkpoint.restore(target, source) expected a persistable state tree')
}
