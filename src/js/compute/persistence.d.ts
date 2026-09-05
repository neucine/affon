declare module "affon:compute/persistence.ts" {
export type DType = "f32" | "f64" | "i64"
export type Device = "cpu" | "metal" | "cuda" | `cuda:${number}`

export type ComputeValue = {
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

export type ComputeState = Record<string, any> | any[] | ComputeValue | number | string | boolean | null
export type ModuleRuntime = ((...args: ComputeValue[]) => ComputeValue) & {
  readonly __affon_compute_module: true
  readonly __affon_compute_state: ComputeState
}

export function flattenPersistableState(state: any, out: Record<string, ComputeValue>, prefix?: string): void
export function saveStateTree(state: ComputeState | ModuleRuntime, path: string): void
export function loadStateTree(path: string): Record<string, ComputeValue>
export function restorePersistedState(target: any, source: Record<string, ComputeValue>, prefix?: string): any
}
