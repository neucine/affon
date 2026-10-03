import type { Tensor } from 'affon:compute'
import process from 'std:process'

type FiniteSummary = {
  ok: boolean
  first_bad_flat_index: number
  first_bad_value: number
}

function isTruthyEnv(value: string | null | undefined): boolean {
  return value === '1' || value === 'true' || value === 'yes' || value === 'on'
}

function diagnosticsModeFromEnv(value: string | null): 'off' | 'error' | 'warn' | null {
  if (value === null) return null
  if (value === 'off' || value === '0' || value === 'false' || value === 'no') return 'off'
  if (value === 'warn') return 'warn'
  if (value === 'error' || isTruthyEnv(value)) return 'error'
  return null
}

let finiteChecksEnabled = diagnosticsModeFromEnv(process.getEnv('AFFON_NN_DIAGNOSTICS')) !== 'off'

export function setFiniteChecksEnabled(enabled: boolean): void {
  finiteChecksEnabled = enabled
}

export function getFiniteChecksEnabled(): boolean {
  return finiteChecksEnabled
}

export function describeValue(value: number): string {
  if (Number.isNaN(value)) return 'NaN'
  if (value === Number.POSITIVE_INFINITY) return 'Infinity'
  if (value === Number.NEGATIVE_INFINITY) return '-Infinity'
  return String(value)
}

export function tensorFiniteSummary(value: Tensor): FiniteSummary {
  const values = (value.to_array() as any[]).flat(Infinity).map(Number)
  const first = values.findIndex(entry => !Number.isFinite(entry))
  return { ok: first < 0, first_bad_flat_index: first, first_bad_value: first < 0 ? 0 : values[first] }
}

export function tensorAbsMax(value: Tensor): number | null {
  const values = (value.to_array() as any[]).flat(Infinity).map(Number)
  return values.every(Number.isFinite) ? values.reduce((maximum, entry) => Math.max(maximum, Math.abs(entry)), 0) : null
}
