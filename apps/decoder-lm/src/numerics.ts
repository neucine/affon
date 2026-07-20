import { finite_abs_max, finite_summary } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import nn from 'affon:nn'
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

const envDiagnosticsMode = diagnosticsModeFromEnv(process.getEnv('AFFON_NN_DIAGNOSTICS'))
if (envDiagnosticsMode && envDiagnosticsMode !== 'off') {
  nn.diagnostics.configure({ mode: envDiagnosticsMode, include: ['finite:*'] })
}

export function setFiniteChecksEnabled(enabled: boolean): void {
  nn.diagnostics.configure({
    mode: enabled ? 'error' : 'off',
    include: enabled ? ['finite:*'] : [],
  })
}

export function getFiniteChecksEnabled(): boolean {
  return nn.diagnostics.get_config().mode !== 'off'
}

export function describeValue(value: number): string {
  if (Number.isNaN(value)) return 'NaN'
  if (value === Number.POSITIVE_INFINITY) return 'Infinity'
  if (value === Number.NEGATIVE_INFINITY) return '-Infinity'
  return String(value)
}

export function tensorFiniteSummary(value: Tensor): FiniteSummary {
  return finite_summary(value) as FiniteSummary
}

export function tensorAbsMax(value: Tensor): number | null {
  return finite_abs_max(value)
}
