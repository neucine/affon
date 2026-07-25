import fs from 'std:fs'
import { getEnv } from 'std:process'

type HistoryEntry = {
  epoch?: unknown
  step?: unknown
  trainLoss?: unknown
  valLoss?: unknown
}

type WorkflowSummary = {
  initialValLoss?: unknown
  finalValLoss?: unknown
  history?: unknown
}

function requiredEnv(name: string): string {
  const value = getEnv(name)
  if (!value) throw new AffonError('invalid_arg', `Set ${name}`)
  return value
}

function numberEnv(name: string, fallback: number): number {
  const value = getEnv(name)
  if (!value) return fallback
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    throw new AffonError('invalid_arg', `${name} must be a finite number, got ${value}`)
  }
  return parsed
}

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new AffonError('invalid_arg', `WikiText canary summary missing finite ${name}`)
  }
  return value
}

const summaryPath = requiredEnv('AFFON_WIKITEXT_CANARY_SUMMARY')
const minEpochs = numberEnv('AFFON_WIKITEXT_CANARY_MIN_EPOCHS', 5)
const minSteps = numberEnv('AFFON_WIKITEXT_CANARY_MIN_STEPS', 6250)
const maxFinalValLoss = numberEnv('AFFON_WIKITEXT_CANARY_MAX_FINAL_VAL_LOSS', 6.0)
const minValLossDrop = numberEnv('AFFON_WIKITEXT_CANARY_MIN_VAL_LOSS_DROP', 4.0)
const maxValLossRise = numberEnv('AFFON_WIKITEXT_CANARY_MAX_VAL_LOSS_RISE', 0.02)

const summary = JSON.parse(fs.readFileSync(summaryPath)) as WorkflowSummary
if (!Array.isArray(summary.history)) {
  throw new AffonError('invalid_arg', 'WikiText canary summary missing history')
}

const history = summary.history as HistoryEntry[]
if (history.length < minEpochs) {
  throw new AffonError('invalid_arg', `WikiText canary history has ${history.length} epochs; expected at least ${minEpochs}`)
}

const initialValLoss = finiteNumber(summary.initialValLoss, 'initialValLoss')
const finalValLoss = finiteNumber(summary.finalValLoss, 'finalValLoss')
const finalStep = finiteNumber(history[history.length - 1]?.step, 'history[-1].step')

if (finalStep < minSteps) {
  throw new AffonError('invalid_arg', `WikiText canary only ran ${finalStep} steps; expected at least ${minSteps}`)
}
if (finalValLoss > maxFinalValLoss) {
  throw new AffonError('assertion', `WikiText canary final val loss ${finalValLoss.toFixed(6)} exceeds ${maxFinalValLoss.toFixed(6)}`)
}
if (initialValLoss - finalValLoss < minValLossDrop) {
  throw new AffonError(
    'assertion',
    `WikiText canary val loss drop ${(initialValLoss - finalValLoss).toFixed(6)} is below ${minValLossDrop.toFixed(6)}`,
  )
}

for (let index = 1; index < history.length; index += 1) {
  const previous = finiteNumber(history[index - 1]?.valLoss, `history[${index - 1}].valLoss`)
  const current = finiteNumber(history[index]?.valLoss, `history[${index}].valLoss`)
  if (current > previous + maxValLossRise) {
    throw new AffonError(
      'assertion',
      `WikiText canary val loss rose from ${previous.toFixed(6)} to ${current.toFixed(6)} at history index ${index}`,
    )
  }
}

console.log(`WikiText canary passed: epochs=${history.length} steps=${finalStep} initial_val_loss=${initialValLoss.toFixed(6)} final_val_loss=${finalValLoss.toFixed(6)}`)
