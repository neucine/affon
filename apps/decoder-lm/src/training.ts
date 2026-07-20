import checkpointModule from 'affon:checkpoint'
import telemetry from 'std:telemetry'
import {
  adam,
  adamw,
  axes,
  clear_grad,
  clip_grad_norm,
  copy,
  empty,
  exportBundleFile,
  grad,
  mul,
  no_grad,
  tensor,
} from 'affon:compute'
import type { ComputeState, Parameter, Tensor } from 'affon:compute'

import { CausalLMLoss, causal_lm_eval_loss_forward } from '../../../packages/lm/src/causal-lm.ts'
import type { DecoderModelModule } from '../../../packages/lm/src/model.ts'
import { describeValue, tensorAbsMax, tensorFiniteSummary } from './numerics.ts'
import type {
  PackedCorpusOptions,
  TokenBatchOptions,
  TokenWindow,
} from '../../../packages/lm/src/token-windows.ts'

export type { PackedCorpusOptions, TokenBatchOptions, TokenWindow } from '../../../packages/lm/src/token-windows.ts'
export type DecoderLMForward = (tokenIds: Tensor<[number, number], 'f32'>) => Tensor<number[], 'f32'>
type DecoderModelState = ReturnType<DecoderModelModule['state']>
type DecoderLMOptimizerState = {
  kind: string
  scalars: Record<string, number>
  tensors: Record<string, Tensor>
}

export interface DecoderLMCheckpoint {
  epoch: number
  step: number
  trainLoss: number
  valLoss: number | null
  state: DecoderModelState
  optimizerState?: DecoderLMOptimizerState | null
  resumeEpoch?: number
  resumeBatchIndex?: number
  batchOrder?: number[] | null
  shuffleState?: number
  epochLossAccum?: number
}

export interface DecoderLMCheckpointMetadata {
  readonly [key: string]: unknown
}

export interface SavedDecoderLMCheckpoint {
  epoch: number
  step: number
  trainLoss: number
  valLoss: number | null
  state: DecoderModelState
  metadata: DecoderLMCheckpointMetadata | null
  optimizerState: DecoderLMOptimizerState | null
  resumeEpoch: number
  resumeBatchIndex: number
  batchOrder: number[] | null
  shuffleState: number | null
  epochLossAccum: number
}

export interface DecoderLMEpochMetrics {
  epoch: number
  step: number
  trainLoss: number
  valLoss: number | null
  trainPerplexity: number
  valPerplexity: number | null
}

export interface DecoderLMBatchMetrics {
  epoch: number
  batch: number
  batches: number
  step: number
  batchLoss: number
  lr: number
  gradNorm: number | null
  optimizerStepped: boolean
}

export interface DecoderLMBatchPhaseMetrics {
  epoch: number
  batch: number
  batches: number
  step: number
  phase: 'forward' | 'backward' | 'step'
}

export interface DecoderLMBatchArtifacts {
  epoch: number
  batch: number
  batches: number
  step: number
  inputs: Tensor<[number, number], 'f32'>
  tokenIds: Tensor<[number, number], 'f32'>
  logits: Tensor<number[], 'f32'>
  loss: Tensor<[1], 'f32'>
  exportBundleFile: typeof exportBundleFile
}

export interface DecoderLMTrainOptions extends PackedCorpusOptions, TokenBatchOptions {
  epochs: number
  lr?: number
  optimizer?: {
    kind?: 'adam' | 'adamw'
    weightDecay?: number
  }
  gradientAccumulationSteps?: number
  lrSchedule?: (ctx: { epoch: number, step: number }) => number
  maxGradNorm?: number
  forward?: DecoderLMForward
  maxTrainBatchesPerEpoch?: number
  maxEvalBatches?: number
  initialEpoch?: number
  initialStep?: number
  resumeCheckpoint?: SavedDecoderLMCheckpoint
  evaluateInitialTrainLoss?: boolean
  validationBatchSize?: number
  validationWindows?: readonly TokenWindow[]
  checkpointEveryEpochs?: number
  checkpointEverySteps?: number
  checkpointPathPrefix?: string
  checkpointMetadata?: DecoderLMCheckpointMetadata
  keepCheckpointsInMemory?: boolean
  onStatus?: (status: string) => void
  onBatchPhase?: (metrics: DecoderLMBatchPhaseMetrics) => void
  onBatch?: (metrics: DecoderLMBatchMetrics) => void
  onBatchArtifacts?: (artifacts: DecoderLMBatchArtifacts) => void
  onEpoch?: (metrics: DecoderLMEpochMetrics) => void
}

export interface DecoderLMTrainResult {
  steps: number
  initialTrainLoss: number | null
  finalTrainLoss: number
  initialValLoss: number | null
  finalValLoss: number | null
  initialTrainPerplexity: number | null
  finalTrainPerplexity: number
  initialValPerplexity: number | null
  finalValPerplexity: number | null
  history: DecoderLMEpochMetrics[]
  checkpoints: DecoderLMCheckpoint[]
  checkpointPaths: string[]
}

function assertFiniteMetric(name: string, value: number, context: string): void {
  if (!Number.isFinite(value)) {
    throw new AffonError('grad_error', `${name} became non-finite during ${context}`)
  }
}

function formatDiagnosticNumber(value: number): string {
  if (!Number.isFinite(value)) return describeValue(value)
  return value.toFixed(6)
}

function isTensorValue(value: unknown): value is Tensor {
  return !!value && typeof value === 'object' && Array.isArray((value as any).shape) && typeof (value as any).item === 'function'
}

function resolveLearningRate(
  opts: DecoderLMTrainOptions,
  epoch: number,
  step: number,
  fallback: number,
): number {
  if (!opts.lrSchedule) return fallback
  const lr = opts.lrSchedule({ epoch, step })
  if (!Number.isFinite(lr) || lr <= 0) {
    throw new AffonError('invalid_arg', `trainDecoderLM lrSchedule produced invalid lr=${describeValue(lr)} at epoch ${epoch} step ${step}`)
  }
  return lr
}

function errorField(error: unknown, key: string): unknown {
  if (typeof error !== 'object' || error === null) return undefined
  return (error as Record<string, unknown>)[key]
}

function summarizeErrorCause(error: unknown): string {
  const code = errorField(error, 'code')
  const message = errorField(error, 'message')
  const nativeStack = errorField(error, 'nativeStack')

  const codeText = typeof code === 'string' && code.length > 0 ? code : 'unknown'
  const messageText = typeof message === 'string' && message.length > 0
    ? message.replaceAll('\n', ' ')
    : String(error)
  const nativeStackText = typeof nativeStack === 'string' && nativeStack.length > 0
    ? nativeStack.split('\n')[0]
    : 'none'

  return `cause_code=${codeText} cause_message=${JSON.stringify(messageText)} cause_native_stack=${JSON.stringify(nativeStackText)}`
}

function summarizeModelParameters(model: DecoderModelModule, limit = 4): string {
  const rawParams = model.parameters as ReadonlyArray<Tensor> & { named?: () => ReadonlyArray<readonly [string, Tensor]> }
  const named = typeof rawParams.named === 'function'
    ? rawParams.named()
    : rawParams.map((param, index) => [`param_${index}`, param] as const)
  const finite: Array<{ name: string; absMax: number }> = []
  const nonFinite: string[] = []

  for (let i = 0; i < named.length; i++) {
    const [name, param] = named[i]
    const absMax = tensorAbsMax(param)
    if (absMax !== null && Number.isFinite(absMax)) {
      finite.push({ name, absMax })
    } else {
      nonFinite.push(name)
    }
  }

  finite.sort((a, b) => b.absMax - a.absMax)
  const topFinite = finite
    .slice(0, limit)
    .map((entry) => `${entry.name}:${formatDiagnosticNumber(entry.absMax)}`)
    .join(', ')
  const nonFiniteSummary = nonFinite.length > 0 ? nonFinite.join(', ') : 'none'

  return `param_nonfinite=[${nonFiniteSummary}] param_top_abs=[${topFinite}]`
}

function summarizeModelGradients(model: DecoderModelModule, limit = 4): string {
  const rawParams = model.parameters as ReadonlyArray<Tensor> & { named?: () => ReadonlyArray<readonly [string, Tensor]> }
  const named = typeof rawParams.named === 'function'
    ? rawParams.named()
    : rawParams.map((param, index) => [`param_${index}`, param] as const)
  const finite: Array<{ name: string; absMax: number }> = []
  const nonFinite: string[] = []
  const missing: string[] = []

  for (let i = 0; i < named.length; i++) {
    const [name, param] = named[i]
    const grad = param.grad
    if (!isTensorValue(grad)) {
      missing.push(name)
      continue
    }
    const absMax = tensorAbsMax(grad)
    if (absMax !== null && Number.isFinite(absMax)) {
      finite.push({ name, absMax })
    } else {
      nonFinite.push(name)
    }
  }

  finite.sort((a, b) => b.absMax - a.absMax)
  const topFinite = finite
    .slice(0, limit)
    .map((entry) => `${entry.name}:${formatDiagnosticNumber(entry.absMax)}`)
    .join(', ')
  const nonFiniteSummary = nonFinite.length > 0 ? nonFinite.join(', ') : 'none'
  const missingSummary = missing.length > 0 ? missing.slice(0, limit).join(', ') : 'none'

  return `grad_nonfinite=[${nonFiniteSummary}] grad_missing=[${missingSummary}] grad_top_abs=[${topFinite}]`
}

function buildNonFiniteLossDiagnostics(
  phase: string,
  context: string,
  model: DecoderModelModule,
  tokenIds: Tensor<[number, number], 'f32'>,
  logits: Tensor<number[], 'f32'>,
  batchLoss: number,
): string {
  const logitsSummary = tensorFiniteSummary(logits)
  const logitsAbsMax = tensorAbsMax(logits)
  const parameterSummary = summarizeModelParameters(model)

  return `${phase} batch loss became non-finite during ${context}; loss=${formatDiagnosticNumber(batchLoss)} logits_shape=${JSON.stringify(logits.shape)} logits_first_bad_flat_index=${logitsSummary.first_bad_flat_index} logits_first_bad_value=${describeValue(logitsSummary.first_bad_value)} logits_abs_max=${logitsAbsMax === null ? 'unavailable' : formatDiagnosticNumber(logitsAbsMax)} batch_shape=${JSON.stringify(tokenIds.shape)} ${parameterSummary}`
}

function buildBackwardFailureDiagnostics(
  context: string,
  model: DecoderModelModule,
  tokenIds: Tensor<[number, number], 'f32'>,
  logits: Tensor<number[], 'f32'>,
  batchLoss: number,
  cause: unknown,
): string {
  const logitsSummary = tensorFiniteSummary(logits)
  const logitsAbsMax = tensorAbsMax(logits)
  const parameterSummary = summarizeModelParameters(model)
  const gradientSummary = summarizeModelGradients(model)
  const causeSummary = summarizeErrorCause(cause)

  return `trainDecoderLM backward failed during ${context}; loss=${formatDiagnosticNumber(batchLoss)} logits_shape=${JSON.stringify(logits.shape)} logits_first_bad_flat_index=${logitsSummary.first_bad_flat_index} logits_first_bad_value=${describeValue(logitsSummary.first_bad_value)} logits_abs_max=${logitsAbsMax === null ? 'unavailable' : formatDiagnosticNumber(logitsAbsMax)} batch_shape=${JSON.stringify(tokenIds.shape)} ${parameterSummary} ${gradientSummary} ${causeSummary}`
}

interface DecoderLMCheckpointManifest {
  format: 'affon-transformers-decoder-lm-checkpoint/v1' | 'affon-transformers-decoder-lm-checkpoint/v2'
  epoch: number
  step: number
  trainLoss: number
  valLoss: number | null
  statePath: string
  optimizerStatePath?: string | null
  optimizerStateKind?: string | null
  optimizerStateScalars?: Record<string, number> | null
  basePath?: string | null
  metadata: DecoderLMCheckpointMetadata | null
  resumeEpoch?: number
  resumeBatchIndex?: number
  batchOrder?: number[] | null
  shuffleState?: number | null
  epochLossAccum?: number | null
}

function scalar1(value: Tensor<[1], 'f32'>): number {
  return Number(value.item())
}

function shouldReportDetailedEvalBatch(batch: number, batches: number): boolean {
  return batch === 1 || batch === batches || batch % 10 === 0
}

export function perplexityFromLoss(loss: number): number {
  assertFiniteMetric('perplexityFromLoss input loss', loss, 'perplexity computation')
  const perplexity = Math.exp(loss)
  assertFiniteMetric('perplexityFromLoss result', perplexity, 'perplexity computation')
  return perplexity
}

interface TrainPrng {
  state: number
}

function createTrainPrng(seed?: number | null): TrainPrng {
  const fallback = Math.floor(Math.random() * 0x100000000)
  const initial = seed === undefined || seed === null ? fallback : seed
  return { state: (initial >>> 0) || 1 }
}

function nextTrainPrng(prng: TrainPrng): number {
  prng.state = (Math.imul(prng.state, 1664525) + 1013904223) >>> 0
  return prng.state
}

function shuffleInPlace<T>(values: T[], prng: TrainPrng): void {
  for (let i = values.length - 1; i > 0; i--) {
    const j = nextTrainPrng(prng) % (i + 1)
    const tmp = values[i]
    values[i] = values[j]
    values[j] = tmp
  }
}

function cloneStateTree<T>(state: T): T {
  return cloneStateTreeToDevice(state)
}

function cloneStateTreeToDevice<T>(
  state: T,
  device?: 'cpu' | 'metal',
): T {
  return no_grad(function(): T {
    const cloneValue = (value: unknown): unknown => {
      if (isTensorValue(value)) {
        const clone = empty(value.shape as number[], {
          dtype: value.dtype as 'f32' | 'f64' | 'i64',
          device: device ?? value.device as 'cpu' | 'metal',
        })
        copy(clone, value)
        return clone
      }
      if (Array.isArray(value)) return value.map((entry) => cloneValue(entry))
      if (value && typeof value === 'object') {
        const out: Record<string, unknown> = {}
        const keys = Object.keys(value as Record<string, unknown>)
        for (let i = 0; i < keys.length; i++) {
          const key = keys[i]
          out[key] = cloneValue((value as Record<string, unknown>)[key])
        }
        return out
      }
      return value
    }
    return cloneValue(state) as T
  })
}

function cloneOptimizerState(state: DecoderLMOptimizerState): DecoderLMOptimizerState {
  return cloneOptimizerStateToDevice(state)
}

function cloneOptimizerStateToDevice(
  state: DecoderLMOptimizerState,
  device?: 'cpu' | 'metal',
): DecoderLMOptimizerState {
  return {
    kind: state.kind,
    scalars: { ...state.scalars },
    tensors: cloneStateTreeToDevice(state.tensors, device),
  }
}

function batchToTensor(batch: readonly TokenWindow[]): Tensor<[number, number], 'f32'> {
  return tensor(batch.map((row) => row.slice()), { dtype: 'f32', axes: [axes.batch, axes.token] }) as Tensor<[number, number], 'f32'>
}

function batchOrder(count: number, shuffle: boolean, prng: TrainPrng): number[] {
  const order = Array.from({ length: count }, (_, index) => index)
  if (shuffle) shuffleInPlace(order, prng)
  return order
}

function batchFromOrder(
  windows: readonly TokenWindow[],
  order: readonly number[],
  start: number,
  batchSize: number,
): Tensor<[number, number], 'f32'> {
  const rows: TokenWindow[] = []
  const end = Math.min(start + batchSize, order.length)
  for (let i = start; i < end; i++) {
    rows.push(windows[order[i]].slice())
  }
  return batchToTensor(rows)
}

function createCheckpointSnapshot(
  model: DecoderModelModule,
  optimizerState: DecoderLMOptimizerState,
  keepCheckpointsInMemory: boolean,
): {
  state: DecoderModelState
  optimizerState: DecoderLMOptimizerState
} {
  if (keepCheckpointsInMemory) {
    return {
      state: cloneStateTree(model.state()) as DecoderModelState,
      optimizerState: cloneOptimizerState(optimizerState),
    }
  }

  // Disk-only checkpoints should not stage another full Metal copy of the
  // model and optimizer state. Snapshot them on CPU so the live training
  // allocations can be released immediately after serialization.
  return {
    state: cloneStateTreeToDevice(model.state(), 'cpu') as DecoderModelState,
    optimizerState: cloneOptimizerStateToDevice(optimizerState, 'cpu'),
  }
}

function assertWindowLength(
  windows: readonly TokenWindow[],
  seqLen: number,
  name: string,
): void {
  const expected = seqLen + 1
  for (let i = 0; i < windows.length; i++) {
    if (windows[i].length !== expected) {
      throw new AffonError('shape_mismatch', `${name} expects token windows of length ${expected}`)
    }
  }
}

export function saveDecoderLMCheckpoint(
  pathPrefix: string,
  checkpoint: DecoderLMCheckpoint,
  opts?: { metadata?: DecoderLMCheckpointMetadata | null },
): void {
  if (pathPrefix.length === 0) {
    throw new AffonError('invalid_arg', 'saveDecoderLMCheckpoint pathPrefix must be non-empty')
  }
  const manifest: DecoderLMCheckpointManifest = {
    format: checkpoint.optimizerState || checkpoint.resumeEpoch !== undefined
      ? 'affon-transformers-decoder-lm-checkpoint/v2'
      : 'affon-transformers-decoder-lm-checkpoint/v1',
    epoch: checkpoint.epoch,
    step: checkpoint.step,
    trainLoss: checkpoint.trainLoss,
    valLoss: checkpoint.valLoss,
    statePath: '',
    optimizerStatePath: checkpoint.optimizerState ? 'optimizer' : null,
    optimizerStateKind: checkpoint.optimizerState?.kind ?? null,
    optimizerStateScalars: checkpoint.optimizerState?.scalars ?? null,
    metadata: opts?.metadata ?? null,
    resumeEpoch: checkpoint.resumeEpoch,
    resumeBatchIndex: checkpoint.resumeBatchIndex,
    batchOrder: checkpoint.batchOrder ?? null,
    shuffleState: checkpoint.shuffleState ?? null,
    epochLossAccum: checkpoint.epochLossAccum ?? null,
  }
  checkpointModule.saveBundle(pathPrefix, {
    state: checkpoint.state,
    tensorGroups: checkpoint.optimizerState
      ? { optimizer: checkpoint.optimizerState.tensors }
      : undefined,
    manifest: manifest as unknown as Record<string, unknown>,
  })
}

function legacyDecoderLMStateKey(key: string): string {
  return key
    .replaceAll('tokenEmbedding', 'token_embedding')
    .replaceAll('positionEmbedding', 'position_embedding')
    .replaceAll('attnNorm', 'attn_norm')
    .replaceAll('selfAttention', 'self_attention')
    .replaceAll('qProj', 'q_proj')
    .replaceAll('kProj', 'k_proj')
    .replaceAll('vProj', 'v_proj')
    .replaceAll('outProj', 'out_proj')
    .replaceAll('ffNorm', 'ff_norm')
    .replaceAll('feedForward', 'feed_forward')
    .replaceAll('finalNorm', 'final_norm')
}

function normalizeLegacyDecoderLMStateKeys(state: Record<string, Tensor>): Record<string, Tensor> {
  const out: Record<string, Tensor> = {}
  for (const [key, value] of Object.entries(state)) {
    out[legacyDecoderLMStateKey(key)] = value
  }
  return out
}

export function loadDecoderLMCheckpoint(
  pathPrefix: string,
  model: DecoderModelModule,
): SavedDecoderLMCheckpoint {
  if (pathPrefix.length === 0) {
    throw new AffonError('invalid_arg', 'loadDecoderLMCheckpoint pathPrefix must be non-empty')
  }
  const bundle = checkpointModule.loadBundle(pathPrefix)
  const manifest = bundle.manifest as unknown as DecoderLMCheckpointManifest
  if (
    manifest.format !== 'affon-transformers-decoder-lm-checkpoint/v1'
    && manifest.format !== 'affon-transformers-decoder-lm-checkpoint/v2'
  ) {
    throw new AffonError('invalid_arg', `Unsupported decoder lm checkpoint format: ${String((manifest as any).format)}`)
  }
  try {
    checkpointModule.restore(model, bundle.state)
  } catch (error) {
    checkpointModule.restore(model, normalizeLegacyDecoderLMStateKeys(bundle.state as Record<string, Tensor>))
  }
  const state = cloneStateTreeToDevice(model.state(), 'cpu')
  const optimizerState = manifest.optimizerStatePath
    ? {
      kind: manifest.optimizerStateKind ?? 'unknown',
      scalars: manifest.optimizerStateScalars ?? {},
      tensors: (bundle.tensorGroups.optimizer ?? {}) as Record<string, Tensor>,
    }
    : null
  return {
    epoch: manifest.epoch,
    step: manifest.step,
    trainLoss: manifest.trainLoss,
    valLoss: manifest.valLoss,
    state,
    metadata: manifest.metadata,
    optimizerState,
    resumeEpoch: manifest.resumeEpoch ?? (manifest.epoch + 1),
    resumeBatchIndex: manifest.resumeBatchIndex ?? 0,
    batchOrder: manifest.batchOrder ?? null,
    shuffleState: manifest.shuffleState ?? null,
    epochLossAccum: manifest.epochLossAccum ?? 0,
  }
}

export function evaluateDecoderLM(
  model: DecoderModelModule,
  windows: readonly TokenWindow[],
  opts: { batchSize: number; seqLen?: number; statusLabel?: string; onStatus?: (status: string) => void; maxBatches?: number; forward?: DecoderLMForward },
): number {
  return telemetry.trace('runtime/training/evaluate_decoder_lm', function(): number {
    if (windows.length === 0) {
      throw new AffonError('invalid_arg', 'evaluateDecoderLM requires at least one token window')
    }
    if (opts.maxBatches !== undefined && (!Number.isInteger(opts.maxBatches) || opts.maxBatches <= 0)) {
      throw new AffonError('invalid_arg', 'evaluateDecoderLM maxBatches must be a positive integer')
    }
    if (opts.seqLen !== undefined) {
      assertWindowLength(windows, opts.seqLen, 'evaluateDecoderLM')
    }
    const criterion = CausalLMLoss()
    const forward: DecoderLMForward = opts.forward ?? ((tokenIds) => model(tokenIds) as Tensor<number[], 'f32'>)
    model.eval?.()
    let totalLoss = 0
    const order = batchOrder(windows.length, false, createTrainPrng(0))
    const allBatches = Math.ceil(order.length / opts.batchSize)
    const batches = opts.maxBatches === undefined ? allBatches : Math.min(allBatches, opts.maxBatches)
    let batchCount = 0
    for (let batchIndex = 0; batchIndex < batches; batchIndex++) {
      const i = batchIndex * opts.batchSize
      const batch = batchIndex + 1
      totalLoss += no_grad(function(): number {
        const evalTrace = telemetry.startTrace('compute/eval/batch')
        try {
        if (opts.statusLabel && shouldReportDetailedEvalBatch(batch, batches)) {
          opts.onStatus?.(`${opts.statusLabel} ${batch}/${batches}...`)
        }
        let tokenIds: Tensor<[number, number], 'f32'> | null = batchFromOrder(windows, order, i, opts.batchSize)
        let inputs: Tensor<[number, number], 'f32'> | null = tokenIds.slice([':', `0:${tokenIds.shape[1] as number - 1}`]) as Tensor<[number, number], 'f32'>
        let logits: Tensor<number[], 'f32'> | null = null
        const forwardTrace = telemetry.startSpan('compute/eval/forward')
        try {
          logits = forward(inputs as Tensor<[number, number], 'f32'>)
        } finally {
          forwardTrace.end()
        }
        let loss: Tensor<[1], 'f32'> | null = null
        const lossTrace = telemetry.startSpan('compute/eval/loss')
        try {
          loss = causal_lm_eval_loss_forward(logits as Tensor<number[], 'f32'>, tokenIds as Tensor<number[], 'f32'>)
        } finally {
          lossTrace.end()
        }
        const batchLoss = scalar1(loss)
        if (!Number.isFinite(batchLoss)) {
          throw new AffonError('grad_error', buildNonFiniteLossDiagnostics(
            'evaluateDecoderLM',
            `${opts.statusLabel ?? 'evaluation'} batch ${batch}/${batches}`,
            model,
            tokenIds as Tensor<[number, number], 'f32'>,
            logits as Tensor<number[], 'f32'>,
            batchLoss,
          ))
        }
        loss = null
        logits = null
        inputs = null
        tokenIds = null
        return batchLoss
        } finally {
          evalTrace.end()
        }
      })
      batchCount++
    }
    model.train?.()
    const meanLoss = totalLoss / batchCount
    assertFiniteMetric('evaluateDecoderLM mean loss', meanLoss, opts.statusLabel ?? 'evaluation')
    return meanLoss
  })
}

export function trainDecoderLM(
  model: DecoderModelModule,
  windows: readonly TokenWindow[],
  opts: DecoderLMTrainOptions,
): DecoderLMTrainResult {
  return telemetry.trace('runtime/training/train_decoder_lm', function(): DecoderLMTrainResult {
    if (!Number.isInteger(opts.epochs) || opts.epochs <= 0) {
      throw new AffonError('invalid_arg', 'trainDecoderLM epochs must be a positive integer')
    }
    if (!Number.isInteger(opts.seqLen) || opts.seqLen <= 0) {
      throw new AffonError('invalid_arg', 'trainDecoderLM seqLen must be a positive integer')
    }
    if (windows.length === 0) {
      throw new AffonError('invalid_arg', 'trainDecoderLM requires at least one token window')
    }
    assertWindowLength(windows, opts.seqLen, 'trainDecoderLM')
    if (opts.validationWindows && opts.validationWindows.length > 0) {
      assertWindowLength(opts.validationWindows, opts.seqLen, 'trainDecoderLM')
    }
    if (opts.maxTrainBatchesPerEpoch !== undefined && (!Number.isInteger(opts.maxTrainBatchesPerEpoch) || opts.maxTrainBatchesPerEpoch <= 0)) {
      throw new AffonError('invalid_arg', 'trainDecoderLM maxTrainBatchesPerEpoch must be a positive integer')
    }
    if (opts.maxEvalBatches !== undefined && (!Number.isInteger(opts.maxEvalBatches) || opts.maxEvalBatches <= 0)) {
      throw new AffonError('invalid_arg', 'trainDecoderLM maxEvalBatches must be a positive integer')
    }
    const gradientAccumulationSteps = opts.gradientAccumulationSteps ?? 1
    if (!Number.isInteger(gradientAccumulationSteps) || gradientAccumulationSteps <= 0) {
      throw new AffonError('invalid_arg', 'trainDecoderLM gradientAccumulationSteps must be a positive integer')
    }
    const checkpointEveryEpochs = opts.checkpointEveryEpochs ?? 0
    const checkpointEverySteps = opts.checkpointEverySteps ?? 0
    const keepCheckpointsInMemory = opts.keepCheckpointsInMemory ?? true
    if (checkpointEveryEpochs < 0 || !Number.isInteger(checkpointEveryEpochs)) {
      throw new AffonError('invalid_arg', 'trainDecoderLM checkpointEveryEpochs must be a non-negative integer')
    }
    if (checkpointEverySteps < 0 || !Number.isInteger(checkpointEverySteps)) {
      throw new AffonError('invalid_arg', 'trainDecoderLM checkpointEverySteps must be a non-negative integer')
    }

    const criterion = CausalLMLoss()
    const forward: DecoderLMForward = opts.forward ?? ((tokenIds) => model(tokenIds) as Tensor<number[], 'f32'>)
    opts.onStatus?.('initializing optimizer...')
    const params = model.parameters as readonly Parameter[]
    const optimizerKind = opts.optimizer?.kind ?? 'adam'
    const weightDecay = opts.optimizer?.weightDecay ?? 0
    if (!Number.isFinite(weightDecay) || weightDecay < 0) {
      throw new AffonError('invalid_arg', `trainDecoderLM optimizer weightDecay must be a non-negative finite number, got ${describeValue(weightDecay)}`)
    }
    const optimizer = optimizerKind === 'adamw'
      ? adamw({ lr: opts.lr ?? 0.001, weight_decay: weightDecay })
      : adam({ lr: opts.lr ?? 0.001 })
    const history: DecoderLMEpochMetrics[] = []
    const checkpoints: DecoderLMCheckpoint[] = []
    const checkpointPathsWritten: string[] = []
    let lastCheckpointStepSaved = 0
    const resumed = opts.resumeCheckpoint ?? null
    if (resumed?.optimizerState) {
      optimizer.restore(resumed.optimizerState)
    }
    let step = resumed?.step ?? (opts.initialStep ?? 0)
    const initialEpoch = resumed?.epoch ?? (opts.initialEpoch ?? 0)
    const trainPrng = createTrainPrng(resumed?.shuffleState ?? opts.shuffleSeed ?? null)
    const startEpoch = resumed?.resumeEpoch ?? (initialEpoch + 1)
    const finalEpoch = resumed
      ? startEpoch + opts.epochs - 1
      : initialEpoch + opts.epochs

    model.train?.()
    let initialTrainLoss: number | null = null
    const validationBatchSize = opts.validationBatchSize ?? opts.batchSize
    if (opts.evaluateInitialTrainLoss ?? true) {
      opts.onStatus?.('evaluating initial train loss...')
      initialTrainLoss = evaluateDecoderLM(model, windows, {
        batchSize: opts.batchSize,
        seqLen: opts.seqLen,
        statusLabel: 'evaluating initial train loss',
        onStatus: opts.onStatus,
        maxBatches: opts.maxEvalBatches,
        forward,
      })
    }
    let initialValLoss: number | null = null
    if (opts.validationWindows && opts.validationWindows.length > 0) {
      opts.onStatus?.('evaluating initial validation loss...')
      initialValLoss = evaluateDecoderLM(model, opts.validationWindows, {
        batchSize: validationBatchSize,
        seqLen: opts.seqLen,
        statusLabel: 'evaluating initial validation loss',
        onStatus: opts.onStatus,
        maxBatches: opts.maxEvalBatches,
        forward,
      })
    }
    opts.onStatus?.('starting training loop...')

    for (let epoch = startEpoch; epoch <= finalEpoch; epoch++) {
      const order = resumed && epoch === resumed.resumeEpoch && resumed.batchOrder
        ? resumed.batchOrder.slice()
        : batchOrder(windows.length, opts.shuffle ?? true, trainPrng)
      const allBatches = Math.ceil(order.length / opts.batchSize)
      const batches = opts.maxTrainBatchesPerEpoch === undefined ? allBatches : Math.min(allBatches, opts.maxTrainBatchesPerEpoch)
      let epochLoss = resumed && epoch === resumed.resumeEpoch ? resumed.epochLossAccum : 0
      const startBatchIndex = resumed && epoch === resumed.resumeEpoch ? resumed.resumeBatchIndex : 0
      for (let batchIndex = startBatchIndex; batchIndex < batches; batchIndex++) {
        const i = batchIndex * opts.batchSize
        const batch = batchIndex + 1
        const accumulationIndex = batchIndex % gradientAccumulationSteps
        const accumulationWindowStart = batchIndex - accumulationIndex
        const accumulationWindowSize = Math.min(gradientAccumulationSteps, batches - accumulationWindowStart)
        const accumulationWindowOffset = batchIndex - accumulationWindowStart
        const shouldStepOptimizer = accumulationWindowOffset + 1 === accumulationWindowSize
        const stepTrace = telemetry.startTrace('compute/train/step')
        try {
          if (accumulationIndex === 0) {
            clear_grad(params)
          }
          opts.onBatchPhase?.({ epoch, batch, batches, step, phase: 'forward' })
          let tokenIds: Tensor<[number, number], 'f32'> | null = batchFromOrder(windows, order, i, opts.batchSize)
          let inputs: Tensor<[number, number], 'f32'> | null = tokenIds.slice([':', `0:${tokenIds.shape[1] as number - 1}`]) as Tensor<[number, number], 'f32'>
          let logits: Tensor<number[], 'f32'> | null = null
          const forwardTrace = telemetry.startSpan('compute/train/forward')
          try {
            logits = forward(inputs as Tensor<[number, number], 'f32'>)
          } finally {
            forwardTrace.end()
          }
          let loss: Tensor<[1], 'f32'> | null = null
          const lossTrace = telemetry.startSpan('compute/train/loss')
          try {
            loss = criterion(logits as Tensor<number[], 'f32'>, tokenIds as Tensor<number[], 'f32'>)
          } finally {
            lossTrace.end()
          }
          opts.onBatchArtifacts?.({
            epoch,
            batch,
            batches,
            step,
            inputs,
            tokenIds,
            logits: logits as Tensor<number[], 'f32'>,
            loss: loss as Tensor<[1], 'f32'>,
            exportBundleFile,
          })
          const batchLoss = scalar1(loss)
          if (!Number.isFinite(batchLoss)) {
            throw new AffonError('grad_error', buildNonFiniteLossDiagnostics(
              'trainDecoderLM',
              `epoch ${epoch} batch ${batch}/${batches} step ${step}`,
              model,
              tokenIds as Tensor<[number, number], 'f32'>,
              logits as Tensor<number[], 'f32'>,
              batchLoss,
            ))
          }
          epochLoss += batchLoss
          opts.onBatchPhase?.({ epoch, batch, batches, step, phase: 'backward' })
          try {
            let scaledLoss: Tensor<[1], 'f32'> | null = null
            const backwardTrace = telemetry.startSpan('compute/train/backward')
            try {
              scaledLoss = accumulationWindowSize === 1
                ? loss as Tensor<[1], 'f32'>
                : mul(loss as Tensor<[1], 'f32'>, tensor(1 / accumulationWindowSize, { dtype: (loss as Tensor<[1], 'f32'>).dtype })) as Tensor<[1], 'f32'>
              grad(
                accumulationWindowSize === 1
                  ? loss as Tensor<[1], 'f32'>
                  : scaledLoss,
                params,
                { assign: 'accumulate' },
              )
            } finally {
              backwardTrace.end()
            }
          } catch (error) {
            throw new AffonError('grad_error', buildBackwardFailureDiagnostics(
              `epoch ${epoch} batch ${batch}/${batches} step ${step}`,
              model,
              tokenIds as Tensor<[number, number], 'f32'>,
              logits as Tensor<number[], 'f32'>,
              batchLoss,
              error,
            ))
          }
          let batchGradNorm: number | null = null
          let batchLr = optimizer.lr
          if (shouldStepOptimizer) {
            if (opts.maxGradNorm !== undefined) {
              const gradNorm = clip_grad_norm(params, opts.maxGradNorm)
              assertFiniteMetric('trainDecoderLM gradient norm', gradNorm, `epoch ${epoch} batch ${batch}/${batches} step ${step}`)
              batchGradNorm = gradNorm
            }
            opts.onBatchPhase?.({ epoch, batch, batches, step, phase: 'step' })
            optimizer.lr = resolveLearningRate(opts, epoch, step, optimizer.lr)
            batchLr = optimizer.lr
            const optimizerTrace = telemetry.startSpan('compute/train/optimizer_step')
            try {
              optimizer(params)
            } finally {
              optimizerTrace.end()
            }
            step++
          }
          loss = null
          logits = null
          inputs = null
          tokenIds = null
          opts.onBatch?.({
            epoch,
            batch,
            batches,
            step,
            batchLoss,
            lr: batchLr,
            gradNorm: batchGradNorm,
            optimizerStepped: shouldStepOptimizer,
          })

          const epochCheckpointDueAtBatchEnd = checkpointEveryEpochs > 0
            && batch === batches
            && epoch % checkpointEveryEpochs === 0
          const stepCheckpointDue = checkpointEverySteps > 0
            && step % checkpointEverySteps === 0
            && step !== lastCheckpointStepSaved
          if (stepCheckpointDue && !epochCheckpointDueAtBatchEnd) {
            const shouldMaterializeCheckpoint = keepCheckpointsInMemory || !!opts.checkpointPathPrefix
            if (shouldMaterializeCheckpoint) {
              const optimizerState = optimizer.state()
              const checkpointSnapshot = createCheckpointSnapshot(
                model,
                optimizerState as DecoderLMOptimizerState,
                keepCheckpointsInMemory,
              )
              const checkpoint: DecoderLMCheckpoint = {
                epoch,
                step,
                trainLoss: batchLoss,
                valLoss: null,
                state: checkpointSnapshot.state,
                optimizerState: checkpointSnapshot.optimizerState,
                resumeEpoch: epoch,
                resumeBatchIndex: batchIndex + 1,
                batchOrder: order.slice(),
                shuffleState: trainPrng.state,
                epochLossAccum: epochLoss,
              }
              if (keepCheckpointsInMemory) {
                checkpoints.push(checkpoint)
              }
              if (opts.checkpointPathPrefix) {
                const path = `${opts.checkpointPathPrefix}-step-${step}`
                saveDecoderLMCheckpoint(path, checkpoint, { metadata: opts.checkpointMetadata ?? null })
                checkpointPathsWritten.push(path)
              }
            }
            lastCheckpointStepSaved = step
          }
        } finally {
          stepTrace.end()
        }
      }

      const trainLoss = epochLoss / batches
      assertFiniteMetric('trainDecoderLM epoch train loss', trainLoss, `epoch ${epoch}`)
      const valLoss = opts.validationWindows && opts.validationWindows.length > 0
        ? evaluateDecoderLM(model, opts.validationWindows, {
          batchSize: validationBatchSize,
          seqLen: opts.seqLen,
          statusLabel: `evaluating validation loss for epoch ${epoch}`,
          onStatus: opts.onStatus,
          maxBatches: opts.maxEvalBatches,
          forward,
        })
        : null
      if (valLoss !== null) assertFiniteMetric('trainDecoderLM validation loss', valLoss, `epoch ${epoch}`)
      const metrics = {
        epoch,
        step,
        trainLoss,
        valLoss,
        trainPerplexity: perplexityFromLoss(trainLoss),
        valPerplexity: valLoss === null ? null : perplexityFromLoss(valLoss),
      }
      history.push(metrics)
      opts.onEpoch?.(metrics)

      if (
        checkpointEveryEpochs > 0
        && epoch % checkpointEveryEpochs === 0
        && step !== lastCheckpointStepSaved
      ) {
        const shouldMaterializeCheckpoint = keepCheckpointsInMemory || !!opts.checkpointPathPrefix
        if (shouldMaterializeCheckpoint) {
          const optimizerState = optimizer.state()
          const checkpointSnapshot = createCheckpointSnapshot(
            model,
            optimizerState as DecoderLMOptimizerState,
            keepCheckpointsInMemory,
          )
          const checkpoint: DecoderLMCheckpoint = {
            epoch,
            step,
            trainLoss,
            valLoss,
            state: checkpointSnapshot.state,
            optimizerState: checkpointSnapshot.optimizerState,
            resumeEpoch: epoch + 1,
            resumeBatchIndex: 0,
            batchOrder: null,
            shuffleState: trainPrng.state,
            epochLossAccum: 0,
          }
          if (keepCheckpointsInMemory) {
            checkpoints.push(checkpoint)
          }
          if (opts.checkpointPathPrefix) {
            const path = `${opts.checkpointPathPrefix}-epoch-${epoch}`
            saveDecoderLMCheckpoint(path, checkpoint, { metadata: opts.checkpointMetadata ?? null })
            checkpointPathsWritten.push(path)
          }
        }
        lastCheckpointStepSaved = step
      }
  }

    const finalMetrics = history[history.length - 1]
    return {
      steps: step,
      initialTrainLoss,
      finalTrainLoss: finalMetrics.trainLoss,
      initialValLoss,
      finalValLoss: finalMetrics.valLoss,
      initialTrainPerplexity: initialTrainLoss === null ? null : perplexityFromLoss(initialTrainLoss),
      finalTrainPerplexity: finalMetrics.trainPerplexity,
      initialValPerplexity: initialValLoss === null ? null : perplexityFromLoss(initialValLoss),
      finalValPerplexity: finalMetrics.valPerplexity,
      history,
      checkpoints,
      checkpointPaths: checkpointPathsWritten,
    }
  })
}
