import checkpoint from 'affon:checkpoint'
import { optimize, program, Tensor, type ExecutionState, type Program, type FormalTensor, type Session } from 'affon:compute'
import { cross_entropy } from 'affon:nn'
import { adam, adamw, type Optimizer } from 'affon:optim'
import { decoderProgram, type DecoderModel } from './model.ts'
import type { PackedCorpusOptions, TokenBatchOptions, TokenWindow } from './data/index.ts'

export type { PackedCorpusOptions, TokenBatchOptions, TokenWindow } from './data/index.ts'
export type DecoderLMCheckpointMetadata = Readonly<Record<string, unknown>>
export type DecoderLMOptimizerState = { kind: string; scalars: Record<string, number>; tensors: Record<string, Tensor> }
export interface DecoderLMRuntime { model: DecoderModel; session: Session; state: ExecutionState }
export interface DecoderLMCheckpoint {
  epoch: number; step: number; trainLoss: number; valLoss: number | null
  state: Record<string, Tensor>; optimizerState?: DecoderLMOptimizerState | null
  resumeEpoch?: number; resumeBatchIndex?: number; epochLossAccum?: number
}
export interface SavedDecoderLMCheckpoint extends DecoderLMCheckpoint { metadata: DecoderLMCheckpointMetadata | null; optimizerState: DecoderLMOptimizerState | null; resumeEpoch: number; resumeBatchIndex: number; epochLossAccum: number }
export interface DecoderLMEpochMetrics { epoch: number; step: number; trainLoss: number; valLoss: number | null; trainPerplexity: number; valPerplexity: number | null }
export interface DecoderLMBatchMetrics { epoch: number; batch: number; batches: number; step: number; batchLoss: number; lr: number; optimizerStepped: boolean }
export interface DecoderLMBatchPhaseMetrics { epoch: number; batch: number; batches: number; step: number; phase: 'forward' | 'step' }

const objective = cross_entropy()
const evaluationLosses = new Map<string, Program<Record<string, FormalTensor>, FormalTensor>>()

function evaluationLoss(batch: number, length: number, vocabulary: number) {
  const key = `${batch}x${length}x${vocabulary}`
  const cached = evaluationLosses.get(key)
  if (cached) return cached
  const source = program('decoder_loss', p => objective({
    input: p.argument('logits', Tensor.f32([batch, length, vocabulary])),
    target: p.argument('labels', Tensor.i64([batch, length], { axes: ['batch', 'token'] })),
  }, 'objective'))
  evaluationLosses.set(key, source)
  return source
}

export interface DecoderLMTrainOptions extends PackedCorpusOptions, TokenBatchOptions {
  epochs: number; lr?: number; optimizer?: { kind?: 'adam' | 'adamw'; weightDecay?: number }
  lrSchedule?: (ctx: { epoch: number; step: number }) => number; maxTrainBatchesPerEpoch?: number; maxEvalBatches?: number
  initialEpoch?: number; initialStep?: number; resumeCheckpoint?: SavedDecoderLMCheckpoint
  evaluateInitialTrainLoss?: boolean; evaluateInitialValidationLoss?: boolean; validationBatchSize?: number; validationWindows?: readonly TokenWindow[]
  checkpointEveryEpochs?: number; checkpointEverySteps?: number; checkpointPathPrefix?: string; checkpointMetadata?: DecoderLMCheckpointMetadata
  keepCheckpointsInMemory?: boolean; onStatus?: (status: string) => void; onBatchPhase?: (metrics: DecoderLMBatchPhaseMetrics) => void
  onBatch?: (metrics: DecoderLMBatchMetrics) => void; onEpoch?: (metrics: DecoderLMEpochMetrics) => void
}
export interface DecoderLMTrainResult {
  steps: number; initialTrainLoss: number | null; finalTrainLoss: number; initialValLoss: number | null; finalValLoss: number | null
  initialTrainPerplexity: number | null; finalTrainPerplexity: number; initialValPerplexity: number | null; finalValPerplexity: number | null
  history: DecoderLMEpochMetrics[]; checkpoints: DecoderLMCheckpoint[]; checkpointPaths: string[]
}

function checkWindows(windows: readonly TokenWindow[], sequenceLength?: number) {
  if (!windows.length) throw new AffonError('invalid_arg', 'decoder training requires at least one token window')
  if (sequenceLength !== undefined && windows.some(window => window.length !== sequenceLength + 1)) throw new AffonError('shape_mismatch', `token windows must have length ${sequenceLength + 1}`)
}
function batches(windows: readonly TokenWindow[], batchSize: number, shuffle: boolean, seed: number): TokenWindow[][] {
  if (!Number.isInteger(batchSize) || batchSize <= 0) throw new AffonError('invalid_arg', 'batchSize must be a positive integer')
  const order = windows.map((_, index) => index)
  let state = seed >>> 0 || 1
  if (shuffle) for (let index = order.length - 1; index > 0; index--) { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; const other = state % (index + 1); [order[index], order[other]] = [order[other], order[index]] }
  const result: TokenWindow[][] = []
  for (let index = 0; index < order.length; index += batchSize) result.push(order.slice(index, index + batchSize).map(position => windows[position].slice()))
  return result
}
function optimizerState(runtime: DecoderLMRuntime, kind: string): DecoderLMOptimizerState {
  const tensors: Record<string, Tensor> = {}, scalars: Record<string, number> = {}
  for (const [name, value] of Object.entries(runtime.state.optimizer_state)) {
    if (value && typeof value === 'object' && 'to_array' in value) tensors[name] = value as Tensor
    else if (typeof value === 'number') scalars[name] = value
  }
  return { kind, scalars, tensors }
}
function clone(values: Readonly<Record<string, Tensor>>, session: Session) {
  return Object.fromEntries(Object.entries(values).map(([name, value]) => [name, session.tensor(value.to_array() as any, { dtype: value.dtype }) as Tensor]))
}
function disposeState(values: Readonly<Record<string, Tensor>>) { for (const value of Object.values(values)) value.dispose() }

export function perplexityFromLoss(loss: number): number {
  const value = Math.exp(loss)
  if (!Number.isFinite(loss) || !Number.isFinite(value)) throw new AffonError('invalid_arg', 'perplexity requires a finite loss')
  return value
}

export function saveDecoderLMCheckpoint(pathPrefix: string, value: DecoderLMCheckpoint, opts?: { metadata?: DecoderLMCheckpointMetadata | null }) {
  if (!pathPrefix) throw new AffonError('invalid_arg', 'checkpoint path must be non-empty')
  checkpoint.saveBundle(pathPrefix, { state: value.state, tensorGroups: value.optimizerState ? { optimizer: value.optimizerState.tensors } : undefined, manifest: {
    format: 'affon-decoder-program-checkpoint/v2', epoch: value.epoch, step: value.step, trainLoss: value.trainLoss, valLoss: value.valLoss,
    optimizerKind: value.optimizerState?.kind ?? null, optimizerScalars: value.optimizerState?.scalars ?? null, metadata: opts?.metadata ?? null,
    resumeEpoch: value.resumeEpoch ?? value.epoch + 1, resumeBatchIndex: value.resumeBatchIndex ?? 0, epochLossAccum: value.epochLossAccum ?? 0,
  } })
}

export function loadDecoderLMCheckpoint(pathPrefix: string, runtime: DecoderLMRuntime): SavedDecoderLMCheckpoint {
  const bundle = checkpoint.loadBundle(pathPrefix), manifest = bundle.manifest as any
  if (!['affon-decoder-program-checkpoint/v1', 'affon-decoder-program-checkpoint/v2'].includes(manifest.format)) throw new AffonError('invalid_arg', 'Unsupported decoder checkpoint format')
  for (const [name, current] of Object.entries(runtime.state.parameters)) {
    const loaded = bundle.state[name]
    if (!loaded || loaded.dtype !== current.dtype || JSON.stringify(loaded.shape) !== JSON.stringify(current.shape)) throw new AffonError('shape_mismatch', `Invalid decoder checkpoint parameter: ${name}`)
    const replacement = runtime.session.tensor(loaded.to_array() as any, { dtype: loaded.dtype })
    current.dispose(); runtime.state.parameters[name] = replacement
  }
  const loadedOptimizer = bundle.tensorGroups.optimizer ?? {}
  const optimizerTensors = Object.fromEntries(Object.entries(loadedOptimizer).map(([name, value]) => [name, runtime.session.tensor(value.to_array() as any, { dtype: value.dtype }) as Tensor]))
  const optimizer: DecoderLMOptimizerState | null = manifest.optimizerKind ? { kind: manifest.optimizerKind, scalars: manifest.optimizerScalars ?? {}, tensors: optimizerTensors } : null
  if (optimizer) { Object.assign(runtime.state.optimizer_state, optimizer.tensors); Object.assign(runtime.state.optimizer_state, optimizer.scalars) }
  for (const value of Object.values(bundle.state)) value.dispose()
  for (const group of Object.values(bundle.tensorGroups)) for (const value of Object.values(group)) value.dispose()
  return { epoch: manifest.epoch, step: manifest.step, trainLoss: manifest.trainLoss, valLoss: manifest.valLoss, state: clone(runtime.state.parameters, runtime.session), metadata: manifest.metadata ?? null, optimizerState: optimizer, resumeEpoch: manifest.resumeEpoch, resumeBatchIndex: manifest.resumeBatchIndex, epochLossAccum: manifest.epochLossAccum }
}

function runLoss(runtime: DecoderLMRuntime, rows: readonly (readonly number[])[], optimizer?: Optimizer): number {
  const batch = rows.length, length = rows[0].length - 1
  const inputs = runtime.session.tensor(rows.map(row => row.slice(0, -1)), { dtype: 'i64' })
  const labels = runtime.session.tensor(rows.map(row => row.slice(1)), { dtype: 'i64' })
  const model = decoderProgram(runtime.model, batch, length)
  if (optimizer) {
    const source = optimize(model, objective, optimizer)
    const result = runtime.session.compile(source).run({ token_ids: inputs, labels }, runtime.state) as Tensor
    try { return result.item() } finally { result.dispose(); inputs.dispose(); labels.dispose() }
  }
  const logits = runtime.session.compile(model).run({ token_ids: inputs }, runtime.state) as Tensor
  const result = runtime.session.compile(evaluationLoss(batch, length, runtime.model.vocabSize)).run({ logits, labels }) as Tensor
  try { return result.item() } finally { result.dispose(); logits.dispose(); inputs.dispose(); labels.dispose() }
}

export function evaluateDecoderLM(runtime: DecoderLMRuntime, windows: readonly TokenWindow[], opts: { batchSize: number; seqLen?: number; statusLabel?: string; onStatus?: (status: string) => void; maxBatches?: number }): number {
  checkWindows(windows, opts.seqLen)
  const groups = batches(windows, opts.batchSize, false, 1).slice(0, opts.maxBatches)
  let total = 0
  for (let index = 0; index < groups.length; index++) { opts.onStatus?.(`${opts.statusLabel ?? 'evaluating'} ${index + 1}/${groups.length}...`); total += runLoss(runtime, groups[index]) }
  return total / groups.length
}

export function trainDecoderLM(runtime: DecoderLMRuntime, windows: readonly TokenWindow[], opts: DecoderLMTrainOptions): DecoderLMTrainResult {
  if (!Number.isInteger(opts.epochs) || opts.epochs <= 0) throw new AffonError('invalid_arg', 'epochs must be a positive integer')
  checkWindows(windows, opts.seqLen)
  const initialTrainLoss = opts.evaluateInitialTrainLoss === false ? null : evaluateDecoderLM(runtime, windows, { batchSize: opts.batchSize, seqLen: opts.seqLen, maxBatches: opts.maxEvalBatches })
  const initialValLoss = opts.evaluateInitialValidationLoss === false || !opts.validationWindows?.length ? null : evaluateDecoderLM(runtime, opts.validationWindows, { batchSize: opts.validationBatchSize ?? opts.batchSize, seqLen: opts.seqLen, maxBatches: opts.maxEvalBatches })
  const history: DecoderLMEpochMetrics[] = [], snapshots: DecoderLMCheckpoint[] = [], checkpointPaths: string[] = []
  let step = opts.resumeCheckpoint?.step ?? opts.initialStep ?? 0
  const firstEpoch = opts.resumeCheckpoint?.resumeEpoch ?? (opts.initialEpoch ?? 0) + 1
  let finalTrainLoss = initialTrainLoss ?? 0, finalValLoss = initialValLoss
  const kind = opts.optimizer?.kind ?? 'adam', weightDecay = opts.optimizer?.weightDecay ?? 0
  for (let offset = 0; offset < opts.epochs; offset++) {
    const epoch = firstEpoch + offset
    const groups = batches(windows, opts.batchSize, opts.shuffle ?? true, (opts.shuffleSeed ?? 0) + epoch).slice(0, opts.maxTrainBatchesPerEpoch)
    const start = offset === 0 ? opts.resumeCheckpoint?.resumeBatchIndex ?? 0 : 0
    let total = offset === 0 ? opts.resumeCheckpoint?.epochLossAccum ?? 0 : 0
    for (let index = start; index < groups.length; index++) {
      const lr = opts.lrSchedule?.({ epoch, step }) ?? opts.lr ?? 0.001
      const selected = kind === 'adamw' ? adamw({ learning_rate: lr, weight_decay: weightDecay }) : adam({ learning_rate: lr })
      opts.onBatchPhase?.({ epoch, batch: index + 1, batches: groups.length, step, phase: 'forward' })
      const loss = runLoss(runtime, groups[index], selected); total += loss; step++
      opts.onBatchPhase?.({ epoch, batch: index + 1, batches: groups.length, step, phase: 'step' })
      opts.onBatch?.({ epoch, batch: index + 1, batches: groups.length, step, batchLoss: loss, lr, optimizerStepped: true })
      if (opts.checkpointEverySteps && step % opts.checkpointEverySteps === 0 && opts.checkpointPathPrefix) {
        const snapshot: DecoderLMCheckpoint = { epoch, step, trainLoss: total / (index + 1), valLoss: null, state: clone(runtime.state.parameters, runtime.session), optimizerState: optimizerState(runtime, kind), resumeEpoch: epoch, resumeBatchIndex: index + 1, epochLossAccum: total }
        const path = `${opts.checkpointPathPrefix}-step-${step}`; let saved = false
        try { saveDecoderLMCheckpoint(path, snapshot, { metadata: opts.checkpointMetadata }); checkpointPaths.push(path); saved = true }
        finally { if (!saved || opts.keepCheckpointsInMemory === false) disposeState(snapshot.state) }
        if (opts.keepCheckpointsInMemory !== false) snapshots.push(snapshot)
      }
    }
    finalTrainLoss = total / groups.length
    finalValLoss = opts.validationWindows?.length ? evaluateDecoderLM(runtime, opts.validationWindows, { batchSize: opts.validationBatchSize ?? opts.batchSize, seqLen: opts.seqLen, maxBatches: opts.maxEvalBatches }) : null
    const metrics = { epoch, step, trainLoss: finalTrainLoss, valLoss: finalValLoss, trainPerplexity: perplexityFromLoss(finalTrainLoss), valPerplexity: finalValLoss === null ? null : perplexityFromLoss(finalValLoss) }
    history.push(metrics); opts.onEpoch?.(metrics)
    if (opts.checkpointEveryEpochs && epoch % opts.checkpointEveryEpochs === 0 && opts.checkpointPathPrefix) {
      const snapshot: DecoderLMCheckpoint = { epoch, step, trainLoss: finalTrainLoss, valLoss: finalValLoss, state: clone(runtime.state.parameters, runtime.session), optimizerState: optimizerState(runtime, kind), resumeEpoch: epoch + 1, resumeBatchIndex: 0, epochLossAccum: 0 }
      const path = `${opts.checkpointPathPrefix}-epoch-${epoch}`; let saved = false
      try { saveDecoderLMCheckpoint(path, snapshot, { metadata: opts.checkpointMetadata }); checkpointPaths.push(path); saved = true }
      finally { if (!saved || opts.keepCheckpointsInMemory === false) disposeState(snapshot.state) }
      if (opts.keepCheckpointsInMemory !== false) snapshots.push(snapshot)
    }
  }
  return { steps: step, initialTrainLoss, finalTrainLoss, initialValLoss, finalValLoss, initialTrainPerplexity: initialTrainLoss === null ? null : perplexityFromLoss(initialTrainLoss), finalTrainPerplexity: perplexityFromLoss(finalTrainLoss), initialValPerplexity: initialValLoss === null ? null : perplexityFromLoss(initialValLoss), finalValPerplexity: finalValLoss === null ? null : perplexityFromLoss(finalValLoss), history, checkpoints: snapshots, checkpointPaths }
}
