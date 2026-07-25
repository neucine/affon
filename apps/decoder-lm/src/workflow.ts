import fs from 'std:fs'
import telemetry from 'std:telemetry'
import { axes, compile, exportBundleFile, tensor } from 'affon:compute'
import type { Tensor } from 'affon:compute'

import {
  createPackedTextCorpusFromConfig,
  DecoderModel,
  generate,
  saveTokenRows,
  type DecoderModelModule,
  type DecoderModelOptions,
  type PackedFileCorpusConfig,
  type PackedTextCorpus,
} from '@affon/lm'
import {
  createHFTokenizerFromFile,
  createLookupTokenizer,
  createSentencePieceTokenizerFromFile,
  type HFTokenizerOptions,
  type LookupTokenizerOptions,
  type SentencePieceTokenizerOptions,
  type SpecialTokens,
  type Tokenizer,
} from '@affon/tokenizers'
import {
  type DecoderLMBatchMetrics,
  type DecoderLMBatchPhaseMetrics,
  loadDecoderLMCheckpoint,
  perplexityFromLoss,
  trainDecoderLM,
  type DecoderLMCheckpointMetadata,
  type DecoderLMEpochMetrics,
  type DecoderLMTrainResult,
} from './training.ts'

function tensorValues(value: any): any {
  if (value && typeof value === 'object' && typeof value.to_array === 'function') {
    return value.to_array()
  }
  return value
}

export type DecoderLMTokenizerConfig =
  | {
    family: 'lookup'
    vocab: Record<string, number>
    specialTokens?: SpecialTokens
    split?: LookupTokenizerOptions['split']
  }
  | {
    family: 'hf-tokenizer-json'
    path: string
    specialTokens?: HFTokenizerOptions['specialTokens']
  }
  | {
    family: 'sentencepiece'
    path: string
    specialTokens?: SentencePieceTokenizerOptions['specialTokens']
  }

export interface DecoderLMWorkflowCorpusConfig extends PackedFileCorpusConfig {}

export interface DecoderLMWorkflowModelConfig extends DecoderModelOptions {
  dModel: number
}

export interface DecoderLMWorkflowTrainingConfig {
  epochs: number
  batchSize: number
  gradientAccumulationSteps?: number
  compileModelForward?: boolean
  shuffleSeed?: number
  validationBatchSize?: number
  lr?: number
  optimizer?: {
    kind?: 'adam' | 'adamw'
    weightDecay?: number
  }
  lrSchedule?: DecoderLMWorkflowLRScheduleConfig
  maxGradNorm?: number
  maxTrainBatchesPerEpoch?: number
  maxEvalBatches?: number
  evaluateInitialTrainLoss?: boolean
  evaluateInitialValidationLoss?: boolean
}

export interface DecoderLMWorkflowStepDurationConfig {
  unit: 'step'
  value: number
}

export interface DecoderLMWorkflowEpochDurationConfig {
  unit: 'epoch'
  value: number
}

export type DecoderLMWorkflowDurationConfig =
  | DecoderLMWorkflowStepDurationConfig
  | DecoderLMWorkflowEpochDurationConfig

export type DecoderLMWorkflowLRScheduleConfig =
  | {
    kind: 'constant'
    lr: number
  }
  | {
    kind: 'warmup_constant'
    start: number
    lr: number
    duration: DecoderLMWorkflowDurationConfig
  }
  | {
    kind: 'warmup_cosine'
    start: number
    peak: number
    end: number
    warmup: DecoderLMWorkflowDurationConfig
    total: DecoderLMWorkflowDurationConfig
  }
  | {
    kind: 'linear'
    start: number
    end: number
    duration: DecoderLMWorkflowDurationConfig
  }
  | {
    kind: 'cosine'
    start: number
    end: number
    duration: DecoderLMWorkflowDurationConfig
  }
  | {
    kind: 'step'
    base: number
    gamma: number
    every: DecoderLMWorkflowDurationConfig
    duration?: DecoderLMWorkflowDurationConfig
  }

export interface DecoderLMWorkflowCheckpointConfig {
  prefix: string
  everyNEpochs?: number
  everyNSteps?: number
  resumeFrom?: string
  resumeStrategy?: 'exact' | 'best_epoch'
  restoreOptimizerState?: boolean
  metadata?: DecoderLMCheckpointMetadata
}

export interface DecoderLMWorkflowSampleConfig {
  prompt: string
  max_new_tokens: number
  everyNEpochs?: number
  temperature?: number
  top_k?: number
  addBos?: boolean
  skipSpecialTokens?: boolean
  banSpecialTokens?: boolean
}

export interface DecoderLMWorkflowProgressConfig {
  trainPhase?: boolean
  trainLossEveryBatches?: number
  runtimeStatsEveryBatches?: number
  evalEveryBatches?: number
  epochSummary?: boolean
}

export interface DecoderLMWorkflowMonitorConfig {
  everyBatches?: number
  maxSamples?: number
  path?: string
}

export interface DecoderLMWorkflowExportConfig {
  dir: string
  everyNSteps?: number
  format?: 'text' | 'json'
  mode?: 'summary' | 'annotated'
  includeForward?: boolean
  includeBackward?: boolean
  prefix?: string
  runId?: string
  graphIdPrefix?: string
}

export interface DecoderLMWorkflowReportConfig {
  summaryPath?: string
  progress?: DecoderLMWorkflowProgressConfig
  sample?: DecoderLMWorkflowSampleConfig
  monitor?: DecoderLMWorkflowMonitorConfig
  export?: DecoderLMWorkflowExportConfig
}

export interface DecoderLMWorkflowConfig {
  device?: 'cpu' | 'metal'
  tokenizer: DecoderLMTokenizerConfig
  corpus: DecoderLMWorkflowCorpusConfig
  model: DecoderLMWorkflowModelConfig
  training: DecoderLMWorkflowTrainingConfig
  checkpoint?: DecoderLMWorkflowCheckpointConfig
  report?: DecoderLMWorkflowReportConfig
}

export interface DecoderLMWorkflowSample {
  epoch: number
  promptIds: number[]
  generatedIds: number[]
  promptText: string
  generatedText: string
}

export type DecoderLMWorkflowMetricSnapshot = ReturnType<typeof telemetry.metrics>[number]

export interface DecoderLMWorkflowRuntimeSnapshot {
  phase: 'batch' | 'epoch'
  epoch: number
  batch: number
  batches: number
  step: number
  runtimeMetrics: DecoderLMWorkflowMetricSnapshot[]
  unattributedBytes: number
}

export interface DecoderLMWorkflowMonitorSummary {
  everyBatches: number
  maxSamples: number
  droppedSamples: number
  path: string | null
  snapshots: DecoderLMWorkflowRuntimeSnapshot[]
}

export interface DecoderLMWorkflowSummary {
  tokenizerFamily: Tokenizer['family']
  modelForwardMode: 'eager' | 'eager-forward' | 'graph'
  modelForwardGraphRuntime: string | null
  modelForwardGraphCaptureError: string | null
  modelForwardGraphLoweringAnalysis: any | null
  trainRows: number
  validationRows: number
  trainWindows: number
  validationWindows: number
  initialTrainLoss: number | null
  finalTrainLoss: number
  initialValLoss: number | null
  finalValLoss: number | null
  initialTrainPerplexity: number | null
  finalTrainPerplexity: number
  initialValPerplexity: number | null
  finalValPerplexity: number | null
  checkpointPaths: string[]
  bundleExportPaths: string[]
  history: DecoderLMEpochMetrics[]
  samples: DecoderLMWorkflowSample[]
  monitor: DecoderLMWorkflowMonitorSummary | null
}

export interface DecoderLMWorkflowResult {
  tokenizer: Tokenizer
  corpus: PackedTextCorpus
  model: DecoderModelModule
  compiledModel: DecoderModelModule | null
  training: DecoderLMTrainResult
  samples: DecoderLMWorkflowSample[]
  monitor: DecoderLMWorkflowMonitorSummary | null
  summary: DecoderLMWorkflowSummary
}

export interface DecoderLMWorkflowHooks {
  onStatus?: (status: string) => void
  onBatchPhase?: (metrics: DecoderLMBatchPhaseMetrics) => void
  onBatch?: (metrics: DecoderLMBatchMetrics) => void
  onEpoch?: (metrics: DecoderLMEpochMetrics) => void
  onSample?: (sample: DecoderLMWorkflowSample) => void
}

function optimizerStateKindForConfig(config: DecoderLMWorkflowTrainingConfig): string {
  return (config.optimizer?.kind ?? 'adam') === 'adamw' ? 'AdamW' : 'Adam'
}

export function loadDecoderLMWorkflowConfig(path: string): DecoderLMWorkflowConfig {
  return JSON.parse(fs.readFileSync(path)) as DecoderLMWorkflowConfig
}

export function createTokenizerFromConfig(config: DecoderLMTokenizerConfig): Tokenizer {
  switch (config.family) {
    case 'lookup':
      return createLookupTokenizer(config.vocab, {
        specialTokens: config.specialTokens,
        split: config.split,
      })
    case 'hf-tokenizer-json':
      return createHFTokenizerFromFile(config.path, {
        specialTokens: config.specialTokens,
      })
    case 'sentencepiece':
      return createSentencePieceTokenizerFromFile(config.path, {
        specialTokens: config.specialTokens,
      })
  }
}

function workflowSummaryPath(config: DecoderLMWorkflowConfig): string | null {
  return config.report?.summaryPath
    ?? (config.checkpoint?.prefix ? `${config.checkpoint.prefix}-summary.json` : null)
}

function bestEpochFromSummary(summary: DecoderLMWorkflowSummary): number {
  if (!summary.history.length) {
    throw new AffonError('invalid_arg', 'workflow best_epoch resume requires a non-empty run summary history')
  }

  const hasValidationLoss = summary.history.some((entry) => entry.valLoss !== null)
  let bestEpoch = summary.history[0].epoch
  let bestLoss = hasValidationLoss
    ? (summary.history[0].valLoss ?? Number.POSITIVE_INFINITY)
    : summary.history[0].trainLoss

  for (let i = 1; i < summary.history.length; i++) {
    const entry = summary.history[i]
    const candidateLoss = hasValidationLoss
      ? (entry.valLoss ?? Number.POSITIVE_INFINITY)
      : entry.trainLoss
    if (candidateLoss <= bestLoss) {
      bestLoss = candidateLoss
      bestEpoch = entry.epoch
    }
  }

  if (!Number.isFinite(bestLoss)) {
    throw new AffonError('invalid_arg', 'workflow best_epoch resume could not find a finite checkpoint loss in the run summary')
  }
  return bestEpoch
}

function resolveResumeCheckpointPath(config: DecoderLMWorkflowConfig): string | null {
  const checkpointConfig = config.checkpoint
  if (!checkpointConfig) return null

  const strategy = checkpointConfig.resumeStrategy ?? 'exact'
  if (strategy === 'exact') return checkpointConfig.resumeFrom ?? null

  if (!checkpointConfig.prefix) {
    throw new AffonError('invalid_arg', 'workflow best_epoch resume requires checkpoint.prefix')
  }
  const summaryPath = workflowSummaryPath(config)
  if (!summaryPath) {
    throw new AffonError('invalid_arg', 'workflow best_epoch resume requires a summary path')
  }
  let summary: DecoderLMWorkflowSummary
  try {
    summary = JSON.parse(fs.readFileSync(summaryPath)) as DecoderLMWorkflowSummary
  } catch {
    throw new AffonError('invalid_arg', `workflow best_epoch resume requires an existing summary file at ${summaryPath ?? '<missing>'}`)
  }
  const bestEpoch = bestEpochFromSummary(summary)
  return `${checkpointConfig.prefix}-epoch-${bestEpoch}`
}

function durationFromConfig(config: DecoderLMWorkflowDurationConfig): DecoderLMWorkflowDurationConfig {
  if (!Number.isFinite(config.value) || Math.floor(config.value) !== config.value || config.value <= 0) {
    throw new AffonError('invalid_arg', `workflow lrSchedule duration value must be a positive integer, got ${config.value}`)
  }
  return config
}

function progressForDuration(ctx: { epoch: number, step: number }, duration: DecoderLMWorkflowDurationConfig): number {
  return duration.unit === 'step' ? ctx.step : ctx.epoch
}

function boundedRatio(progress: number, duration: DecoderLMWorkflowDurationConfig): number {
  return Math.min(Math.max(progress / duration.value, 0), 1)
}

function lrScheduleFromConfig(config: DecoderLMWorkflowLRScheduleConfig): (ctx: { epoch: number, step: number }) => number {
  switch (config.kind) {
    case 'constant':
      return function constantSchedule(): number {
        return config.lr
      }
    case 'warmup_constant':
      return function warmupConstantSchedule(ctx): number {
        const duration = durationFromConfig(config.duration)
        const ratio = boundedRatio(progressForDuration(ctx, duration), duration)
        return config.start + (config.lr - config.start) * ratio
      }
    case 'warmup_cosine':
      if (config.warmup.unit !== config.total.unit) {
        throw new AffonError('invalid_arg', 'workflow warmup_cosine requires warmup and total to use the same unit')
      }
      return function warmupCosineSchedule(ctx): number {
        const warmup = durationFromConfig(config.warmup)
        const total = durationFromConfig(config.total)
        if (warmup.value >= total.value) {
          throw new AffonError('invalid_arg', 'workflow warmup_cosine requires warmup duration to be shorter than total duration')
        }
        const progress = progressForDuration(ctx, total)
        if (progress <= warmup.value) {
          const ratio = boundedRatio(progress, warmup)
          return config.start + (config.peak - config.start) * ratio
        }
        const decayProgress = Math.min(Math.max(progress - warmup.value, 0), total.value - warmup.value)
        const decayRatio = decayProgress / (total.value - warmup.value)
        const weight = (1 + Math.cos(Math.PI * decayRatio)) / 2
        return config.end + (config.peak - config.end) * weight
      }
    case 'linear':
      return function linearSchedule(ctx): number {
        const ratio = boundedRatio(progressForDuration(ctx, durationFromConfig(config.duration)), config.duration)
        return config.start + (config.end - config.start) * ratio
      }
    case 'cosine':
      return function cosineSchedule(ctx): number {
        const ratio = boundedRatio(progressForDuration(ctx, durationFromConfig(config.duration)), config.duration)
        const weight = (1 + Math.cos(Math.PI * ratio)) / 2
        return config.end + (config.start - config.end) * weight
      }
    case 'step':
      if (config.duration && config.duration.unit !== config.every.unit) {
        throw new AffonError('invalid_arg', 'workflow step lrSchedule duration must use the same unit as every')
      }
      return function stepSchedule(ctx): number {
        const progress = progressForDuration(ctx, durationFromConfig(config.every))
        const bounded = config.duration ? Math.min(progress, config.duration.value) : progress
        const decayCount = Math.floor(bounded / config.every.value)
        return config.base * Math.pow(config.gamma, decayCount)
      }
  }
}

function specialTokenIds(tokenizer: Tokenizer): number[] {
  return tokenizer.allSpecialTokenIds.slice()
}

function metricValue(
  metrics: readonly DecoderLMWorkflowMetricSnapshot[],
  group: string,
  name: string,
): number {
  const entry = metrics.find((metric) => metric.scope === group && metric.name === name)
  return entry?.value ?? 0
}

function captureRuntimeSnapshot(
  metrics: DecoderLMBatchMetrics | DecoderLMEpochMetrics,
  phase: DecoderLMWorkflowRuntimeSnapshot['phase'],
): DecoderLMWorkflowRuntimeSnapshot {
  const runtimeMetrics = telemetry.metrics()
  const physFootprintBytes = metricValue(runtimeMetrics, 'runtime.memory', 'physical_footprint_bytes')
  const metalLiveBytes = metricValue(runtimeMetrics, 'compute.storage', 'live_metal_bytes')
  const metalPooledBytes = metricValue(runtimeMetrics, 'compute.memory', 'metal_pool_live_bytes')
  const cpuLiveBytes = metricValue(runtimeMetrics, 'compute.storage', 'live_cpu_bytes')
  const qjsMemoryUsedBytes = metricValue(runtimeMetrics, 'runtime.memory', 'qjs_heap_used_bytes')
  const hostMaterializationBytes = 0
  const temporaryWorkspaceBytes = 0
  const autogradStateBytes = 0
  const unattributedBytes = Math.max(
    0,
    physFootprintBytes
      - metalLiveBytes
      - metalPooledBytes
      - cpuLiveBytes
      - hostMaterializationBytes
      - temporaryWorkspaceBytes
      - autogradStateBytes
      - qjsMemoryUsedBytes,
  )

  return {
    phase,
    epoch: metrics.epoch,
    batch: 'batch' in metrics ? metrics.batch : 0,
    batches: 'batches' in metrics ? metrics.batches : 0,
    step: metrics.step,
    runtimeMetrics,
    unattributedBytes,
  }
}

export function trainDecoderLMFromConfig(
  config: DecoderLMWorkflowConfig,
  hooks?: DecoderLMWorkflowHooks,
): DecoderLMWorkflowResult {
  const previousDevice = 'cpu'
  setDevice(config.device ?? 'cpu')
  try {
    hooks?.onStatus?.('building tokenizer...')
    const tokenizer = createTokenizerFromConfig(config.tokenizer)
    hooks?.onStatus?.('building corpus...')
    const corpus = createPackedTextCorpusFromConfig(tokenizer, config.corpus)
    if (config.corpus.tokenCachePath && corpus.trainRows.length > 0) {
      saveTokenRows(config.corpus.tokenCachePath, corpus.trainRows, { key: config.corpus.tokenCacheKey })
    }
    if (config.corpus.validationTokenCachePath && corpus.validationRows.length > 0) {
      saveTokenRows(config.corpus.validationTokenCachePath, corpus.validationRows, { key: config.corpus.validationTokenCacheKey })
    }
    if (corpus.trainWindows.length === 0) {
      throw new AffonError('invalid_arg', 'trainDecoderLMFromConfig requires at least one training window')
    }

    hooks?.onStatus?.('building model...')
    const model = DecoderModel(tokenizer.vocabSize, config.model.dModel, {
      numLayers: config.model.numLayers,
      numHeads: config.model.numHeads,
      hiddenDim: config.model.hiddenDim,
      causal: config.model.causal,
      positional: config.model.positional,
      maxSeqLen: config.model.maxSeqLen,
      tieEmbeddings: config.model.tieEmbeddings,
      dropout: config.model.dropout,
    })
    const compileModelForward = !!config.training.compileModelForward
    const compiledModel = compileModelForward ? compile(model) : null
    const forward = compiledModel
      ? ((tokenIds: Tensor<[number, number], 'f32'>) => compiledModel(tokenIds) as Tensor<number[], 'f32'>)
      : ((tokenIds: Tensor<[number, number], 'f32'>) => model(tokenIds) as Tensor<number[], 'f32'>)
    const lrSchedule = config.training.lrSchedule
      ? lrScheduleFromConfig(config.training.lrSchedule)
      : undefined
    const resumeCheckpointPath = resolveResumeCheckpointPath(config)
    const resumedRaw = resumeCheckpointPath
      ? loadDecoderLMCheckpoint(resumeCheckpointPath, model)
      : null
    const requestedOptimizerKind = optimizerStateKindForConfig(config.training)
    const optimizerKindMismatch = resumedRaw?.optimizerState?.kind
      ? resumedRaw.optimizerState.kind !== requestedOptimizerKind
      : false
    if (optimizerKindMismatch) {
      hooks?.onStatus?.(
        `resume checkpoint optimizer ${resumedRaw?.optimizerState?.kind ?? 'unknown'} does not match requested ${requestedOptimizerKind}; restoring model weights only with fresh optimizer state`,
      )
    }
    const shouldRestoreOptimizerState = config.checkpoint?.restoreOptimizerState !== false && !optimizerKindMismatch
    const resumed = resumedRaw && !shouldRestoreOptimizerState
      ? {
          ...resumedRaw,
          optimizerState: null,
        }
      : resumedRaw

    const samples: DecoderLMWorkflowSample[] = []
    const bundleExportPaths: string[] = []
    const sampleConfig = config.report?.sample
    const samplePromptIds = sampleConfig
      ? tokenizer.encode(sampleConfig.prompt, { addBos: sampleConfig.addBos ?? true })
      : null
    const sampleForbiddenIds = sampleConfig?.banSpecialTokens
      ? specialTokenIds(tokenizer)
      : undefined
    const monitorConfig = config.report?.monitor
    const monitorEveryBatches = monitorConfig?.everyBatches ?? 0
    const monitorMaxSamples = Math.max(1, monitorConfig?.maxSamples ?? 512)
    const monitorPath = monitorConfig?.path ?? null
    const monitorSnapshots: DecoderLMWorkflowRuntimeSnapshot[] = []
    let monitorDroppedSamples = 0
    const persistMonitor = (): void => {
      if (!monitorPath) return
      const monitor: DecoderLMWorkflowMonitorSummary = {
        everyBatches: monitorEveryBatches,
        maxSamples: monitorMaxSamples,
        droppedSamples: monitorDroppedSamples,
        path: monitorPath,
        snapshots: monitorSnapshots.slice(),
      }
      fs.writeFileSync(monitorPath, JSON.stringify(monitor, null, 2))
    }

    const training = trainDecoderLM(model, corpus.trainWindows, {
      seqLen: config.corpus.seqLen,
      batchSize: config.training.batchSize,
      gradientAccumulationSteps: config.training.gradientAccumulationSteps,
      epochs: config.training.epochs,
      lr: config.training.lr,
      optimizer: config.training.optimizer,
      lrSchedule,
      forward,
      shuffleSeed: config.training.shuffleSeed,
      maxGradNorm: config.training.maxGradNorm,
      maxTrainBatchesPerEpoch: config.training.maxTrainBatchesPerEpoch,
      maxEvalBatches: config.training.maxEvalBatches,
      validationBatchSize: config.training.validationBatchSize,
      resumeCheckpoint: resumed ?? undefined,
      initialEpoch: resumed?.epoch,
      initialStep: resumed?.step,
      evaluateInitialTrainLoss: config.training.evaluateInitialTrainLoss ?? false,
      evaluateInitialValidationLoss: config.training.evaluateInitialValidationLoss ?? true,
      checkpointEveryEpochs: config.checkpoint?.everyNEpochs,
      checkpointEverySteps: config.checkpoint?.everyNSteps,
      validationWindows: corpus.validationWindows,
      checkpointPathPrefix: config.checkpoint?.prefix,
      checkpointMetadata: config.checkpoint?.metadata,
      keepCheckpointsInMemory: config.checkpoint?.prefix ? false : true,
      shuffle: config.corpus.shuffle,
      onStatus: (status) => {
        hooks?.onStatus?.(status)
      },
      onBatchPhase: (metrics) => {
        hooks?.onBatchPhase?.(metrics)
      },
      onBatch: (metrics) => {
        if (
          monitorEveryBatches > 0
          && (
            metrics.batch === 1
            || metrics.batch === metrics.batches
            || metrics.batch % monitorEveryBatches === 0
          )
        ) {
          if (monitorSnapshots.length === monitorMaxSamples) {
            monitorSnapshots.shift()
            monitorDroppedSamples += 1
          }
          monitorSnapshots.push(captureRuntimeSnapshot(metrics, 'batch'))
          persistMonitor()
        }
        hooks?.onBatch?.(metrics)
      },
      onBatchArtifacts: (artifacts) => {
        const exportConfig = config.report?.export
        if (!exportConfig) return
        const everyNSteps = exportConfig.everyNSteps ?? 1
        const exportStep = artifacts.step + 1
        if (everyNSteps <= 0 || exportStep % everyNSteps !== 0) return

        const runId = exportConfig.runId ?? 'decoder-lm'
        const graphIdPrefix = exportConfig.graphIdPrefix ?? 'decoder-lm'
        const format = exportConfig.format ?? 'json'
        const mode = exportConfig.mode ?? 'annotated'
        const prefix = exportConfig.prefix ?? runId
        const common = {
          dir: exportConfig.dir,
          format,
          mode,
          boundary: 'step' as const,
          runId,
          epoch: artifacts.epoch,
          step: exportStep,
          batch: artifacts.batch,
          prefix,
        }

        if (exportConfig.includeForward !== false && compiledModel) {
          const forwardExportOpts = {
            ...common,
            phase: 'forward' as const,
            graphId: `${graphIdPrefix}-forward`,
          }
          let forwardExportPath: string | null = null
          try {
            forwardExportPath = exportBundleFile(compiledModel, artifacts.inputs, forwardExportOpts)
          } catch {
            const exportCompiledModel = compile(model)
            forwardExportPath = exportBundleFile(exportCompiledModel, artifacts.inputs, forwardExportOpts)
          }
          bundleExportPaths.push(forwardExportPath)
        }

        if (exportConfig.includeBackward !== false) {
          bundleExportPaths.push(artifacts.exportBundleFile(artifacts.loss, {
            ...common,
            phase: 'backward',
            graphId: `${graphIdPrefix}-backward`,
          }))
        }
      },
      onEpoch: (metrics) => {
        if (monitorEveryBatches > 0 || monitorPath) {
          if (monitorSnapshots.length === monitorMaxSamples) {
            monitorSnapshots.shift()
            monitorDroppedSamples += 1
          }
          monitorSnapshots.push(captureRuntimeSnapshot(metrics, 'epoch'))
          persistMonitor()
        }
        hooks?.onEpoch?.(metrics)
        if (!sampleConfig || !samplePromptIds) return
        const everyNEpochs = sampleConfig.everyNEpochs ?? 1
        if (everyNEpochs <= 0 || metrics.epoch % everyNEpochs !== 0) return
        let sampleInput: Tensor<number[], 'f32'> | null = tensor([samplePromptIds], { dtype: 'f32', axes: [axes.batch, axes.token] }).to(model.embedding.token_embedding.weight.device) as Tensor<number[], 'f32'>
        let generated: Tensor<number[], 'f32'> | null = null
        const sampleTrace = telemetry.startTrace('compute/sample/generate')
        try {
          generated = generate(
            model,
            sampleInput as Tensor<number[], 'f32'>,
            {
              max_new_tokens: sampleConfig.max_new_tokens,
              temperature: sampleConfig.temperature,
              top_k: sampleConfig.top_k,
              forbidden_token_ids: sampleForbiddenIds,
            },
            forward,
          )
        } finally {
          sampleTrace.end()
        }
        const generatedIds = (tensorValues(generated) as number[][])[0]
        const sample = {
          epoch: metrics.epoch,
          promptIds: samplePromptIds.slice(),
          generatedIds,
          promptText: tokenizer.decode(samplePromptIds, {
            skipSpecialTokens: sampleConfig.skipSpecialTokens ?? true,
          }),
          generatedText: tokenizer.decode(generatedIds, {
            skipSpecialTokens: sampleConfig.skipSpecialTokens ?? true,
          }),
        }
        samples.push(sample)
        generated = null
        sampleInput = null
        hooks?.onSample?.(sample)
      },
    })

    const monitor = monitorEveryBatches > 0 || monitorPath
      ? {
        everyBatches: monitorEveryBatches,
        maxSamples: monitorMaxSamples,
        droppedSamples: monitorDroppedSamples,
        path: monitorPath,
        snapshots: monitorSnapshots.slice(),
      }
      : null

    const compiledModelSummary = compiledModel?.summary?.() ?? null
    const summary: DecoderLMWorkflowSummary = {
      tokenizerFamily: tokenizer.family,
      modelForwardMode: (compiledModelSummary?.mode ?? (compiledModel ? 'eager-forward' : 'eager')) as 'eager' | 'eager-forward' | 'graph',
      modelForwardGraphRuntime: compiledModelSummary?.graphRuntime ?? null,
      modelForwardGraphCaptureError: compiledModelSummary?.graphCaptureError ?? null,
      modelForwardGraphLoweringAnalysis: compiledModelSummary?.graphLoweringAnalysis ?? null,
      trainRows: corpus.trainRows.length,
      validationRows: corpus.validationRows.length,
      trainWindows: corpus.trainWindows.length,
      validationWindows: corpus.validationWindows.length,
      initialTrainLoss: training.initialTrainLoss,
      finalTrainLoss: training.finalTrainLoss,
      initialValLoss: training.initialValLoss,
      finalValLoss: training.finalValLoss,
      initialTrainPerplexity: training.initialTrainPerplexity,
      finalTrainPerplexity: training.finalTrainPerplexity,
      initialValPerplexity: training.initialValPerplexity,
      finalValPerplexity: training.finalValPerplexity,
      checkpointPaths: training.checkpointPaths.slice(),
      bundleExportPaths: bundleExportPaths.slice(),
      history: training.history.slice(),
      samples,
      monitor,
    }
    const summaryPath = config.report?.summaryPath
      ?? (config.checkpoint?.prefix ? `${config.checkpoint.prefix}-summary.json` : null)
    if (summaryPath) {
      fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2))
    }

    return { tokenizer, corpus, model, compiledModel: compiledModel as DecoderModelModule | null, training, samples, monitor, summary }
  } finally {
    setDevice(previousDevice)
  }
}
