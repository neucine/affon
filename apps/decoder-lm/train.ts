import { getEnv } from 'std:process'
import telemetry from 'std:telemetry'

import { loadDecoderLMWorkflowConfig, trainDecoderLMFromConfig } from './src/index.ts'

const configPath = getEnv('AFFON_TRAIN_CONFIG')
if (!configPath) {
  throw new AffonError('invalid_arg', 'Set AFFON_TRAIN_CONFIG to a workflow JSON config path')
}

console.log(`config: ${configPath}`)
console.log('loading workflow config...')
const config = loadDecoderLMWorkflowConfig(configPath)
const progress = config.report?.progress ?? {}
const trainPhase = progress.trainPhase ?? false
const trainLossEveryBatches = progress.trainLossEveryBatches ?? 0
const runtimeStatsEveryBatches = progress.runtimeStatsEveryBatches ?? 0
const evalEveryBatches = progress.evalEveryBatches ?? 10
const epochSummary = progress.epochSummary ?? true

type RuntimeMetric = ReturnType<typeof telemetry.metrics>[number]

let previousRuntimeMetrics: RuntimeMetric[] | null = null

function metricValue(metrics: readonly RuntimeMetric[], group: string, name: string): number {
  const entry = metrics.find((metric) => metric.scope === group && metric.name === name)
  return entry?.value ?? 0
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}mb`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)}kb`
  return `${bytes}b`
}

function formatTopPoolBuckets(
  metrics: readonly RuntimeMetric[],
  bucketPrefix: string,
  countPrefix: string,
): string {
  const parts: string[] = []
  for (let i = 1; i <= 8; i += 1) {
    const bucketBytes = metricValue(metrics, 'compute.memory', `${bucketPrefix}_${i}`)
    const count = metricValue(metrics, 'compute.memory', `${countPrefix}_${i}`)
    if (bucketBytes <= 0 || count <= 0) continue
    parts.push(`${formatBytes(bucketBytes)}:${count}`)
  }
  return parts.length > 0 ? parts.join(',') : 'none'
}

function formatRuntimeStats(): string {
  const stats = telemetry.metrics()
  const toMb = (bytes: number) => (bytes / (1024 * 1024)).toFixed(1)
  const deltaPoolHits = metricValue(stats, 'compute.memory', 'metal_pool_hit_count')
    - metricValue(previousRuntimeMetrics ?? [], 'compute.memory', 'metal_pool_hit_count')
  const deltaPoolMisses = metricValue(stats, 'compute.memory', 'metal_pool_miss_count')
    - metricValue(previousRuntimeMetrics ?? [], 'compute.memory', 'metal_pool_miss_count')
  const deltaAllocations = metricValue(stats, 'compute.storage', 'allocation_count')
    - metricValue(previousRuntimeMetrics ?? [], 'compute.storage', 'allocation_count')
  const deltaReuses = metricValue(stats, 'compute.storage', 'reuse_count')
    - metricValue(previousRuntimeMetrics ?? [], 'compute.storage', 'reuse_count')
  const failureCount = 0
  const failureText = failureCount > 0
    ? ` metal_alloc_failures=${failureCount}`
    : ''
  const topMisses = formatTopPoolBuckets(stats, 'metal_pool_top_miss_bucket_bytes', 'metal_pool_top_miss_count')
  const topDrops = formatTopPoolBuckets(stats, 'metal_pool_top_drop_bucket_bytes', 'metal_pool_top_drop_count')
  previousRuntimeMetrics = stats
  return `metal_live_mb=${toMb(metricValue(stats, 'compute.storage', 'live_metal_bytes'))} metal_device_mb=${toMb(metricValue(stats, 'compute.memory', 'metal_device_current_allocated_bytes'))} metal_peak_mb=${toMb(metricValue(stats, 'compute.storage', 'peak_metal_bytes'))} metal_pool_mb=${toMb(metricValue(stats, 'compute.memory', 'metal_pool_live_bytes'))} metal_pool_peak_mb=${toMb(metricValue(stats, 'compute.memory', 'metal_pool_peak_bytes'))} cpu_live_mb=${toMb(metricValue(stats, 'compute.storage', 'live_cpu_bytes'))} cpu_peak_mb=${toMb(metricValue(stats, 'compute.storage', 'peak_cpu_bytes'))} autograd_mb=0.0 qjs_used_mb=${toMb(metricValue(stats, 'runtime.memory', 'qjs_heap_used_bytes'))} resident_mb=${toMb(metricValue(stats, 'runtime.memory', 'resident_bytes'))} resident_peak_mb=${toMb(metricValue(stats, 'runtime.memory', 'resident_peak_bytes'))} phys_mb=${toMb(metricValue(stats, 'runtime.memory', 'physical_footprint_bytes'))} phys_peak_mb=${toMb(metricValue(stats, 'runtime.memory', 'physical_footprint_peak_bytes'))} ioaccel_mb=0.0 storage_allocations_since_prev=${deltaAllocations} storage_reuses_since_prev=${deltaReuses} pool_hits_since_prev=${deltaPoolHits} pool_misses_since_prev=${deltaPoolMisses} pool_top_misses=${topMisses} pool_top_drops=${topDrops}${failureText}`
}
console.log('starting training workflow...')
const result = trainDecoderLMFromConfig(config, {
  onStatus: (status) => {
    if (status.startsWith('evaluating') && evalEveryBatches > 0) {
      const match = status.match(/^(.*?)(?: (\d+)\/(\d+)\.\.\.)?$/)
      if (match && match[2] && match[3]) {
        const batch = Number(match[2])
        const batches = Number(match[3])
        if (batch !== 1 && batch !== batches && batch % evalEveryBatches !== 0) return
      }
    }
    console.log(status)
  },
  onBatchPhase: (metrics) => {
    if (!trainPhase) return
    console.log(
      `epoch ${metrics.epoch} batch ${metrics.batch}/${metrics.batches}: ${metrics.phase}...`,
    )
  },
  onBatch: (metrics) => {
    if (
      runtimeStatsEveryBatches > 0
      && (
        metrics.batch === 1
        || metrics.batch === metrics.batches
        || metrics.batch % runtimeStatsEveryBatches === 0
      )
    ) {
      console.log(
        `epoch ${metrics.epoch} batch ${metrics.batch}/${metrics.batches}: ${formatRuntimeStats()}`,
      )
    }
    if (trainLossEveryBatches <= 0) return
    if (
      metrics.batch !== 1
      && metrics.batch !== metrics.batches
      && metrics.batch % trainLossEveryBatches !== 0
    ) {
      return
    }
    const gradNorm = metrics.gradNorm === null ? 'null' : metrics.gradNorm.toFixed(6)
    console.log(
      `epoch ${metrics.epoch} batch ${metrics.batch}/${metrics.batches}: step=${metrics.step} loss=${metrics.batchLoss.toFixed(6)} lr=${metrics.lr.toExponential(6)} grad_norm=${gradNorm}`,
    )
  },
  onEpoch: (metrics) => {
    if (!epochSummary) return
    const trainLoss = metrics.trainLoss.toFixed(6)
    const trainPpl = metrics.trainPerplexity.toFixed(6)
    const valLoss = metrics.valLoss === null ? 'null' : metrics.valLoss.toFixed(6)
    const valPpl = metrics.valPerplexity === null ? 'null' : metrics.valPerplexity.toFixed(6)
    console.log(
      `epoch ${metrics.epoch}: train_loss=${trainLoss} train_ppl=${trainPpl} val_loss=${valLoss} val_ppl=${valPpl}`,
    )
  },
  onSample: (sample) => {
    console.log(`epoch ${sample.epoch}: sample=${sample.generatedText}`)
  },
})

console.log('training complete')
console.log('tokenizer.family:', result.tokenizer.family)
console.log('trainRows:', result.corpus.trainRows.length)
console.log('validationRows:', result.corpus.validationRows.length)
console.log('trainWindows:', result.corpus.trainWindows.length)
console.log('validationWindows:', result.corpus.validationWindows.length)
console.log('modelForwardMode:', result.summary.modelForwardMode)
console.log('steps:', result.training.steps)
console.log('initialTrainLoss:', result.training.initialTrainLoss === null ? 'null' : result.training.initialTrainLoss.toFixed(6))
console.log('finalTrainLoss:', result.training.finalTrainLoss.toFixed(6))
console.log('initialTrainPerplexity:', result.training.initialTrainPerplexity === null ? 'null' : result.training.initialTrainPerplexity.toFixed(6))
console.log('finalTrainPerplexity:', result.training.finalTrainPerplexity.toFixed(6))
console.log('initialValLoss:', result.training.initialValLoss === null ? 'null' : result.training.initialValLoss.toFixed(6))
console.log('finalValLoss:', result.training.finalValLoss === null ? 'null' : result.training.finalValLoss.toFixed(6))
console.log('initialValPerplexity:', result.training.initialValPerplexity === null ? 'null' : result.training.initialValPerplexity.toFixed(6))
console.log('finalValPerplexity:', result.training.finalValPerplexity === null ? 'null' : result.training.finalValPerplexity.toFixed(6))
console.log('checkpoints:', JSON.stringify(result.training.checkpointPaths))
console.log('monitor:', result.monitor === null ? 'null' : JSON.stringify({
  path: result.monitor.path,
  snapshots: result.monitor.snapshots.length,
  droppedSamples: result.monitor.droppedSamples,
}))
console.log('samples:', JSON.stringify(result.samples))
