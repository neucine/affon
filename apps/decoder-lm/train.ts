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

const trainDevice = getEnv('AFFON_TRAIN_DEVICE')
if (trainDevice) {
  if (trainDevice !== 'cpu' && trainDevice !== 'metal' && trainDevice !== 'cuda') {
    throw new AffonError('invalid_arg', "AFFON_TRAIN_DEVICE must be 'cpu', 'metal', or 'cuda'")
  }
  config.device = trainDevice
  console.log(`device override: ${trainDevice}`)
}

const trainEpochs = getEnv('AFFON_TRAIN_EPOCHS')
if (trainEpochs) {
  const epochs = Number(trainEpochs)
  if (!Number.isInteger(epochs) || epochs <= 0) {
    throw new AffonError('invalid_arg', 'AFFON_TRAIN_EPOCHS must be a positive integer')
  }
  config.training.epochs = epochs
  console.log(`epoch override: ${epochs}`)
}

const checkpointPrefix = getEnv('AFFON_TRAIN_CHECKPOINT_PREFIX')
if (checkpointPrefix) {
  config.checkpoint = { ...(config.checkpoint ?? {}), prefix: checkpointPrefix }
  console.log(`checkpoint override: ${checkpointPrefix}`)
}

const summaryPath = getEnv('AFFON_TRAIN_SUMMARY_PATH')
if (summaryPath) config.report = { ...(config.report ?? {}), summaryPath }

const monitorPath = getEnv('AFFON_TRAIN_MONITOR_PATH')
if (monitorPath) {
  config.report = {
    ...(config.report ?? {}),
    monitor: { ...(config.report?.monitor ?? { everyBatches: 25 }), path: monitorPath },
  }
}

const progress = config.report?.progress ?? {}
const trainPhase = progress.trainPhase ?? false
const trainLossEveryBatches = progress.trainLossEveryBatches ?? 0
const runtimeStatsEveryBatches = progress.runtimeStatsEveryBatches ?? 0
const evalEveryBatches = progress.evalEveryBatches ?? 10
const epochSummary = progress.epochSummary ?? true

type RuntimeMetric = ReturnType<typeof telemetry.metrics>[number]

let previousRuntimeMetrics: RuntimeMetric[] | null = null

function metricValue(metrics: readonly RuntimeMetric[], scope: string, name: string): number {
  return metrics.find((metric) => metric.scope === scope && metric.name === name)?.value ?? 0
}

function formatRuntimeStats(): string {
  const stats = telemetry.metrics()
  const toMb = (bytes: number) => (bytes / (1024 * 1024)).toFixed(1)
  const activeDevice = config.device === 'cuda' ? 'cuda' : config.device === 'metal' ? 'metal' : 'cpu'
  const liveDeviceBytes = metricValue(stats, 'compute.storage', `live_${activeDevice}_bytes`)
  const peakDeviceBytes = metricValue(stats, 'compute.storage', `peak_${activeDevice}_bytes`)
  const deviceAllocatedBytes = activeDevice === 'cuda'
    ? metricValue(stats, 'compute.memory', 'cuda_device_current_allocated_bytes')
    : activeDevice === 'metal'
      ? metricValue(stats, 'compute.memory', 'metal_device_current_allocated_bytes')
      : 0
  const deltaPoolHits = metricValue(stats, 'compute.memory', 'device_pool_hit_count')
    - metricValue(previousRuntimeMetrics ?? [], 'compute.memory', 'device_pool_hit_count')
  const deltaPoolMisses = metricValue(stats, 'compute.memory', 'device_pool_miss_count')
    - metricValue(previousRuntimeMetrics ?? [], 'compute.memory', 'device_pool_miss_count')
  const deltaAllocations = metricValue(stats, 'compute.storage', 'allocation_count')
    - metricValue(previousRuntimeMetrics ?? [], 'compute.storage', 'allocation_count')
  const deltaReuses = metricValue(stats, 'compute.storage', 'reuse_count')
    - metricValue(previousRuntimeMetrics ?? [], 'compute.storage', 'reuse_count')
  previousRuntimeMetrics = stats
  return `${activeDevice}_live_mb=${toMb(liveDeviceBytes)} ${activeDevice}_device_mb=${toMb(deviceAllocatedBytes)} ${activeDevice}_peak_mb=${toMb(peakDeviceBytes)} device_pool_mb=${toMb(metricValue(stats, 'compute.memory', 'device_pool_live_bytes'))} device_pool_peak_mb=${toMb(metricValue(stats, 'compute.memory', 'device_pool_peak_bytes'))} cpu_live_mb=${toMb(metricValue(stats, 'compute.storage', 'live_cpu_bytes'))} cpu_peak_mb=${toMb(metricValue(stats, 'compute.storage', 'peak_cpu_bytes'))} qjs_used_mb=${toMb(metricValue(stats, 'runtime.memory', 'qjs_heap_used_bytes'))} resident_mb=${toMb(metricValue(stats, 'runtime.memory', 'resident_bytes'))} resident_peak_mb=${toMb(metricValue(stats, 'runtime.memory', 'resident_peak_bytes'))} phys_mb=${toMb(metricValue(stats, 'runtime.memory', 'physical_footprint_bytes'))} phys_peak_mb=${toMb(metricValue(stats, 'runtime.memory', 'physical_footprint_peak_bytes'))} storage_allocations_since_prev=${deltaAllocations} storage_reuses_since_prev=${deltaReuses} pool_hits_since_prev=${deltaPoolHits} pool_misses_since_prev=${deltaPoolMisses}`
}

console.log('starting training workflow...')
const result = trainDecoderLMFromConfig(config, {
  onStatus: (status) => {
    if (status.startsWith('evaluating') && evalEveryBatches > 0) {
      const match = status.match(/^(.*?)(?: (\d+)\/(\d+)\.\.\.)?$/)
      if (match?.[2] && match[3]) {
        const batch = Number(match[2])
        const batches = Number(match[3])
        if (batch !== 1 && batch !== batches && batch % evalEveryBatches !== 0) return
      }
    }
    console.log(status)
  },
  onBatchPhase: (metrics) => {
    if (trainPhase) console.log(`epoch ${metrics.epoch} batch ${metrics.batch}/${metrics.batches}: ${metrics.phase}...`)
  },
  onBatch: (metrics) => {
    if (
      runtimeStatsEveryBatches > 0
      && (metrics.batch === 1 || metrics.batch === metrics.batches || metrics.batch % runtimeStatsEveryBatches === 0)
    ) {
      console.log(`epoch ${metrics.epoch} batch ${metrics.batch}/${metrics.batches}: ${formatRuntimeStats()}`)
    }
    if (
      trainLossEveryBatches > 0
      && (metrics.batch === 1 || metrics.batch === metrics.batches || metrics.batch % trainLossEveryBatches === 0)
    ) {
      console.log(`epoch ${metrics.epoch} batch ${metrics.batch}/${metrics.batches}: step=${metrics.step} loss=${metrics.batchLoss.toFixed(6)} lr=${metrics.lr.toExponential(6)} optimizer_stepped=${metrics.optimizerStepped}`)
    }
  },
  onEpoch: (metrics) => {
    if (!epochSummary) return
    const valLoss = metrics.valLoss === null ? 'null' : metrics.valLoss.toFixed(6)
    const valPpl = metrics.valPerplexity === null ? 'null' : metrics.valPerplexity.toFixed(6)
    console.log(`epoch ${metrics.epoch}: train_loss=${metrics.trainLoss.toFixed(6)} train_ppl=${metrics.trainPerplexity.toFixed(6)} val_loss=${valLoss} val_ppl=${valPpl}`)
  },
  onSample: (sample) => console.log(`epoch ${sample.epoch}: sample=${sample.generatedText}`),
})

try {
  console.log('training complete')
  console.log('program:', `${result.summary.program.name} (${result.summary.program.parameters} parameters)`)
  console.log('steps:', result.training.steps)
  console.log('initialTrainLoss:', result.training.initialTrainLoss === null ? 'null' : result.training.initialTrainLoss.toFixed(6))
  console.log('finalTrainLoss:', result.training.finalTrainLoss.toFixed(6))
  console.log('initialValLoss:', result.training.initialValLoss === null ? 'null' : result.training.initialValLoss.toFixed(6))
  console.log('finalValLoss:', result.training.finalValLoss === null ? 'null' : result.training.finalValLoss.toFixed(6))
  console.log('checkpoints:', JSON.stringify(result.training.checkpointPaths))
  console.log('samples:', JSON.stringify(result.samples))
} finally {
  result.state.dispose()
  result.session.dispose()
}
