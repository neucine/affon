// Capture one warmed Program training step with composite runtime evidence.
import fs from 'std:fs'
import { getEnv } from 'std:process'
import telemetry from 'std:telemetry'
import { Session, Tensor, optimize, program } from 'affon:compute'
import { cross_entropy } from 'affon:nn'
import { adamw } from 'affon:optim'
import { DecoderModel } from '../../../apps/decoder-lm/src/model.ts'

const output = getEnv('AFFON_TRAINING_PROFILE_PATH')
if (!output) throw Error('Set AFFON_TRAINING_PROFILE_PATH')

const device = (getEnv('AFFON_PROFILE_DEVICE') || 'cpu') as 'cpu' | 'metal' | 'cuda'
const captureRecords = Number(getEnv('AFFON_PROFILE_CAPTURE_RECORDS') || '262144')
if (!Number.isInteger(captureRecords) || captureRecords < 1 || captureRecords > 4194304)
  throw Error('AFFON_PROFILE_CAPTURE_RECORDS must be an integer between 1 and 4194304')

const vocabulary = 32
const batch = 2
const sequence = 8
const model = DecoderModel(vocabulary, 16, {
  numLayers: 1,
  numHeads: 2,
  hiddenDim: 32,
  causal: true,
  positional: 'learned',
  maxSeqLen: sequence,
  tieEmbeddings: true,
})
const source = program('training_decoder', p => model({
  token_ids: p.argument(
    'token_ids',
    Tensor.i64([batch, sequence], { axes: ['batch', 'token'] }),
  ),
}, 'decoder'))
const training = optimize(
  source,
  cross_entropy(),
  adamw({ learning_rate: 0.001, weight_decay: 0.01 }),
)

const session = new Session({
  device,
  telemetry: {
    computeTiming: 'invocation',
    bridgeTiming: true,
    profilerMarkers: true,
    captureRecords,
  },
})
const state = session.initialize(training, { seed: 7 })
const ids = session.tensor([
  [1, 4, 7, 10, 13, 16, 19, 22],
  [2, 5, 8, 11, 14, 17, 20, 23],
], { dtype: 'i64', axes: ['batch', 'token'] })
const labels = session.tensor([
  [4, 7, 10, 13, 16, 19, 22, 25],
  [5, 8, 11, 14, 17, 20, 23, 26],
], { dtype: 'i64', axes: ['batch', 'token'] })
const executable = session.compile(training)

try {
  const warmup = executable.run({ token_ids: ids, labels }, state)
  try {
    if (!Number.isFinite(warmup.item())) throw Error('warmup training loss is not finite')
  } finally {
    warmup.dispose()
  }

  session.resetEvidence()
  const start = performance.now()
  const selected = telemetry.trace(
    'training.step.selected',
    () => executable.run({ token_ids: ids, labels }, state),
  )
  const elapsedMs = performance.now() - start
  try {
    const loss = selected.item()
    if (!Number.isFinite(loss)) throw Error('selected training loss is not finite')
    const evidence = session.runtimeEvidence(executable)
    const report = {
      schema_version: 1,
      suite: 'training_affon_profile',
      device,
      protocol: {
        vocabulary,
        batch,
        sequence,
        warmup_steps: 1,
        measured_steps: 1,
        optimizer: 'adamw',
        learning_rate: 0.001,
        weight_decay: 0.01,
      },
      result: { loss, elapsed_ms: elapsedMs },
      observation: {
        capture_records: captureRecords,
        compute_timing: 'invocation',
        kernel_count_evidence: true,
        bridge_timing: true,
        profiler_markers: true,
      },
      runtime_evidence: evidence,
      executable_explanation: executable.explain(),
      telemetry: { metrics: telemetry.metrics() },
      limitations: [
        'This is one fixed-shape warmed training step for attribution, not a throughput measurement.',
        'The gradient executable owns both primal forward and backward timing when it returns both values.',
        'The optimizer component is the actual update executable selected for this step.',
        'Profiler spans and bridge timing include observation overhead.',
      ],
    }
    fs.writeFileSync(output, JSON.stringify(report, null, 2))
    console.log(JSON.stringify({
      suite: report.suite,
      output,
      loss,
      elapsed_ms: elapsedMs,
    }))
  } finally {
    selected.dispose()
  }
} finally {
  executable.dispose()
  labels.dispose()
  ids.dispose()
  state.dispose()
  session.dispose()
}
