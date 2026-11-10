// Capture one warmed deterministic GPT-2 inference Program with runtime evidence.
import fs from 'std:fs'
import { getEnv } from 'std:process'
import telemetry from 'std:telemetry'
import { Session, Tensor } from 'affon:compute'
import { create_gpt2 } from '../../../packages/@affon/models/src/gpt2/model.ts'

const output = getEnv('AFFON_INFERENCE_PROFILE_PATH')
if (!output) throw Error('Set AFFON_INFERENCE_PROFILE_PATH')

const device = (getEnv('AFFON_PROFILE_DEVICE') || 'cuda') as 'cpu' | 'metal' | 'cuda'
const captureRecords = Number(getEnv('AFFON_PROFILE_CAPTURE_RECORDS') || '262144')
if (!Number.isInteger(captureRecords) || captureRecords < 1 || captureRecords > 4194304)
  throw Error('AFFON_PROFILE_CAPTURE_RECORDS must be an integer between 1 and 4194304')

const width = 64
const heads = 4
const layers = 2
const contextLength = 64
const vocabulary = 256
const sequence = 32
const outputStart = sequence - 1
const inner = 128

const session = new Session({
  device,
  telemetry: {
    computeTiming: 'invocation',
    bridgeTiming: true,
    profilerMarkers: true,
    captureRecords,
  },
})

let seed = 7
function values(shape: readonly number[]): number[] | number[][] | number[][][] | number[][][][] {
  const build = (dimensions: readonly number[]): any => {
    if (dimensions.length === 0) return (((seed++ * 13) % 29) - 14) / 29
    return Array.from({ length: dimensions[0] }, () => build(dimensions.slice(1)))
  }
  return build(shape)
}
function weight(shape: readonly number[]): Tensor {
  return session.tensor(values(shape), { dtype: 'f32' })
}
function affine(input: number, output: number) {
  return { weight: weight([input, output]), bias: weight([output]) }
}
function norm() {
  return { weight: weight([width]), bias: weight([width]) }
}

const model = create_gpt2({
  width,
  heads,
  layers,
  contextLength,
  vocabSize: vocabulary,
  epsilon: 1e-5,
}, {
  tokenEmbedding: weight([vocabulary, width]),
  positionEmbedding: weight([contextLength, width]),
  finalNorm: norm(),
  blocks: Array.from({ length: layers }, () => ({
    attentionNorm: norm(),
    feedForwardNorm: norm(),
    qkv: affine(width, 3 * width),
    attentionOutput: affine(width, width),
    expand: affine(width, inner),
    contract: affine(inner, width),
  })),
})

const source = model.forward(sequence, outputStart)
const state = session.initialize(source)
const executable = session.compile(source, { explanationLevel: 'detailed' })
const ids = session.tensor(Array.from({ length: sequence }, (_, index) => (index * 7 + 3) % vocabulary), { dtype: 'i64' })
const positions = session.tensor(Array.from({ length: sequence }, (_, index) => index), { dtype: 'i64' })
const mask = session.tensor([[[
  ...Array.from({ length: sequence }, (_, row) =>
    Array.from({ length: sequence }, (_, column) => column > row ? 1 : 0)),
]]], { dtype: 'i64' })

function selectedToken(bytes: ArrayBuffer): number {
  const row = new Float32Array(bytes)
  if (row.length !== vocabulary) throw Error('Unexpected GPT-2 logit shape')
  let selected = 0
  for (let index = 0; index < row.length; index++) {
    if (!Number.isFinite(row[index])) throw Error('GPT-2 produced a non-finite logit')
    if (row[index] > row[selected]) selected = index
  }
  return selected
}

function runSelected() {
  const outputs = executable.run({ ids, positions, mask }, state) as Tensor[]
  try {
    if (outputs[0]?.shape.join(',') !== `1,1,${vocabulary}`)
      throw Error('GPT-2 produced an invalid logits shape')
    return selectedToken(outputs[0].to_bytes())
  } finally {
    for (const value of outputs) value.dispose()
  }
}

try {
  const warmupToken = runSelected()
  session.resetEvidence()
  const start = performance.now()
  const selectedTokenId = telemetry.trace(
    'gpt2.prefix_inference.selected',
    runSelected,
  )
  const elapsedMs = performance.now() - start
  if (selectedTokenId !== warmupToken)
    throw Error('GPT-2 selected token changed between warmup and selected run')
  const evidence = session.runtimeEvidence(executable) as any
  const report = {
    schema_version: 1,
    suite: 'gpt2_program_inference_profile',
    device,
    protocol: {
      model: 'deterministic_tiny_gpt2_program',
      width,
      heads,
      layers,
      context_length: contextLength,
      vocabulary,
      sequence,
      output_start: outputStart,
      warmup_steps: 1,
      measured_steps: 1,
      dtype: 'f32',
    },
    result: {
      selected_token_id: selectedTokenId,
      elapsed_ms: elapsedMs,
    },
    observation: {
      capture_records: captureRecords,
      compute_timing: 'invocation',
      bridge_timing: true,
      profiler_markers: true,
      dropped_event_count: evidence.dropped_event_count,
    },
    runtime_evidence: evidence,
    executable_explanation: executable.explain(),
    telemetry: { metrics: telemetry.metrics() },
    limitations: [
      'This is one warmed fixed-shape deterministic inference Program for attribution, not a throughput measurement.',
      'The selected trace includes logit readback and native token selection observation, but not model loading or compilation.',
      'The tiny GPT-2 model uses deterministic synthetic weights and is not a pretrained-model quality result.',
      'Program run, bridge, and readback spans include observation overhead.',
    ],
  }
  fs.writeFileSync(output, JSON.stringify(report, null, 2))
  console.log(JSON.stringify({
    suite: report.suite,
    output,
    selected_token_id: selectedTokenId,
    elapsed_ms: elapsedMs,
    dropped_event_count: evidence.dropped_event_count,
  }))
} finally {
  mask.dispose()
  positions.dispose()
  ids.dispose()
  executable.dispose()
  state.dispose()
  session.dispose()
}
