// Capture one warmed cached-prefill or cached-decode step with runtime evidence
// and NVTX ranges.
import fs from 'std:fs'
import { getEnv } from 'std:process'
import telemetry from 'std:telemetry'
import { load_apertus } from '../../../packages/@affon/huggingface/src/adapters/apertus.ts'
import { CausalProgramRuntime } from '../../../apps/hf-inference/src/inference/program-runtime.ts'

const checkpoint = getEnv('AFFON_APERTUS_CHECKPOINT')
const output = getEnv('AFFON_APERTUS_PROFILE_PATH')
if (!checkpoint || !output) throw Error('Set AFFON_APERTUS_CHECKPOINT and AFFON_APERTUS_PROFILE_PATH')

const device = (getEnv('AFFON_PROFILE_DEVICE') || 'cuda') as 'cpu' | 'metal' | 'cuda'
const workload = getEnv('AFFON_PROFILE_WORKLOAD') || 'decode'
if (workload !== 'prefill' && workload !== 'decode')
  throw Error('AFFON_PROFILE_WORKLOAD must be prefill or decode')
const captureRecords = Number(getEnv('AFFON_PROFILE_CAPTURE_RECORDS') || '262144')
if (!Number.isInteger(captureRecords) || captureRecords < 1 || captureRecords > 4194304)
  throw Error('AFFON_PROFILE_CAPTURE_RECORDS must be an integer between 1 and 4194304')

const prompt = (getEnv('AFFON_PROFILE_PROMPT') || '1,1784,8961,1307,32265,1395')
  .split(',')
  .map(value => Number(value))
if (prompt.length === 0 || prompt.some(value => !Number.isInteger(value) || value < 0))
  throw Error('AFFON_PROFILE_PROMPT must be a comma-separated list of nonnegative token IDs')
const capacity = Number(getEnv('AFFON_PROFILE_CAPACITY') || '320')
const budget = Number(getEnv('AFFON_PROFILE_BUDGET') || '4')
if (!Number.isInteger(capacity) || capacity < 2)
  throw Error('AFFON_PROFILE_CAPACITY must be an integer of at least 2')
if (!Number.isInteger(budget) || budget < (workload === 'decode' ? 3 : 1))
  throw Error(workload === 'decode'
    ? 'AFFON_PROFILE_BUDGET must include prefill, one warmup, and one selected token'
    : 'AFFON_PROFILE_BUDGET must include one selected prefill token')
if (prompt.length + budget > capacity)
  throw Error('Prompt and generation budget must fit the fixed cache capacity')
const model = load_apertus(checkpoint)
const runtime = new CausalProgramRuntime(model, device, {
  telemetry: {
    computeTiming: "invocation",
    bridgeTiming: true,
    profilerMarkers: true,
    captureRecords,
  },
})

try {
  runtime.prepareFixedCache(capacity, [prompt.length])
  const executable = workload === 'decode'
    ? runtime.session.compile(model.decode(capacity))
    : runtime.session.compile(model.prefill(prompt.length, capacity))
  const warmGeneration = runtime.createGeneration(prompt, workload === 'decode' ? 2 : 1)
  try {
    warmGeneration.next()
    if (workload === 'decode') warmGeneration.next()
  } finally {
    warmGeneration.close()
  }
  const generation = runtime.createGeneration(prompt, workload === 'decode' ? budget : 1)
  try {
    runtime.session.resetEvidence()
    const start = performance.now()
    const selected = telemetry.trace(`apertus.cached_${workload}.selected`, () => generation.next())
    const elapsedMs = performance.now() - start
    const evidence = runtime.session.runtimeEvidence(executable) as any
    const report = {
      schema_version: 1,
      suite: 'apertus_affon_profile',
      device,
      checkpoint,
      protocol: {
        prompt,
        capacity,
        budget,
        workload,
        measured_steps: 1,
        warmup_steps: workload === 'decode' ? 2 : 1,
      },
      result: {
        ids: selected.ids,
        done: selected.done,
        elapsed_ms: elapsedMs,
      },
      observation: {
        capture_records: captureRecords,
        compute_timing: "invocation",
        kernel_count_evidence: true,
        bridge_timing: true,
        profiler_markers: true,
        dropped_event_count: evidence.dropped_event_count,
      },
      runtime_evidence: evidence,
      executable_explanation: executable.explain(),
      telemetry: { metrics: telemetry.metrics() },
      limitations: [
        `This is one selected warmed ${workload} step for attribution, not a throughput measurement.`,
        'Program run, bridge, and readback spans include observation overhead.',
        'Use the Nsight timeline and Compute aggregate to distinguish device work, gaps, host submission, and readback.',
        'Check dropped_event_count and the telemetry collector missed flag before treating this artifact as complete.',
      ],
    }
    fs.writeFileSync(output, JSON.stringify(report, null, 2))
    console.log(JSON.stringify({
      suite: 'apertus_affon_profile',
      output,
      workload,
      ids: selected.ids,
      elapsed_ms: elapsedMs,
      dropped_event_count: evidence.dropped_event_count,
    }))
  } finally {
    generation.close()
  }
} finally {
  runtime.dispose()
}
