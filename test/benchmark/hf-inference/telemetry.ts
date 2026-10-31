import telemetry, {type MetricSnapshot} from 'std:telemetry'

export interface PhaseProfile {
  name: string
  status: 'ok' | 'err'
  elapsed_ms: number
  counters: Record<string, number>
  gauges: Record<string, {before: number | null; after: number}>
  metal_gpu_timing_complete: boolean | null
}

const scopes = new Set(['compute.execution', 'compute.storage', 'compute.memory', 'runtime.memory'])
const snapshot = () => telemetry.metrics().filter(metric => scopes.has(metric.scope))

/** Counters are interval deltas; gauges (including process-lifetime peaks) are
 * snapshots, never mislabeled as phase-local peaks. New counters start at zero. */
export function metric_changes(before: MetricSnapshot[], after: MetricSnapshot[]) {
  const previous = new Map(before.map(metric => [`${metric.scope}.${metric.name}`, metric]))
  const counters: PhaseProfile['counters'] = {}, gauges: PhaseProfile['gauges'] = {}
  for (const metric of after) {
    const key = `${metric.scope}.${metric.name}`, old = previous.get(key)
    if (metric.kind === 'counter') counters[key] = metric.value - (old?.value ?? 0)
    else if (metric.kind === 'gauge') gauges[key] = {before: old?.value ?? null, after: metric.value}
  }
  const count = counters['compute.execution.metal_command_count']
  const valid = counters['compute.execution.metal_command_gpu_valid_count']
  return {counters, gauges, metal_gpu_timing_complete: count === undefined || count === 0 ? null : valid === count}
}

/** Uses the existing telemetry backend; callers must include device readback or
 * another appropriate completion boundary in work when measuring GPU workloads. */
export function create_phase_profiler() {
  const phases: PhaseProfile[] = []
  function begin(name: string) {
    const before = snapshot(), start = Date.now()
    return (status: 'ok' | 'err') => {
      const elapsed_ms = Date.now() - start
      phases.push({name, status, elapsed_ms, ...metric_changes(before, snapshot())})
    }
  }
  return {
    phases,
    sync<T>(name: string, work: () => T): T {
      const end = begin(name)
      let status: 'ok' | 'err' = 'err'
      try { const value = telemetry.trace(name, work); status = 'ok'; return value }
      finally { end(status) }
    },
    async async<T>(name: string, work: () => Promise<T>): Promise<T> {
      const end = begin(name)
      let status: 'ok' | 'err' = 'err'
      try { const value = await telemetry.trace(name, work); status = 'ok'; return value }
      finally { end(status) }
    },
  }
}
