// Diagnostic only: synthetic graph inputs measure submission and retention.
// Use check-whisper.ts, check-audio.ts and audit.ts for model accuracy.
import fs from 'std:fs'
import { getEnv } from 'std:process'
import telemetry from 'std:telemetry'
import { tensor, reshape, type Tensor } from 'affon:compute'
import { load_graph, load_graph_for_scope_validation } from '../../../packages/@affon/onnx/src/runtime.ts'

const directory = getEnv('AFFON_ONNX_DIR')!
const ordinary = load_graph_for_scope_validation(directory, 'metal', false)
const scoped = load_graph_for_scope_validation(directory, 'metal')
const default_model = load_graph(directory, 'metal')
const inputs: Record<string, Tensor> = {}
for (const [name, shape] of Object.entries(ordinary.graph.inputs)) {
  inputs[name] = reshape(tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(0),
    { dtype: 'f32', device: 'metal' }), shape)
}
const metrics = () => Object.fromEntries(telemetry.metrics()
  .filter(m => m.scope === 'compute.execution' && m.name.startsWith('metal_command_') ||
    m.scope === 'compute.storage' && ['allocation_count', 'live_bytes', 'peak_bytes'].includes(m.name))
  .map(m => [`${m.scope}/${m.name}`, m.value]))

ordinary.forward(inputs)
scoped.forward(inputs)
default_model.forward(inputs)
const runs = []
const outputs: Record<string, Tensor>[] = []
for (const [name, model] of [['ordinary', ordinary], ['scoped', scoped], ['default', default_model]] as const) {
  const before = metrics(), start = Date.now()
  outputs.push(model.forward(inputs))
  const elapsed_ms = Date.now() - start, after = metrics()
  runs.push({ name, elapsed_ms,
    counters: Object.fromEntries(Object.entries(after).map(([key, value]) =>
      [key, key.endsWith('bytes') ? value : value - (before[key] ?? 0)])),
    scope: name === 'scoped' ? scoped.scope_stats() : null })
}
let max_absolute_error = 0
for (const candidate of outputs.slice(1)) {
  for (const key of Object.keys(outputs[0])) {
    const a = (outputs[0][key].to_array() as number[]).flat(Infinity) as number[]
    const b = (candidate[key].to_array() as number[]).flat(Infinity) as number[]
    if (a.length !== b.length) throw Error('Scope output length mismatch')
    for (let i = 0; i < a.length; i++) {
      const error = Math.abs(a[i] - b[i])
      if (!Number.isFinite(error) || error > 1e-5 + 1e-5 * Math.abs(a[i]))
        throw Error(`Scope output mismatch: ${key}[${i}]`)
      max_absolute_error = Math.max(max_absolute_error, error)
    }
  }
}
const report = { directory, synthetic_inputs: true, max_absolute_error, runs }
fs.writeFileSync(getEnv('PROFILE_OUTPUT')!, JSON.stringify(report, null, 2))
console.log(JSON.stringify(report))
