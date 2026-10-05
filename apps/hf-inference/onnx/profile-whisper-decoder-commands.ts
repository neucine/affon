// Synthetic decoder inputs isolate per-node command counts, not model accuracy.
import fs from 'std:fs'
import { getEnv } from 'std:process'
import telemetry from 'std:telemetry'
import { Session, type Tensor } from 'affon:compute'
import { load_graph } from '../../../packages/@affon/onnx/src/index.ts'
const root = getEnv('AFFON_WHISPER_DIR') ?? 'apps/hf-inference/artifacts/whisper'
const model = load_graph(`${root}/step`)
const session = new Session({ device: 'metal' })
const state = session.initialize(model.forward, { parameters: model.parameters })
const executable = session.compile(model.forward)
const zeros = (shape: number[]): unknown => shape.length === 1 ? Array(shape[0]).fill(0) : Array.from({ length: shape[0] }, () => zeros(shape.slice(1)))
const inputs: Record<string, Tensor> = {}
for (const [name, shape] of Object.entries(model.graph.inputs)) {
  inputs[name] = session.tensor(zeros(shape) as any, { dtype: 'f32' })
}
function metrics() {
  return Object.fromEntries(
    telemetry
      .metrics()
      .filter(
        (m) =>
          (m.scope === 'compute.execution' &&
            m.name.startsWith('metal_command_')) ||
          (m.scope === 'compute.storage' && m.name === 'allocation_count'),
      )
      .map((m) => [`${m.scope}/${m.name}`, m.value]),
  )
}
const dispose = (value: Tensor | Tensor[]) => (Array.isArray(value) ? value : [value]).forEach(output => output.dispose())
dispose(executable.run(inputs, state) as Tensor | Tensor[])
let before = metrics()
const start = Date.now()
dispose(executable.run(inputs, state) as Tensor | Tensor[])
const after = metrics()
const counters = Object.fromEntries(Object.entries(after).map(([key, value]) => [key, value - (before[key] ?? 0)]))
const operators = model.forward.inspect().nodes.map(node => ({ id: node.id, op: node.op, shape: node.spec.shape }))
fs.writeFileSync(
  getEnv('PROFILE_OUTPUT')!,
  JSON.stringify({ synthetic_inputs: true, elapsed_ms: Date.now() - start, counters, operators }, null, 2),
)
for (const input of Object.values(inputs)) input.dispose()
state.dispose(); session.dispose()
