// Synthetic decoder inputs isolate per-node command counts, not model accuracy.
import fs from 'std:fs'
import { getEnv } from 'std:process'
import telemetry from 'std:telemetry'
import { Session, type Tensor } from 'affon:compute'
import { load_graph } from '../../../packages/@affon/onnx/src/index.ts'
const root = getEnv('AFFON_WHISPER_DIR') ?? '/private/tmp/affon-onnx-whisper'
const model = load_graph(`${root}/step`, 'metal')
const session = new Session({ device: 'metal' })
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
model.forward(inputs)
let before = metrics()
const nodes: unknown[] = []
model.forward(inputs, (event) => {
  const after = metrics()
  nodes.push({
    ...event,
    counters: Object.fromEntries(
      Object.entries(after).map(([k, v]) => [k, v - (before[k] ?? 0)]),
    ),
  })
  before = after
})
fs.writeFileSync(
  getEnv('PROFILE_OUTPUT')!,
  JSON.stringify({ synthetic_inputs: true, nodes }, null, 2),
)
model.dispose(); session.dispose()
