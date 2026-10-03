import fs from 'std:fs'
import { getEnv } from 'std:process'
import checkpoint from 'affon:checkpoint'
import { Session, type Tensor, type Device } from 'affon:compute'
import { load_graph } from '../../../packages/@affon/onnx/src/index.ts'
const root = getEnv('AFFON_WHISPER_DIR') ?? '/tmp/affon-onnx-whisper'
const model = load_graph(`${root}/encoder`)
const refs = checkpoint.load(`${root}/reference.safetensors`) as Record<
  string,
  Tensor
>
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const session = new Session({ device })
const state = session.initialize(model.forward, { parameters: model.parameters })
const executable = session.compile(model.forward)
const features = session.tensor(refs.features.to_array() as any, { dtype: refs.features.dtype })
const dispose = (value: Tensor | Tensor[]) => (Array.isArray(value) ? value : [value]).forEach(output => output.dispose())
dispose(executable.run({ features }, state) as Tensor | Tensor[])
const start = Date.now()
dispose(executable.run({ features }, state) as Tensor | Tensor[])
const elapsed_ms = Date.now() - start
const nodes = model.forward.inspect().nodes.map(node => ({ id: node.id, op: node.op ?? 'unknown', shape: node.spec.shape }))
const groups: Record<string, number> = {}
for (const node of nodes) groups[node.op] = (groups[node.op] ?? 0) + 1
const report = {
  elapsed_ms,
  operator_counts: groups,
  nodes,
}
const output = getEnv('PROFILE_OUTPUT')
if (output) fs.writeFileSync(output, JSON.stringify(report, null, 2))
console.log(
  JSON.stringify({ elapsed_ms, operator_counts: groups, operators: nodes.length }),
)
features.dispose(); state.dispose(); session.dispose()
