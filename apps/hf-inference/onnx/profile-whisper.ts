import fs from 'std:fs'
import { getEnv } from 'std:process'
import checkpoint from 'affon:checkpoint'
import { Session, type Tensor, type Device } from 'affon:compute'
import { load_graph } from '../../../packages/@affon/onnx/src/index.ts'
const root = getEnv('AFFON_WHISPER_DIR') ?? '/tmp/affon-onnx-whisper'
const model = load_graph(
  `${root}/encoder`,
  (getEnv('AFFON_DEVICE') ?? 'cpu') as Device,
)
const refs = checkpoint.load(`${root}/reference.safetensors`) as Record<
  string,
  Tensor
>
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const session = new Session({ device })
const features = session.tensor(refs.features.to_array() as any, { dtype: refs.features.dtype })
model.forward({ features })
const nodes: {
  name: string
  op: string
  shape: number[]
  elapsed_ms: number
}[] = []
const start = Date.now()
model.forward({ features }, (e) => nodes.push(e))
const elapsed_ms = Date.now() - start
const groups: Record<string, number> = {}
for (const n of nodes) groups[n.op] = (groups[n.op] ?? 0) + n.elapsed_ms
const report = {
  elapsed_ms,
  groups,
  nodes: nodes.sort((a, b) => b.elapsed_ms - a.elapsed_ms),
}
const output = getEnv('PROFILE_OUTPUT')
if (output) fs.writeFileSync(output, JSON.stringify(report, null, 2))
console.log(
  JSON.stringify({ elapsed_ms, groups, slowest: report.nodes.slice(0, 8) }),
)
features.dispose(); model.dispose(); session.dispose()
