import fs from 'std:fs'
import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { add, dense_cross_entropy, layer_norm, log_softmax_nll, rms_norm, slice, transpose } from 'affon:ops'

const cases = JSON.parse(fs.readFileSync(getEnv('AFFON_GENERAL_CASES')!))
const session = new Session({ device: 'cuda', telemetry: false })
const results: any[] = []
try {
  for (const c of cases) {
    const inputs: any = { x: session.tensor(c.x, { dtype: c.dtype }) }
    if (c.y) inputs.y = session.tensor(c.y, { dtype: c.dtype })
    try {
      for (const composed of [false, true]) {
        try {
          const source = program('cuda_general', p => {
            const spec = c.dtype === 'bf16' ? Tensor.bf16(c.physical_shape) : Tensor.f32(c.physical_shape)
            const view = (v: any) => c.permutation ? transpose(v, c.permutation) : c.slices ? slice(v, c.slices) : v
            const x = view(p.argument('x', spec))
            const y = c.y ? view(p.argument('y', spec)) : null
            const value = c.op === 'layer_norm' ? layer_norm(x, c.axis, c.epsilon) :
              c.op === 'rms_norm' ? rms_norm(x, c.axis, c.epsilon) :
              c.op === 'cross_entropy' ? dense_cross_entropy(x, y) : log_softmax_nll(x, y)
            return composed ? add(value, value) : value
          })
          const executable = session.compile(source, { explanationLevel: 'detailed' })
          try {
            const value = executable.run(inputs)
            try {
              const explanation = executable.explain() as any
              results.push({ id: c.id, composed, shape: value.shape, dtype: value.dtype,
                values: (value.to_array() as any[]).flat(Infinity),
                path: explanation.executable.physical_nodes.map((node: any) => node.implementation_id || 'view') })
            } finally { value.dispose() }
          } finally { executable.dispose() }
        } catch (error) { results.push({ id: c.id, composed, error: String(error) }) }
      }
    } finally { for (const value of Object.values(inputs) as any[]) value.dispose() }
  }
  fs.writeFileSync(getEnv('AFFON_GENERAL_OUTPUT')!, JSON.stringify(results))
  console.log(JSON.stringify({ checks: results.length }))
} finally { session.dispose() }
