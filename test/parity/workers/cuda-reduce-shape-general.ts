import fs from 'std:fs'
import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:core'
import { add, reduce_to_shape, slice, transpose } from 'affon:ops'

const cases = JSON.parse(fs.readFileSync(getEnv('AFFON_REDUCE_CASES')!)) as any[]
const session = new Session({ device: 'cuda', telemetry: false })
const results: any[] = []
try {
  for (const c of cases) {
    const input = session.tensor(c.physical, { dtype: c.dtype })
    try {
      for (const composed of [false, true]) {
        try {
          const source = program('cuda_reduce_shape_general', p => {
            const spec = c.dtype === 'i64' ? Tensor.i64(c.physical_shape) : Tensor.f32(c.physical_shape)
            const argument = p.argument('x', spec)
            const view = c.permutation ? transpose(argument, c.permutation) :
              c.slices ? slice(argument, c.slices) : argument
            const reduced = reduce_to_shape(view, c.target_shape)
            return composed ? add(reduced, reduced) : reduced
          })
          const executable = session.compile(source, { explanationLevel: 'detailed' })
          try {
            const output = executable.run({ x: input })
            try {
              const explanation = executable.explain() as any
              const raw = output.to_array() as any
              results.push({ id: c.id, composed, shape: output.shape, dtype: output.dtype,
                values: typeof raw === 'number' ? [raw] : (raw as any[]).flat(Infinity),
                path: explanation.executable.physical_nodes.map((node: any) => node.implementation_id || 'view') })
            } finally { output.dispose() }
          } finally { executable.dispose() }
        } catch (error) { results.push({ id: c.id, composed, error: String(error) }) }
      }
    } finally { input.dispose() }
  }
  fs.writeFileSync(getEnv('AFFON_REDUCE_OUTPUT')!, JSON.stringify(results))
  console.log(JSON.stringify({ checks: results.length }))
} finally { session.dispose() }
