import fs from 'std:fs'
import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:core'
import { add, slice, topk, transpose } from 'affon:ops'

const cases = JSON.parse(fs.readFileSync(getEnv('AFFON_TOPK_GENERAL_CASES')!),
  (_key: string, value: any) => value === '__NaN__' ? NaN : value) as any[]
const session = new Session({ device: 'cuda', telemetry: false })
const results: any[] = []
const spec = (dtype: string, shape: number[]) => dtype === 'bf16' ? Tensor.bf16(shape) : Tensor.f32(shape)
try {
  for (const c of cases) {
    const input = session.tensor(c.physical, { dtype: c.dtype })
    try {
      for (const composed of [false, true]) {
        try {
          const source = program('cuda_topk_general', p => {
            const argument = p.argument('x', spec(c.dtype, c.physical_shape))
            const viewed = c.permutation ? transpose(argument, c.permutation) :
              c.slices ? slice(argument, c.slices) : argument
            const [values, indices] = topk(viewed, c.k, c.axis, c.largest)
            return [composed ? add(values, values) : values, indices]
          })
          const executable = session.compile(source, { explanationLevel: 'detailed' })
          try {
            const [values, indices] = executable.run({ x: input })
            try {
              const explanation = executable.explain() as any
              results.push({ id: c.id, composed, value_shape: values.shape,
                value_dtype: values.dtype, index_dtype: indices.dtype,
                values: (values.to_array() as any[]).flat(Infinity).map((value: number) => Number.isNaN(value) ? '__NaN__' : value),
                indices: (indices.to_array() as any[]).flat(Infinity),
                path: explanation.executable.physical_nodes.map((node: any) => node.implementation_id || 'view') })
            } finally { values.dispose(); indices.dispose() }
          } finally { executable.dispose() }
        } catch (error) { results.push({ id: c.id, composed, error: String(error) }) }
      }
    } finally { input.dispose() }
  }
  fs.writeFileSync(getEnv('AFFON_TOPK_GENERAL_OUTPUT')!, JSON.stringify(results))
  console.log(JSON.stringify({ checks: results.length }))
} finally { session.dispose() }
