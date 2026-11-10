import fs from 'std:fs'
import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { add, quantized_matmul, slice, transpose } from 'affon:ops'

const cases = JSON.parse(fs.readFileSync(getEnv('AFFON_GENERAL_CASES')!))
const session = new Session({ device: 'cuda', telemetry: false })
const results: any[] = []
const spec = (dtype: string, shape: number[]) => dtype === 'u32' ? Tensor.u32(shape) : dtype === 'bf16' ? Tensor.bf16(shape) : Tensor.f32(shape)
try {
  for (const c of cases) {
    const inputs: any = {}
    try {
      for (const name of ['x', 'w', 's', 'b']) inputs[name] = session.tensor(c.inputs[name].values, { dtype: c.inputs[name].dtype })
      for (const composed of [false, true]) {
        try {
          const source = program('cuda_quantized_general', p => {
            const argument = (name: string) => {
              const input = c.inputs[name]
              const value = p.argument(name, spec(input.dtype, input.shape))
              return input.permutation ? transpose(value, input.permutation) : input.slices ? slice(value, input.slices) : value
            }
            const value = quantized_matmul(argument('x'), argument('w'), argument('s'), argument('b'), { bits: c.bits, group_size: c.group_size })
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
