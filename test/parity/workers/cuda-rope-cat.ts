import { Session, Tensor, program } from 'affon:compute'
import { add, cat } from 'affon:ops'

for (const dtype of ['f32', 'bf16'] as const) {
  for (const heads of [8, 32]) {
    const width = 64
    const shape: [number, number, number, number] = [1, 1, heads, width]
    const leftValues = [[Array.from({ length: heads }, (_, head) =>
      Array.from({ length: width }, (_, column) => ((head * 7 + column * 3) % 67) / 4))]]
    const rightValues = [[Array.from({ length: heads }, (_, head) =>
      Array.from({ length: width }, (_, column) => -((head * 5 + column * 11) % 71) / 4))]]
    const session = new Session({ device: 'cuda', telemetry: false })
    const spec = dtype === 'f32' ? Tensor.f32(shape) : Tensor.bf16(shape)
    const left = session.tensor(leftValues, { dtype })
    const right = session.tensor(rightValues, { dtype })
    try {
      for (const kind of ['operator_peer', 'program_peer'] as const) {
        const source = program(`cuda_rope_cat_${dtype}_${heads}_${kind}`, p => {
          const joined = cat([p.argument('left', spec), p.argument('right', spec)], 3)
          return kind === 'program_peer' ? add(joined, joined) : joined
        })
        const executable = session.compile(source)
        try {
          const result = executable.run({ left, right })
          try {
            const flat = (result.to_array() as number[][][][]).flat(Infinity) as number[]
            for (let offset = 0; offset < flat.length; offset += 256)
              console.log(JSON.stringify({ dtype, heads, kind, count: flat.length,
                offset, values: flat.slice(offset, offset + 256) }))
          } finally { result.dispose() }
        } finally { executable.dispose() }
      }
    } finally { left.dispose(); right.dispose(); session.dispose() }
  }
}
