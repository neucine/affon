import { Session, Tensor, program } from 'affon:compute'
import { index_select } from 'affon:ops'

for (const dtype of ['f32', 'bf16'] as const) {
  const session = new Session({ device: 'cuda', telemetry: false })
  const source = program(`cuda_index_bounds_${dtype}`, p => index_select(
    p.argument('table', dtype === 'f32' ? Tensor.f32([3, 2]) : Tensor.bf16([3, 2])),
    0, p.argument('indices', Tensor.i64([2]))))
  const executable = session.compile(source)
  const table = session.tensor([[1, 2], [3, 4], [5, 6]], { dtype })
  try {
    for (const values of [[0, 3], [-1, 0]]) {
      const indices = session.tensor(values, { dtype: 'i64' })
      try {
        let rejected = false
        try { executable.run({ table, indices }) } catch { rejected = true }
        if (!rejected) throw Error(`${dtype}: accepted invalid indices ${values}`)
      } finally { indices.dispose() }
    }
    const indices = session.tensor([2, 0], { dtype: 'i64' })
    try {
      const result = executable.run({ table, indices })
      try {
        if (JSON.stringify(result.to_array()) !== JSON.stringify([[5, 6], [1, 2]]))
          throw Error(`${dtype}: failed to recover after invalid indices`)
      } finally { result.dispose() }
    } finally { indices.dispose() }
    console.log(JSON.stringify({ dtype, invalid_high: 'rejected', invalid_negative: 'rejected', recovery: 'passed' }))
  } finally { table.dispose(); executable.dispose(); session.dispose() }
}
