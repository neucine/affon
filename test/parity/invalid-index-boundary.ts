import { Session, Tensor, program } from 'affon:compute'
import { scatter_add, topk } from 'affon:ops'

for (const dtype of ['f32', 'bf16'] as const) {
  const session = new Session({ device: 'metal', telemetry: false })
  const base = session.tensor([[1, 2], [3, 4]], { dtype })
  const indices = session.tensor([[3, 0]], { dtype: 'i64' })
  const updates = session.tensor([[1, 1]], { dtype })
  try {
    try {
      const result = scatter_add(base, indices, updates, 0)
      try { console.log(JSON.stringify({ id: `metal/${dtype}/scatter_add/invalid_index`,
        status: 'returned', output: result.to_array() })) }
      finally { result.dispose() }
    } catch (error) {
      console.log(JSON.stringify({ id: `metal/${dtype}/scatter_add/invalid_index`,
        status: 'error', error: String(error) }))
    }
    try {
      program('invalid_topk_boundary', p => topk(p.argument('x',
        dtype === 'f32' ? Tensor.f32([2, 2]) : Tensor.bf16([2, 2])), 3, 1))
      console.log(JSON.stringify({ id: `metal/${dtype}/topk/invalid_index`, status: 'accepted' }))
    } catch (error) {
      console.log(JSON.stringify({ id: `metal/${dtype}/topk/invalid_index`,
        status: 'error', error: String(error) }))
    }
  } finally { updates.dispose(); indices.dispose(); base.dispose(); session.dispose() }
}
