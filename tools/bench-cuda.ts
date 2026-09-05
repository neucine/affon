import { tensor, sum, matmul, cross_entropy_indexed, setDevice } from 'affon:compute'

// Transfers and first-use compilation are excluded. Reading each scalar waits
// for completion, so these are synchronous end-to-end operation timings.
const iterations = 20
function measure(name: string, operation: () => any) {
  for (let i = 0; i < 3; i++) operation().item()
  const start = Date.now()
  let checksum = 0
  for (let i = 0; i < iterations; i++) checksum = operation().item()
  return { name, milliseconds_per_iteration: (Date.now() - start) / iterations, checksum }
}
const matrix = Array.from({ length: 512 }, (_, i) => Array.from({ length: 512 }, (_, j) => ((i + j) % 17 - 8) * 0.01))
const logits = Array.from({ length: 512 }, (_, i) => Array.from({ length: 1024 }, (_, j) => ((i + j) % 31 - 15) * 0.1))
const targets = Array.from({ length: 512 }, (_, i) => i % 1024)
const values = Array.from({ length: 1024 * 1024 }, (_, i) => (i % 17) - 8)
for (const device of ['cpu', 'cuda'] as const) {
  setDevice(device)
  const a = tensor(matrix), b = tensor(matrix), x = tensor(values)
  const l = tensor(logits), t = tensor(targets, { dtype: 'i64' })
  const results = [
    measure('sum 1048576 f32', () => sum(x)),
    measure('matmul 512x512 plus sum', () => sum(matmul(a, b))),
    measure('indexed cross entropy 512x1024', () => cross_entropy_indexed(l, t)),
  ]
  console.log(JSON.stringify({ device, iterations, results }))
}
setDevice('cpu')
