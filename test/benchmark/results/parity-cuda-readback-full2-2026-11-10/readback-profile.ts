import { Session, Tensor, program } from 'affon:compute'
import { add } from 'affon:ops'

const session = new Session({ device: 'cuda' })
const shape = [128, 512]
const n = shape[0] * shape[1]
const a = new Float32Array(n)
const b = new Float32Array(n)
for (let i = 0; i < n; i++) {
  a[i] = 1 + i / 8
  b[i] = 0.5 + i / 8
}
const x = session.fromBytes(a.buffer, Tensor.f32(shape))
const y = session.fromBytes(b.buffer, Tensor.f32(shape))

function nest(f: Float32Array, dims: number[]): any {
  if (dims.length === 1) return Array.from(f)
  const rows = new Array(dims[0])
  const width = dims.slice(1).reduce((p, d) => p * d, 1)
  for (let i = 0; i < dims[0]; i++) {
    rows[i] = nest(f.subarray(i * width, (i + 1) * width), dims.slice(1))
  }
  return rows
}

function nestLoops(f: Float32Array, dims: number[]): any {
  if (dims.length === 1) {
    const row = new Array(dims[0])
    for (let i = 0; i < dims[0]; i++) row[i] = f[i]
    return row
  }
  const rows = new Array(dims[0])
  const width = dims.slice(1).reduce((p, d) => p * d, 1)
  for (let i = 0; i < dims[0]; i++) {
    rows[i] = nestLoops(f.subarray(i * width, (i + 1) * width), dims.slice(1))
  }
  return rows
}

try {
  const source = program('split', p => {
    const left = p.argument('x', Tensor.f32(shape))
    const right = p.argument('y', Tensor.f32(shape))
    return add(left, right)
  })
  const exe = session.compile(source, { denseInputs: false })
  try {
    for (let i = 0; i < 20; i++) {
      const result = exe.run({ x, y })
      result.to_array()
      result.dispose()
    }
    const bytes: number[] = []
    const arrays: number[] = []
    const jsFrom: number[] = []
    const jsLoop: number[] = []
    for (let i = 0; i < 30; i++) {
      let result = exe.run({ x, y })
      let start = performance.now()
      const raw = result.to_bytes() as Uint8Array
      bytes.push(performance.now() - start)
      const values = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4)
      start = performance.now()
      nest(values, shape)
      jsFrom.push(performance.now() - start)
      start = performance.now()
      nestLoops(values, shape)
      jsLoop.push(performance.now() - start)
      result.dispose()
      result = exe.run({ x, y })
      start = performance.now()
      result.to_array()
      arrays.push(performance.now() - start)
      result.dispose()
    }
    const median = (values: number[]) =>
      values.slice().sort((left, right) => left - right)[Math.floor(values.length / 2)]
    console.log(JSON.stringify({
      bytes_ms: median(bytes),
      to_array_ms: median(arrays),
      js_array_from_ms: median(jsFrom),
      js_loops_ms: median(jsLoop),
    }))
  } finally {
    exe.dispose()
  }
} finally {
  x.dispose()
  y.dispose()
  session.dispose()
}
