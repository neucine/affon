import fs from 'std:fs'
import { getEnv } from 'std:process'
import telemetry from 'std:telemetry'
import checkpoint from 'affon:checkpoint'
import {
  contiguous,
  matmul,
  mul,
  softmax,
  transpose,
  reshape,
  no_grad,
  type Tensor,
} from 'affon:compute'
const root =
  getEnv('OPERATOR_FIXTURES') ?? '/private/tmp/whisper-operator-fixtures'
const cases = JSON.parse(fs.readFileSync(`${root}/cases.json`))
function metrics() {
  return Object.fromEntries(
    telemetry
      .metrics()
      .filter(
        (m) =>
          m.scope === 'compute.execution' &&
          m.name.startsWith('metal_command_'),
      )
      .map((m) => [m.name, m.value]),
  )
}
const results = no_grad(() =>
  cases.map((c: any) => {
    const loaded = checkpoint.load(`${root}/${c.name}.safetensors`) as Record<
      string,
      Tensor
    >
    const tensors = Object.fromEntries(
      Object.entries(loaded).map(([k, v]) => [k, v.to('metal')]),
    )
    const b = c.transpose_b
      ? transpose(
          tensors.b,
          tensors.b.shape.length - 2,
          tensors.b.shape.length - 1,
        )
      : tensors.b
    const lhs = c.flatten_a
      ? reshape(tensors.a, tensors.a.shape.slice(-2))
      : tensors.a
    function run() {
      const x = matmul(lhs, b)
      return c.kind === 'attention'
        ? matmul(softmax(mul(x, tensors.scale), x.shape.length - 1), tensors.v)
        : x
    }
    let output = run()
    for (let i = 1; i < c.warmup; i++) output = run()
    const runs = Array.from({ length: c.runs }, () => {
      const before = metrics(),
        start = Date.now()
      for (let i = 0; i < c.repeats; i++) output = run()
      const wall_ms = (Date.now() - start) / c.repeats,
        after = metrics()
      const delta = Object.fromEntries(
        Object.entries(after).map(([k, v]) => [
          k,
          (v - (before[k] ?? 0)) / c.repeats,
        ]),
      )
      if (delta.metal_command_count !== delta.metal_command_gpu_valid_count)
        throw Error('Incomplete GPU timings')
      return { wall_ms, counters: delta }
    })
    const flat = reshape(output.to('cpu'), [c.numel])
    const samples = c.samples.map((sample: any) => {
      const actual = (
        contiguous(
          flat.slice([`${sample.index}:${sample.index + 1}`]),
        ).to_array() as number[]
      )[0]
      const error = Math.abs(actual - sample.expected)
      if (!(error <= 1e-4 + Math.abs(sample.expected) * 1e-4))
        throw Error(`${c.name}: sample ${sample.index} error ${error}`)
      return { ...sample, actual, error }
    })
    console.log(
      c.name,
      JSON.stringify(
        runs.map((r) => ({
          wall_ms: r.wall_ms,
          gpu_ms: r.counters.metal_command_gpu_ns / 1e6,
        })),
      ),
    )
    return { name: c.name, runs, samples }
  }),
)
fs.writeFileSync(
  getEnv('PROFILE_OUTPUT')!,
  JSON.stringify({ warmup: 20, repeats: 10, runs: 5, results }, null, 2),
)
