import { matmul, mul, reshape, softmax, transpose } from 'affon:ops'
import fs from 'std:fs'
import { getEnv } from 'std:process'
import telemetry from 'std:telemetry'
import checkpoint from 'affon:checkpoint'
import { Session, Tensor, program } from 'affon:compute'

const root = getEnv('OPERATOR_FIXTURES') ?? '/private/tmp/whisper-operator-fixtures'
const cases = JSON.parse(fs.readFileSync(`${root}/cases.json`))
const metrics = () => Object.fromEntries(telemetry.metrics().filter(m => m.scope === 'compute.execution' && m.name.startsWith('metal_command_')).map(m => [m.name, m.value]))
const results = cases.map((entry: any) => {
  const loaded = checkpoint.load(`${root}/${entry.name}.safetensors`) as unknown as Record<string, { shape: readonly number[]; to_array(): unknown; dispose(): void }>
  const leftShape = entry.flatten_a ? loaded.a.shape.slice(-2) : loaded.a.shape
  const rightShape = [...loaded.b.shape]
  if (entry.transpose_b) [rightShape[rightShape.length - 2], rightShape[rightShape.length - 1]] = [rightShape[rightShape.length - 1], rightShape[rightShape.length - 2]]
  const outputRank = Math.max(leftShape.length, rightShape.length)
  const source = program(`operator_${entry.name}`, builder => {
    const constant = (name: string) => builder.constant(name, loaded[name].to_array(), Tensor.f32(loaded[name].shape))
    let left = constant('a'), right = constant('b')
    if (entry.flatten_a) left = reshape(left, leftShape)
    if (entry.transpose_b) {
      const permutation = [...loaded.b.shape.keys()]
      ;[permutation[permutation.length - 2], permutation[permutation.length - 1]] = [permutation[permutation.length - 1], permutation[permutation.length - 2]]
      right = transpose(right, permutation)
    }
    const product = matmul(left, right)
    if (entry.kind !== 'attention') return product
    return matmul(softmax(mul(product, constant('scale')), outputRank - 1), constant('v'))
  })
  const session = new Session({ device: 'metal' }), executable = session.compile(source), state = session.initialize(source)
  for (const value of Object.values(loaded)) value.dispose()
  let output = executable.run({}, state) as Tensor
  for (let index = 1; index < entry.warmup; index++) { output.dispose(); output = executable.run({}, state) as Tensor }
  const runs = Array.from({ length: entry.runs }, () => {
    const before = metrics(), started = Date.now()
    for (let index = 0; index < entry.repeats; index++) { output.dispose(); output = executable.run({}, state) as Tensor }
    const wall_ms = (Date.now() - started) / entry.repeats, after = metrics()
    return { wall_ms, counters: Object.fromEntries(Object.entries(after).map(([key, value]) => [key, (value - (before[key] ?? 0)) / entry.repeats])) }
  })
  const flat = (output.to_array() as any[]).flat(Infinity).map(Number)
  const samples = entry.samples.map((sample: any) => ({ ...sample, actual: flat[sample.index], error: Math.abs(flat[sample.index] - sample.expected) }))
  output.dispose(); state.dispose(); session.dispose()
  return { name: entry.name, runs, samples }
})
fs.writeFileSync(getEnv('PROFILE_OUTPUT')!, JSON.stringify({ results }, null, 2))
