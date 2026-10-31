// Exercises the CUDA rank-one mean path used by Apertus normalization.
import { Session, Tensor, program } from 'affon:compute'
import { mean, slice } from 'affon:ops'

const session = new Session({ device: 'cuda' })
for (const { width, layout } of [
  { width: 31, layout: 'dense' }, { width: 32, layout: 'dense' },
  { width: 128, layout: 'dense' }, { width: 4096, layout: 'dense' },
  { width: 8192, layout: 'dense' }, { width: 128, layout: 'offset' },
  { width: 4096, layout: 'strided' },
] as const) {
  const inputWidth = layout === 'strided' ? width * 2 : layout === 'offset' ? width + 1 : width
  const inputValues = Array.from({ length: inputWidth }, (_, index) => ((index * 17) % 101 - 50) / 32)
  const values = layout === 'dense' ? inputValues : layout === 'offset' ? inputValues.slice(1) : inputValues.filter((_, index) => index % 2 === 1)
  const expected = values.reduce((total, value) => total + value, 0) / width
  const source = program('rank_one_mean', p => {
    const x = p.argument('x', Tensor.f32([inputWidth]))
    return mean(layout === 'dense' ? x : slice(x, [{ start: 1, stop: inputWidth, step: layout === 'strided' ? 2 : 1 }]))
  })
  const executable = session.compile(source)
  const x = session.tensor(inputValues)
  const result = executable.run({ x })
  const actual = result.item()
  const error = Math.abs(actual - expected)
  if (error > 2e-5) throw Error(`${layout} width ${width}: ${actual} vs ${expected}`)
  console.log(JSON.stringify({ width, layout, actual, expected, error }))
  result.dispose()
  x.dispose()
}
