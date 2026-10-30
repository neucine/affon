import { expect, test } from 'std:test'
import { getEnv } from 'std:process'
import { Session, type Tensor } from 'affon:compute'
import { argmax, cross_entropy, dot, layer_norm, mean, min, softmax, std, sum, variance } from 'affon:ops'

const flatten = (value: unknown) => Array.isArray(value) ? value.flat(Infinity) as number[] : [value as number]

test('BF16 cooperative reductions, softmax, and norm match CPU', () => {
  const target = getEnv('AFFON_TEST_CUDA') === '1' ? 'cuda' : getEnv('AFFON_TEST_METAL') === '1' ? 'metal' : 'cpu'
  const run = (device: 'cpu' | 'cuda' | 'metal') => {
    const session = new Session({device})
    const values = Array.from({length: 2}, (_, batch) => Array.from({length: 64}, (_, row) =>
      Array.from({length: 3}, (_, col) => ((batch * 5 + row * 3 + col * 7) % 29 - 14) / 8)))
    const x = session.tensor(values, {dtype: 'bf16'})
    const a = session.tensor(Array.from({length: 257}, (_, index) => (index % 17 - 8) / 8), {dtype: 'bf16'})
    const b = session.tensor(Array.from({length: 257}, (_, index) => (index % 13 - 6) / 8), {dtype: 'bf16'})
    const outputs: Tensor[] = []
    for (const [name, work] of [
      ['dot', () => dot(a, b)], ['sum', () => sum(x)], ['mean', () => mean(x, 1)],
      ['variance', () => variance(x, 1)], ['std', () => std(x, 1)],
      ['min', () => min(x, 1)], ['argmax', () => argmax(x, 1)],
      ['softmax', () => softmax(x, 1)], ['layer_norm', () => layer_norm(x, 1, 1)],
    ] as const) {
      try { outputs.push(work()) } catch (error) { throw Error(`${name}: ${error}`) }
    }
    try { return outputs.map(value => flatten(value.to_array())) }
    finally { outputs.forEach(value => value.dispose()); b.dispose(); a.dispose(); x.dispose(); session.dispose() }
  }
  const expected = run('cpu'), actual = run(target)
  for (let group = 0; group < expected.length; group++) {
    expect(actual[group].length).toBe(expected[group].length)
    for (let index = 0; index < expected[group].length; index++)
      expect(Math.abs(actual[group][index] - expected[group][index]) < (group === 6 ? 0.001 : 0.03125)).toBe(true)
  }
})

test('BF16 indexed cross entropy matches a stable reference', () => {
  const session = new Session({device: getEnv('AFFON_TEST_CUDA') === '1' ? 'cuda' : getEnv('AFFON_TEST_METAL') === '1' ? 'metal' : 'cpu'})
  const data = Array.from({length: 2}, (_, row) =>
    Array.from({length: 64}, (_, col) => ((row * 11 + col * 7) % 23 - 11) / 8))
  const logits = session.tensor(data, {dtype: 'bf16'})
  const labels = session.tensor([7, 31], {dtype: 'i64'})
  const result = cross_entropy(logits, labels)
  try {
    const expected = data.reduce((total, row, index) => {
      const maximum = Math.max(...row)
      return total + maximum + Math.log(row.reduce((sum, value) => sum + Math.exp(value - maximum), 0)) - row[[7, 31][index]]
    }, 0) / data.length
    expect(Math.abs(result.item() - expected) < 0.03125).toBe(true)
  } finally { result.dispose(); labels.dispose(); logits.dispose(); session.dispose() }
})
