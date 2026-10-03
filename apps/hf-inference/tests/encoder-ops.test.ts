import fs from 'std:fs'
import { describe, expect, test } from 'std:test'
import { Session } from 'affon:compute'
import { erf_gelu } from '../src/legacy-encoder.ts'

describe('encoder GELU approximation', () => {
  test('matches independent PyTorch erf-GELU values over [-8, 8]', () => {
    const reference = JSON.parse(fs.readFileSync('apps/hf-inference/tests/fixtures/encoder-gelu-reference.json'))
    const session = new Session({ device: 'cpu' })
    const input = session.tensor(reference.input, { dtype: 'f32' })
    const output = erf_gelu(input)
    expect(output.to_array()).toBeAllClose(reference.output, { atol: 1e-6, rtol: 1e-6 })
    input.dispose()
    session.dispose()
  })
})
