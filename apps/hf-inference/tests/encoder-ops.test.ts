import fs from 'std:fs'
import { describe, expect, test, values } from 'std:test'
import { tensor } from 'affon:compute'
import { erf_gelu } from '../../../packages/@affon/huggingface/src/encoder-ops.ts'

describe('encoder GELU approximation', () => {
  test('matches independent PyTorch erf-GELU values over [-8, 8]', () => {
    const reference = JSON.parse(fs.readFileSync('apps/hf-inference/tests/fixtures/encoder-gelu-reference.json'))
    const output = erf_gelu(tensor(reference.input, { dtype: 'f32' }))
    expect(values(output)).toBeAllClose(reference.output, { atol: 1e-6, rtol: 1e-6 })
  })
})
