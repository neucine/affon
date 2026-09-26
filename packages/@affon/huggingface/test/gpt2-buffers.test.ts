import { describe, expect, test } from 'std:test'
import { tensor } from 'affon:compute'
import { validate_causal_buffer } from '../src/gpt2-buffers.ts'

describe('legacy GPT-2 checkpoint buffers', () => {
  test('accepts only the exact causal allow-mask', () => {
    validate_causal_buffer(tensor([[[[1, 0], [1, 1]]]], { dtype: 'f32' }), 2)
    expect(() => validate_causal_buffer(tensor([[[[1, 1], [1, 1]]]], { dtype: 'f32' }), 2)).toThrow('values')
    expect(() => validate_causal_buffer(tensor([[1, 0], [1, 1]], { dtype: 'f32' }), 2)).toThrow('shape/dtype')
  })
})
