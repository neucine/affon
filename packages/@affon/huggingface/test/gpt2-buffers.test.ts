import { describe, expect, test } from 'std:test'
import { Session } from 'affon:compute'
import { validate_causal_buffer } from '../src/adapters/gpt2-buffers.ts'

describe('optional GPT-2 checkpoint buffers', () => {
  test('accepts only the exact causal allow-mask', () => {
    const session = new Session({ device: 'cpu' })
    validate_causal_buffer(session.tensor([[[[1, 0], [1, 1]]]]) as any, 2)
    expect(() => validate_causal_buffer(session.tensor([[[[1, 1], [1, 1]]]]) as any, 2)).toThrow('values')
    expect(() => validate_causal_buffer(session.tensor([[1, 0], [1, 1]]) as any, 2)).toThrow('shape/dtype')
    session.dispose()
  })
})
