import { describe, test, expect } from 'std:test'
import nn from 'affon:nn/legacy'
import { tensor } from 'affon:compute/legacy'
import { captureError } from '../../support/errors.ts'

describe('nn recurrent contracts', () => {
  test('recurrent constructors reject invalid dimensions and options as invalid_arg', () => {
    const simpleInput = captureError(() => (nn as any).SimpleRNN(0, 4))
    expect(simpleInput).toBeInstanceOf(AffonError)
    expect(simpleInput.code).toBe('invalid_arg')
    expect(simpleInput.message).toContain('SimpleRNN input_size')

    const simpleHidden = captureError(() => (nn as any).SimpleRNN(3, 0))
    expect(simpleHidden).toBeInstanceOf(AffonError)
    expect(simpleHidden.code).toBe('invalid_arg')
    expect(simpleHidden.message).toContain('SimpleRNN hidden_size')

    const simpleNonlinearity = captureError(() => (nn as any).SimpleRNN(3, 4, { nonlinearity: 'swish' }))
    expect(simpleNonlinearity).toBeInstanceOf(AffonError)
    expect(simpleNonlinearity.code).toBe('invalid_arg')
    expect(simpleNonlinearity.message).toContain('nonlinearity')

    const rnnInput = captureError(() => (nn as any).RNN(0, 4))
    expect(rnnInput).toBeInstanceOf(AffonError)
    expect(rnnInput.code).toBe('invalid_arg')
    expect(rnnInput.message).toContain('RNN input_size')

    const rnnHidden = captureError(() => (nn as any).RNN(3, 0))
    expect(rnnHidden).toBeInstanceOf(AffonError)
    expect(rnnHidden.code).toBe('invalid_arg')
    expect(rnnHidden.message).toContain('RNN hidden_size')

    const lstmInput = captureError(() => (nn as any).LSTM(0, 4))
    expect(lstmInput).toBeInstanceOf(AffonError)
    expect(lstmInput.code).toBe('invalid_arg')
    expect(lstmInput.message).toContain('LSTM input_size')

    const lstmHidden = captureError(() => (nn as any).LSTM(3, 0))
    expect(lstmHidden).toBeInstanceOf(AffonError)
    expect(lstmHidden.code).toBe('invalid_arg')
    expect(lstmHidden.message).toContain('LSTM hidden_size')
  })

  test('SimpleRNN rejects invalid input and h0 shapes', () => {
    const rnn = nn.SimpleRNN(3, 4, { num_layers: 2 })

    const inputErr = captureError(() => rnn(tensor([1, 2, 3], { dtype: 'f32' })))
    expect(inputErr).toBeInstanceOf(AffonError)
    expect(inputErr.code).toBe('invalid_shape')
    expect(inputErr.message).toContain('SimpleRNN expects input shaped')

    const mismatchErr = captureError(() =>
      rnn(tensor([[1, 2]], { dtype: 'f32' })))
    expect(mismatchErr).toBeInstanceOf(AffonError)
    expect(mismatchErr.code).toBe('shape_mismatch')
    expect(mismatchErr.message).toContain('input_size mismatch')

    const h0Err = captureError(() =>
      rnn(
        tensor([[1, 2, 3]], { dtype: 'f32' }),
        tensor([[0, 0, 0, 0]], { dtype: 'f32' }),
      ))
    expect(h0Err).toBeInstanceOf(AffonError)
    expect(h0Err.code).toBe('invalid_shape')
    expect(h0Err.message).toContain('h0 must have shape [num_layers, hidden_size]')
  })

  test('RNN rejects invalid input and h0 shapes in both layout modes', () => {
    const seqFirst = nn.RNN(3, 4, { num_layers: 2 })
    const batchFirst = nn.RNN(3, 4, { num_layers: 2, batch_first: true })

    const seqInputErr = captureError(() => seqFirst(tensor([[1, 2, 3]], { dtype: 'f32' })))
    expect(seqInputErr).toBeInstanceOf(AffonError)
    expect(seqInputErr.code).toBe('invalid_shape')
    expect(seqInputErr.message).toContain('RNN expects input shaped [seq_len, batch, input_size]')

    const seqH0Err = captureError(() =>
      seqFirst(
        tensor([[[1, 2, 3]]], { dtype: 'f32' }),
        tensor([[[0, 0, 0, 0]]], { dtype: 'f32' }),
      ))
    expect(seqH0Err).toBeInstanceOf(AffonError)
    expect(seqH0Err.code).toBe('invalid_shape')
    expect(seqH0Err.message).toContain('RNN h0 must have shape [num_layers, batch, hidden_size]')

    const batchInputErr = captureError(() => batchFirst(tensor([[1, 2, 3]], { dtype: 'f32' })))
    expect(batchInputErr).toBeInstanceOf(AffonError)
    expect(batchInputErr.code).toBe('invalid_shape')
    expect(batchInputErr.message).toContain('when batch_first=true')
  })

  test('LSTM rejects invalid input, h0, and c0 shapes in both layout modes', () => {
    const seqFirst = nn.LSTM(3, 4, { num_layers: 2 })
    const batchFirst = nn.LSTM(3, 4, { num_layers: 2, batch_first: true })

    const seqInputErr = captureError(() => seqFirst(tensor([[1, 2, 3]], { dtype: 'f32' })))
    expect(seqInputErr).toBeInstanceOf(AffonError)
    expect(seqInputErr.code).toBe('invalid_shape')
    expect(seqInputErr.message).toContain('LSTM expects input shaped [seq_len, batch, input_size]')

    const seqH0Err = captureError(() =>
      seqFirst(
        tensor([[[1, 2, 3]]], { dtype: 'f32' }),
        tensor([[[0, 0, 0, 0]]], { dtype: 'f32' }),
      ))
    expect(seqH0Err).toBeInstanceOf(AffonError)
    expect(seqH0Err.code).toBe('invalid_shape')
    expect(seqH0Err.message).toContain('LSTM h0 must have shape [num_layers, batch, hidden_size]')

    const seqC0Err = captureError(() =>
      seqFirst(
        tensor([[[1, 2, 3]]], { dtype: 'f32' }),
        undefined as any,
        tensor([[[0, 0, 0, 0]]], { dtype: 'f32' }),
      ))
    expect(seqC0Err).toBeInstanceOf(AffonError)
    expect(seqC0Err.code).toBe('invalid_arg')
    expect(seqC0Err.message).toContain('expects compute values/parameters')

    const batchInputErr = captureError(() => batchFirst(tensor([[1, 2, 3]], { dtype: 'f32' })))
    expect(batchInputErr).toBeInstanceOf(AffonError)
    expect(batchInputErr.code).toBe('invalid_shape')
    expect(batchInputErr.message).toContain('when batch_first=true')
  })
})
