import { describe, test, expect } from 'std:test'
import { adam, clear_grad, grad, tensor } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import nn from 'affon:nn'

describe('nn recurrent layers', () => {
  test('SimpleRNN preserves sequence output, hidden state, and parameter naming', () => {
    const rnn = nn.SimpleRNN(3, 4, { num_layers: 2 })
    const x = tensor([
      [0.1, 0.2, 0.3],
      [0.3, 0.2, 0.1],
      [0.5, 0.4, 0.3],
    ], { dtype: 'f32' })

    const y = rnn(x)
    const h = rnn.hidden_state()

    expect(y.shape).toEqual([3, 4])
    expect(h).toBeDefined()
    expect((h as Tensor).shape).toEqual([2, 4])
    expect(rnn.parameters.named()[0][0]).toBe('layers.0.weight_ih')
    expect(rnn.parameters.named()[3][0]).toBe('layers.1.weight_ih')

    const y2 = rnn(x, tensor([
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ], { dtype: 'f32' }))
    expect(y2.shape).toEqual([3, 4])
  })

  test('RNN sequence-first preserves output and hidden state shape', () => {
    const rnn = nn.RNN(3, 4, { num_layers: 2 })
    const x = tensor([
      [
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
      ],
      [
        [0.5, 0.4, 0.3],
        [0.2, 0.4, 0.6],
      ],
      [
        [0.7, 0.8, 0.9],
        [0.9, 0.8, 0.7],
      ],
    ], { dtype: 'f32' }) as Tensor<[number, number, 3], 'f32'>

    const y = rnn(x)
    const h = rnn.hidden_state()

    expect(y.shape).toEqual([3, 2, 4])
    expect(h).toBeDefined()
    expect((h as Tensor).shape).toEqual([2, 2, 4])
    expect(rnn.parameters.named().length).toBe(6)
  })

  test('RNN batch_first preserves output and hidden state shape', () => {
    const rnn = nn.RNN(3, 4, { num_layers: 2, batch_first: true })
    const x = tensor([
      [
        [0.1, 0.2, 0.3],
        [0.5, 0.4, 0.3],
        [0.7, 0.8, 0.9],
      ],
      [
        [0.3, 0.2, 0.1],
        [0.2, 0.4, 0.6],
        [0.9, 0.8, 0.7],
      ],
    ], { dtype: 'f32' }) as Tensor<[number, number, 3], 'f32'>

    const y = rnn(x)
    const h = rnn.hidden_state()

    expect(y.shape).toEqual([2, 3, 4])
    expect(h).toBeDefined()
    expect((h as Tensor).shape).toEqual([2, 2, 4])
  })

  test('LSTM sequence-first preserves output, hidden state, and cell state shape', () => {
    const lstm = nn.LSTM(3, 4, { num_layers: 2 })
    const x = tensor([
      [
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
      ],
      [
        [0.5, 0.4, 0.3],
        [0.2, 0.4, 0.6],
      ],
      [
        [0.7, 0.8, 0.9],
        [0.9, 0.8, 0.7],
      ],
    ], { dtype: 'f32' }) as Tensor<[number, number, 3], 'f32'>

    const y = lstm(x)
    const h = lstm.hidden_state()
    const c = lstm.cell_state()

    expect(y.shape).toEqual([3, 2, 4])
    expect((h as Tensor).shape).toEqual([2, 2, 4])
    expect((c as Tensor).shape).toEqual([2, 2, 4])
    expect(lstm.parameters.named().length).toBe(6)
  })

  test('LSTM batch_first preserves output, hidden state, and cell state shape', () => {
    const lstm = nn.LSTM(3, 4, { num_layers: 2, batch_first: true })
    const x = tensor([
      [
        [0.1, 0.2, 0.3],
        [0.5, 0.4, 0.3],
        [0.7, 0.8, 0.9],
      ],
      [
        [0.3, 0.2, 0.1],
        [0.2, 0.4, 0.6],
        [0.9, 0.8, 0.7],
      ],
    ], { dtype: 'f32' }) as Tensor<[number, number, 3], 'f32'>

    const y = lstm(x)
    const h = lstm.hidden_state()
    const c = lstm.cell_state()

    expect(y.shape).toEqual([2, 3, 4])
    expect((h as Tensor).shape).toEqual([2, 2, 4])
    expect((c as Tensor).shape).toEqual([2, 2, 4])
  })

  test('RNN sequential training step matches notebook usage', () => {
    const model = nn.Sequential(
      nn.RNN(3, 4),
      x => x,
    )

    const criterion = nn.MSELoss()
    const params = model.parameters
    const step = adam({ lr: 1e-2 })

    const xBatch = tensor([
      [
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
      ],
      [
        [0.5, 0.4, 0.3],
        [0.2, 0.4, 0.6],
      ],
      [
        [0.7, 0.8, 0.9],
        [0.9, 0.8, 0.7],
      ],
    ], { dtype: 'f32' }) as Tensor<[number, number, 3], 'f32'>

    const target = tensor([
      [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
      ],
      [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
      ],
      [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
      ],
    ], { dtype: 'f32' })

    clear_grad(params)
    const pred = model(xBatch)
    const loss = criterion(pred, target)

    expect(pred.shape).toEqual([3, 2, 4])
    expect(loss.shape).toEqual([1])

    grad(loss, params)
    step(params)

    expect(params[0].grad).toBeDefined()
  })
})
