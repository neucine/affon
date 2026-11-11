import { testDevices, testOn } from '../../devices.ts'
import { Session, Tensor, program } from 'affon:compute'
import { add, binary_cross_entropy, binary_cross_entropy_with_logits, cross_entropy, layer_norm, log_softmax, mean_absolute_error, mean_squared_error, rms_norm, softmax } from 'affon:ops'
import { describe, expect, test, values } from 'std:test'

describe('Normalization and loss operation contracts', () => {
  testOn('metal')('Metal keeps a large normalization between GPU producer and consumer', () => {
    const source = program('large_normalization_resident', p => {
      const input = p.argument('input', Tensor.f32([129, 257]))
      const normalized = layer_norm(add(input, input), 0)
      return add(normalized, normalized)
    })
    const session = new Session({ device: 'metal' })
    const executable = session.compile(source)
    const input = session.tensor(Array.from({ length: 129 }, (_, row) => Array(257).fill(row)))
    const output = executable.run({ input })
    const variance = (129 * 129 - 1) / 12
    const expected = Array.from({ length: 129 }, (_, row) =>
      Array(257).fill(2 * (row - 64) / Math.sqrt(variance + 1e-5 / 4)))
    expect(values(output)).toBeAllClose(expected, { atol: 2e-4, rtol: 2e-4 })
    output.dispose(); input.dispose(); executable.dispose(); session.dispose()
  })

  testOn('metal')('Metal large normalization scales extreme finite values', () => {
    const source = program('large_normalization_extremes', p => layer_norm(
      p.argument('input', Tensor.f32([128, 64])), 0))
    const session = new Session({ device: 'metal' })
    const executable = session.compile(source)
    const input = session.tensor(Array.from({ length: 128 }, (_, row) =>
      Array(64).fill(row % 2 ? 1e20 : -1e20)))
    const output = executable.run({ input })
    const expected = Array.from({ length: 128 }, (_, row) => Array(64).fill(row % 2 ? 1 : -1))
    expect(values(output)).toBeAllClose(expected, { atol: 2e-4, rtol: 2e-4 })
    output.dispose(); input.dispose(); executable.dispose(); session.dispose()
  })

  testOn('metal')('Metal keeps large BF16 normalization and its consumer on GPU', () => {
    const source = program('large_bf16_normalization_resident', p => {
      const input = p.argument('input', Tensor.bf16([128, 64]))
      const normalized = layer_norm(input, 0)
      return add(normalized, normalized)
    })
    const session = new Session({ device: 'metal' })
    const executable = session.compile(source)
    const input = session.tensor(Array.from({ length: 128 }, (_, row) =>
      Array(64).fill(row % 2 ? 1e20 : -1e20)), { dtype: 'bf16' })
    const output = executable.run({ input })
    const expected = Array.from({ length: 128 }, (_, row) => Array(64).fill(row % 2 ? 2 : -2))
    expect(values(output)).toBeAllClose(expected, { atol: 2e-2, rtol: 2e-2 })
    output.dispose(); input.dispose(); executable.dispose(); session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) {
    test(`softmax and log_softmax normalize large scores on ${device}`, () => {
      const session = new Session({ device })
      const scores = session.tensor([[1000, 1000, 998], [-1000, -1000, -1002]])
      const probabilities = softmax(scores, -1)
      const logProbabilities = log_softmax(scores, -1)
      const denominator = 2 + Math.exp(-2)
      const expected = [1 / denominator, 1 / denominator, Math.exp(-2) / denominator]
      expect(values(probabilities)).toBeAllClose([expected, expected], { atol: 1e-5, rtol: 1e-5 })
      expect(values(logProbabilities)).toBeAllClose([expected.map(Math.log), expected.map(Math.log)], { atol: 1e-5, rtol: 1e-5 })
      probabilities.dispose(); logProbabilities.dispose(); scores.dispose(); session.dispose()
    })

    test(`layer_norm is invariant to row offsets and rms_norm handles zeros on ${device}`, () => {
      const source = program(`normalization_${device}`, p => {
        const input = p.argument('input', Tensor.f32([2, 2]))
        return [layer_norm(input, -1), rms_norm(input, -1)]
      })
      const session = new Session({ device })
      const input = session.tensor([[1, 3], [11, 13]])
      const executable = session.compile(source)
      const [centered, scaled] = executable.run({ input }) as Tensor[]
      expect(values(centered)).toBeAllClose([[-1, 1], [-1, 1]], { atol: 1e-4, rtol: 1e-4 })
      expect(values(scaled)).toBeAllClose([
        [1 / Math.sqrt(5 + 1e-5), 3 / Math.sqrt(5 + 1e-5)],
        [11 / Math.sqrt(145 + 1e-5), 13 / Math.sqrt(145 + 1e-5)],
      ], { atol: 1e-4, rtol: 1e-4 })
      const zeros = session.tensor([[0, 0], [0, 0]])
      const zeroScaled = rms_norm(zeros, -1)
      expect(zeroScaled.to_array()).toEqual([[0, 0], [0, 0]])
      const again = executable.run({ input }) as Tensor[]
      expect(again[0].to_array()).toEqual(centered.to_array())
      for (const tensor of [centered, scaled, zeroScaled, ...again, input, zeros]) tensor.dispose()
      session.dispose()
    })
  }

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`indexed cross entropy matches an independent two-class loss on ${device}`, () => {
    const session = new Session({ device })
    const logits = session.tensor([[2, 0], [0, 2]])
    const labels = session.tensor([0, 1], { dtype: 'i64' })
    const immediate = cross_entropy(logits, labels)
    const source = program('indexed_loss', p => cross_entropy(
      p.argument('logits', Tensor.f32([2, 2])),
      p.argument('labels', Tensor.i64([2])),
    ))
    const compiled = session.compile(source).run({ logits, labels })
    const expected = Math.log(1 + Math.exp(-2))
    expect(Math.abs(immediate.item() - expected) < 1e-6).toBe(true)
    expect(Math.abs(compiled.item() - expected) < 1e-6).toBe(true)
    immediate.dispose(); compiled.dispose(); logits.dispose(); labels.dispose(); session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`binary losses agree between probabilities and equivalent logits on ${device}`, () => {
    const session = new Session({ device })
    const probabilities = session.tensor([0.25, 0.75])
    const logits = session.tensor([-Math.log(3), Math.log(3)])
    const labels = session.tensor([0, 1])
    const probabilityLoss = binary_cross_entropy(probabilities, labels)
    const logitLoss = binary_cross_entropy_with_logits(logits, labels)
    const expected = -Math.log(0.75)
    expect(Math.abs(probabilityLoss.item() - expected) < 1e-6).toBe(true)
    expect(Math.abs(logitLoss.item() - expected) < 1e-6).toBe(true)
    probabilityLoss.dispose(); logitLoss.dispose(); probabilities.dispose(); logits.dispose(); labels.dispose(); session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`regression losses reduce every element, including negative errors on ${device}`, () => {
    const session = new Session({ device })
    const prediction = session.tensor([[2, -1], [4, 2]])
    const target = session.tensor([[0, 1], [1, 3]])
    const mse = mean_squared_error(prediction, target)
    const mae = mean_absolute_error(prediction, target)
    expect(mse.item()).toBe(4.5)
    expect(mae.item()).toBe(2)
    mse.dispose(); mae.dispose(); prediction.dispose(); target.dispose(); session.dispose()
  })

  for (const device of testDevices(['cpu', 'metal', 'cuda'] as const)) test(`invalid loss shapes do not prevent the same Session from evaluating valid labels on ${device}`, () => {
    const session = new Session({ device })
    const logits = session.tensor([[1, 2], [3, 4]])
    const bad = session.tensor([0], { dtype: 'i64' })
    const good = session.tensor([1, 1], { dtype: 'i64' })
    expect(() => cross_entropy(logits, bad)).toThrow()
    const result = cross_entropy(logits, good)
    expect(Math.abs(result.item() - Math.log(1 + Math.exp(-1))) < 1e-6).toBe(true)
    result.dispose(); good.dispose(); bad.dispose(); logits.dispose(); session.dispose()
  })
})
