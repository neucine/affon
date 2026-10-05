import { expect, test } from 'std:test'
import { Session, Tensor, program } from 'affon:compute'
import { WhisperProgramRuntime } from '../../src/inference/program-runtime.ts'

test('Whisper app policy executes imported-style Programs with fresh request caches', () => {
  const values = new Session({ device: 'cpu' })
  const encoder = program('whisper_encoder_fixture', p => p.argument('features', Tensor.f32([1])))
  const cross = program('whisper_cross_fixture', p => p.argument('encoded', Tensor.f32([1])))
  const decoder = program('whisper_decoder_fixture', p => {
    p.argument('encoded', Tensor.f32([1]))
    const embeddings = p.argument('embeddings', Tensor.f32([1, 1, 5]))
    p.argument('position', Tensor.f32([1, 1, 5]))
    p.argument('mask', Tensor.f32([1, 1, 1, 6]))
    p.argument('past_0', Tensor.f32([1, 1, 5, 5]))
    p.argument('past_1', Tensor.f32([1, 1, 5, 5]))
    const present0 = p.constant('present_0_value', [[[[0, 0, 0, 0, 0]]]], Tensor.f32([1, 1, 1, 5]))
    const present1 = p.constant('present_1_value', [[[[0, 0, 0, 0, 0]]]], Tensor.f32([1, 1, 1, 5]))
    return [embeddings, present0, present1]
  })
  const model = {
    generation: { width: 5, heads: 1, layers: 1, vocabSize: 5, contextLength: 5, cacheCapacity: 5,
      prefix: [1, 2], eosTokenId: 3, suppressTokens: [0], beginSuppressTokens: [3] },
    encoder: { forward: encoder, parameters: {}, output_names: ['output'] },
    cross: { forward: cross, parameters: {}, output_names: ['encoded'] },
    decoder: { forward: decoder, parameters: {}, output_names: ['logits', 'present_0', 'present_1'] },
    embeddings: values.tensor([
      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [100, 0, 0, 10, 5],
      [0, 0, 0, 0, 0], [0, 0, 0, 10, 5],
    ]),
    positions: values.tensor(Array.from({ length: 5 }, () => [0, 0, 0, 0, 0])),
    decode: (ids: number[]) => ids.join(','),
  }
  const runtime = new WhisperProgramRuntime(model, 'cpu')
  for (let request = 0; request < 2; request++) {
    const features = values.tensor([0])
    const observed: number[] = []
    const result = runtime.transcribe(features, step => observed.push(step))
    features.dispose()
    expect(result.tokens).toEqual([1, 2, 4, 3])
    expect(result.text).toBe('1,2,4,3')
    expect(result.truncated).toBe(false)
    expect(observed).toEqual([0, 1])
  }
  const limitedFeatures = values.tensor([0])
  const limited = runtime.transcribe(
    limitedFeatures,
    undefined,
    undefined,
    1,
  )
  limitedFeatures.dispose()
  expect(limited.tokens).toEqual([1, 2, 4])
  expect(limited.truncated).toBe(true)
  runtime.dispose()
  model.embeddings.dispose(); model.positions.dispose(); values.dispose()
})
