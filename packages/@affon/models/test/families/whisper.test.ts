import { expect, test } from 'std:test'
import { tensor } from 'affon:compute'
import { create_whisper } from '../../src/whisper/index.ts'

test('Whisper owns suppression, EOS, and fresh caches independently of HF loading', () => {
  const config = { width: 2, heads: 1, layers: 1, vocabSize: 5, contextLength: 5, cacheCapacity: 5, prefix: [1, 2], eosTokenId: 3, suppressTokens: [0], beginSuppressTokens: [3] }
  const parameters = {
    embeddings: tensor(Array.from({ length: 5 }, () => [0, 0]), { device: 'cpu' }),
    positions: tensor(Array.from({ length: 5 }, () => [0, 0]), { device: 'cpu' }),
  }
  let step = 0
  const model = create_whisper(config, {
    encode: (features) => { step = 0; return features },
    cross: () => ({}),
    step: (inputs) => {
      if (step++ === 0) expect(inputs.past_0.to_array()).toEqual([[[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]])
      return {
        logits: tensor([[[100, 0, 0, 10, 5]]], { device: 'cpu' }),
        present_0: tensor([[[[1, 1]]]], { device: 'cpu' }),
        present_1: tensor([[[[2, 2]]]], { device: 'cpu' }),
      }
    },
  }, parameters, ids => ids.join(','))
  for (let run = 0; run < 2; run++) {
    const observed: number[] = []
    const result = model.transcribe(tensor([0], { device: 'cpu' }), index => observed.push(index))
    expect(result.tokens).toEqual([1, 2, 4, 3])
    expect(result.text).toBe('1,2,4,3')
    expect(result.truncated).toBe(false)
    expect(observed).toEqual([0, 1])
  }
  expect(() => create_whisper({ ...config, cacheCapacity: 6 }, {} as any, parameters, () => '')).toThrow()
})
