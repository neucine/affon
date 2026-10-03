import { describe, expect, test } from 'std:test'
import { Session } from 'affon:compute'
import { adam } from 'affon:optim'
import { DecoderModel } from '../src/model.ts'
import { CausalLMLoss, generate } from '../src/causal-lm.ts'

describe('decoder-lm Program model', () => {
  test('runs forward, causal loss, generation, and an optimizer step', () => {
    const model = DecoderModel(16, 8, { numLayers: 1, numHeads: 2, hiddenDim: 16, causal: true, positional: 'learned', maxSeqLen: 8, tieEmbeddings: true, seed: 7 })
    const external = new Session({ device: 'cpu' })
    const ids = external.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'i64' })
    const logits = model(ids)
    expect(logits.shape).toEqual([2, 3, 16])
    expect((logits.to_array() as number[][][]).flat(2).every(Number.isFinite)).toBe(true)

    const targets = external.tensor([[1, 2, 3, 4], [4, 5, 6, 7]], { dtype: 'i64' })
    const loss = CausalLMLoss()(logits, targets)
    expect(Number.isFinite(loss.item())).toBe(true)
    const generated = generate(model, ids, { max_new_tokens: 2, forbidden_token_ids: [0] })
    expect(generated.shape).toEqual([2, 5])

    const before = model.state()['decoder_model.token_embedding'].to_array()
    const trainLoss = model.trainBatch([[1, 2, 3, 4], [4, 5, 6, 7]], adam({ learning_rate: 0.001 }))
    expect(Number.isFinite(trainLoss)).toBe(true)
    expect(model.state()['decoder_model.token_embedding'].to_array()).not.toEqual(before)

    generated.dispose(); loss.dispose(); logits.dispose(); targets.dispose(); ids.dispose(); external.dispose(); model.dispose()
  })

  test('validates dimensions, context, and generation options', () => {
    expect(() => DecoderModel(0, 8, { numLayers: 1, numHeads: 2, hiddenDim: 16, maxSeqLen: 8 })).toThrow()
    const model = DecoderModel(8, 4, { numLayers: 1, numHeads: 2, hiddenDim: 8, maxSeqLen: 3 })
    const ids = model.session.tensor([[1, 2, 3, 4]], { dtype: 'i64' })
    expect(() => model(ids)).toThrow()
    const short = model.session.tensor([[1, 2]], { dtype: 'i64' })
    expect(() => generate(model, short, { max_new_tokens: 0 })).toThrow()
    expect(() => generate(model, short, { max_new_tokens: 1, forbidden_token_ids: [0, 1, 2, 3, 4, 5, 6, 7] })).toThrow()
    ids.dispose(); short.dispose(); model.dispose()
  })
})
