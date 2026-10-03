import { describe, expect, test } from 'std:test'
import { Session } from 'affon:compute'
import { adam } from 'affon:optim'
import { DecoderModel } from '../src/model.ts'
import { generate } from '../src/causal-lm.ts'

describe('decoder-lm Program model', () => {
  test('combines separate model and loss Programs for optimization', () => {
    const model = DecoderModel(16, 8, { numLayers: 1, numHeads: 2, hiddenDim: 16, causal: true, positional: 'learned', maxSeqLen: 8, tieEmbeddings: true })
    const forward = model.forward(2, 3)
    const loss = model.loss(2, 3)
    const step = model.train(2, 3, adam({ learning_rate: 0.001 }))
    const inspection = forward.inspect()
    expect(inspection.parameters.length).toBe(20)
    expect(inspection.nodes.find(node => node.path.some(segment => segment.program === 'decoder_embeddings'))!.path).toEqual([
      { program: 'decoder_embeddings', instance: 'embeddings' },
    ])
    expect(inspection.nodes.find(node => node.name === 'layer_0.attention_norm.weight')!.path).toEqual([
      { program: 'decoder_block', instance: 'layer_0' },
      { program: 'decoder_norm', instance: 'attention_norm' },
    ])
    expect(inspection.nodes.find(node => node.path.some(segment => segment.program === 'decoder_attention'))!.path).toEqual([
      { program: 'decoder_block', instance: 'layer_0' },
      { program: 'decoder_attention', instance: 'attention' },
    ])
    expect(inspection.nodes[inspection.outputs[0]].path).toEqual([
      { program: 'decoder_tied_head', instance: 'lm_head' },
    ])
    expect(loss.inspect().parameters).toEqual([])
    expect(loss.inspect().arguments.map(value => value.name)).toEqual(['logits', 'labels'])
    expect(step.inspect().transitions[0].parameters.length).toBe(20)

    const session = new Session({ device: 'cpu' })
    const state = session.initialize(step, { seed: 7 })
    const ids = session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'i64' })
    const labels = session.tensor([[2, 3, 4], [5, 6, 7]], { dtype: 'i64' })
    const logits = session.compile(forward).run({ token_ids: ids }, state)
    expect(logits.shape).toEqual([2, 3, 16])
    const before = Object.fromEntries(Object.entries(state.parameters).map(([name, value]) => [name, value.to_array()]))
    const value = session.compile(step).run({ token_ids: ids, labels }, state)
    expect(Number.isFinite(value.item())).toBe(true)
    expect(Object.entries(state.parameters).filter(([name, tensor]) => JSON.stringify(tensor.to_array()) !== JSON.stringify(before[name])).length).toBe(20)
    const generated = generate(model, session, state, ids, { max_new_tokens: 2, forbidden_token_ids: [0] })
    expect(generated.shape).toEqual([2, 5])

    generated.dispose()
    value.dispose()
    logits.dispose()
    labels.dispose()
    ids.dispose()
    state.dispose()
    session.dispose()
  })

  test('validates dimensions, context, and generation options', () => {
    expect(() => DecoderModel(0, 8, { numLayers: 1, numHeads: 2, hiddenDim: 16, maxSeqLen: 8 })).toThrow()
    const model = DecoderModel(8, 4, { numLayers: 1, numHeads: 2, hiddenDim: 8, maxSeqLen: 3 })
    expect(() => model.forward(1, 4)).toThrow()
    const session = new Session()
    const state = session.initialize(model.forward(1, 2))
    const short = session.tensor([[1, 2]], { dtype: 'i64' })
    expect(() => generate(model, session, state, short, { max_new_tokens: 0 })).toThrow()
    expect(() => generate(model, session, state, short, { max_new_tokens: 1, forbidden_token_ids: [0, 1, 2, 3, 4, 5, 6, 7] })).toThrow()
    short.dispose()
    state.dispose()
    session.dispose()
  })
})
