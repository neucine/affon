import { describe, expect, test } from 'std:test'
import { Session, Tensor, optimize, program } from 'affon:compute'
import { cross_entropy } from 'affon:nn'
import { adam } from 'affon:optim'
import { DecoderModel } from '../src/model.ts'
import { generate } from '../src/causal-lm.ts'

describe('decoder-lm Program model', () => {
  test('combines separate model and loss Programs for optimization', () => {
    const model = DecoderModel(16, 8, { numLayers: 1, numHeads: 2, hiddenDim: 16, causal: true, positional: 'learned', maxSeqLen: 8, tieEmbeddings: true })
    const modelProgram = program('decoder_lm', p => model({
      token_ids: p.argument('token_ids', Tensor.i64([2, 3], { axes: ['batch', 'token'] })),
    }, 'decoder'))
    const objective = cross_entropy()
    const loss = program('decoder_loss', p => objective({
      input: p.argument('logits', Tensor.f32([2, 3, 16])),
      target: p.argument('labels', Tensor.i64([2, 3], { axes: ['batch', 'token'] })),
    }, 'objective'))
    const step = optimize(modelProgram, objective, adam({ learning_rate: 0.001 }))
    const inspection = modelProgram.inspect()
    expect(inspection.parameters.length).toBe(20)
    expect(inspection.parameters.slice(0, 2).map(value => value.name)).toEqual([
      'decoder.token_embedding.weight',
      'decoder.position_embedding.weight',
    ])
    expect(inspection.nodes.find(node => node.name === 'decoder.blocks.0.attention.norm.weight')!.path).toEqual([
      { program: 'decoder_model', instance: 'decoder' },
      { program: 'decoder_block', instance: 'blocks.0' },
    ])
    expect(inspection.nodes.find(node => node.path.some(segment => segment.program === 'decoder_attention'))!.path).toEqual([
      { program: 'decoder_model', instance: 'decoder' },
      { program: 'decoder_block', instance: 'blocks.0' },
      { program: 'decoder_attention', instance: 'attention' },
    ])
    expect(inspection.nodes[inspection.outputs[0]].path).toEqual([
      { program: 'decoder_model', instance: 'decoder' },
      { program: 'decoder_tied_head', instance: 'lm_head' },
    ])
    expect(loss.inspect().parameters).toEqual([])
    expect(loss.inspect().arguments.map(value => value.name)).toEqual(['logits', 'labels'])
    expect(step.inspect().transitions[0].parameters.length).toBe(20)

    const session = new Session({ device: 'cpu' })
    const state = session.initialize(step, { seed: 7 })
    const ids = session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'i64' })
    const labels = session.tensor([[2, 3, 4], [5, 6, 7]], { dtype: 'i64' })
    const logits = session.compile(modelProgram).run({ token_ids: ids }, state)
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
    expect(() => program('too_long', p => model({ token_ids: p.argument('token_ids', Tensor.i64([1, 4])) }))).toThrow()
    const session = new Session()
    const source = program('decoder_lm', p => model({
      token_ids: p.argument('token_ids', Tensor.i64([1, 2], { axes: ['batch', 'token'] })),
    }, 'decoder'))
    const state = session.initialize(source)
    const short = session.tensor([[1, 2]], { dtype: 'i64' })
    expect(() => generate(model, session, state, short, { max_new_tokens: 0 })).toThrow()
    expect(() => generate(model, session, state, short, { max_new_tokens: 1, forbidden_token_ids: [0, 1, 2, 3, 4, 5, 6, 7] })).toThrow()
    short.dispose()
    state.dispose()
    session.dispose()
  })
})
