import { test, expect } from 'std:test'
import { generate_text, dispose_text_runtimes } from '../../src/inference/text.ts'
import type { InferenceModels } from '../../src/inference/models.ts'
import { Tensor, program } from 'affon:compute'
import { contiguous, div, embedding, reshape, slice } from 'affon:ops'
function fixture() {
  let chatCalls = 0, programCalls = 0, truncate = false
  const processor = { encode: () => [5], encode_chat: () => {chatCalls++; return [1,5]}, decode: (ids: number[], opts?: {skipSpecialTokens?: boolean}) => (opts?.skipSpecialTokens ? ids.filter(id => id !== 2) : ids).join(',') }
  const model = {
    config: {eos_token_id: 2, vocab_size: 10, max_position_embeddings: 10}, parameters: {},
    forward(length: number, outputStart = 0) {
      programCalls++
      return program('fixture_text', p => {
        const ids = p.argument('ids', Tensor.i64([length]))
        p.argument('mask', Tensor.i64([1, 1, length, length]))
        const transitions = Array.from({length: 10}, (_, token) => Array.from({length: 10}, (_, candidate) => candidate === (token === 5 ? 8 : token === 8 ? (truncate ? 9 : 2) : 0) ? 1 : 0))
        const logits = reshape(embedding(p.constant('transitions', transitions, Tensor.f32([10, 10])), ids), [1, length, 10])
        return [contiguous(slice(logits, [{start: 0, stop: 1}, {start: outputStart, stop: length}, {start: 0, stop: 10}]))]
      })
    },
  }
  const models = {device:'cpu',default_text_model:'distilgpt2',texts:{distilgpt2:{id:'gpt',chat:false,processor,model},smollm2:{id:'smol',chat:true,processor,model}}} as unknown as InferenceModels
  return {models, processor, model, chatCalls:()=>chatCalls, programCalls:()=>programCalls, truncate:()=>{truncate=true}}
}
test('text requests reuse model state and release it explicitly', () => {
  const f = fixture()
  generate_text(f.models, 'Hi', 1)
  const first = f.programCalls()
  generate_text(f.models, 'Hi', 1)
  expect(f.programCalls() - first).toBe(1)
  dispose_text_runtimes(f.models)
  generate_text(f.models, 'Hi', 1)
  expect(f.programCalls() - first).toBe(3)
  dispose_text_runtimes(f.models)
})
test('non-finite logits fail rather than silently selecting token zero', () => {
  const f = fixture()
  f.model.forward = (length: number) => program('invalid_logits', p => {
    p.argument('ids', Tensor.i64([length]))
    p.argument('mask', Tensor.i64([1, 1, length, length]))
    const zeros = p.constant('zeros', [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], Tensor.f32([1, 1, 10]))
    return [div(zeros, zeros)]
  })
  expect(() => generate_text(f.models, 'Hi', 1)).toThrow('Non-finite model logit')
  dispose_text_runtimes(f.models)
})
test('selected text model uses chat formatting and returns only decoded assistant text', () => {
  const f = fixture()
  const chat = generate_text(f.models,'Hi',2,'smollm2')
  expect(chat.text).toBe('8'); expect(chat.model).toBe('smol'); expect(chat.prompt_tokens).toBe(2)
  expect(chat.truncated).toBe(false); expect(f.chatCalls()).toBe(1)
  expect(generate_text(f.models,'Hi',2).text).toBe('5,8,2')
  expect(f.chatCalls()).toBe(1)
  f.models.default_text_model = 'smollm2'
  expect(generate_text(f.models,'Hi',2).text).toBe('8')
  expect(f.chatCalls()).toBe(2)
  dispose_text_runtimes(f.models)
})
test('text generation rejects unknown models, invalid budgets and oversized formatted prompts', () => {
  const f = fixture()
  for (const key of ['nope','__proto__','constructor']) expect(() => generate_text(f.models,'Hi',2,key)).toThrow()
  for (const budget of [0,65,NaN,1.5]) expect(() => generate_text(f.models,'Hi',budget)).toThrow()
  f.processor.encode_chat = () => Array(257).fill(1)
  expect(() => generate_text(f.models,'Hi',2,'smollm2')).toThrow()
  f.truncate()
  expect(generate_text(f.models,'Hi',2).truncated).toBe(true)
  dispose_text_runtimes(f.models)
})
