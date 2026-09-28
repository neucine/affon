import { test, expect } from 'std:test'
import { generate_text } from '../../src/inference/text.ts'
import type { InferenceModels } from '../../src/inference/models.ts'
function fixture() {
  let chatCalls = 0
  const processor = { encode: () => [5], encode_chat: () => {chatCalls++; return [1,5]}, decode: (ids: number[], opts?: {skipSpecialTokens?: boolean}) => (opts?.skipSpecialTokens ? ids.filter(id => id !== 2) : ids).join(',') }
  const model = { config:{eos_token_id:2}, generate:(ids:number[]) => [...ids,8,2] }
  const models = {texts:{distilgpt2:{id:'gpt',chat:false,processor,model},smollm2:{id:'smol',chat:true,processor,model}}} as unknown as InferenceModels
  return {models, processor, model, chatCalls:()=>chatCalls}
}
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
})
test('text generation rejects unknown models, invalid budgets and oversized formatted prompts', () => {
  const f = fixture()
  for (const key of ['nope','__proto__','constructor']) expect(() => generate_text(f.models,'Hi',2,key)).toThrow()
  for (const budget of [0,65,NaN,1.5]) expect(() => generate_text(f.models,'Hi',budget)).toThrow()
  f.processor.encode_chat = () => Array(257).fill(1)
  expect(() => generate_text(f.models,'Hi',2,'smollm2')).toThrow()
  f.model.generate = ids => [...ids,8,9]
  expect(generate_text(f.models,'Hi',2).truncated).toBe(true)
})
