import { test, expect } from 'std:test'
import { tensor } from 'affon:compute'
import { create_text_generation, generate_text } from '../../src/inference/text.ts'
import { create_handler } from '../../src/serve/http/routes.ts'
import type { InferenceModels } from '../../src/inference/models.ts'
import type { PlaygroundConfig } from '../../src/serve/config.ts'
import { load_llama } from '../../../../packages/@affon/huggingface/src/adapters/llama.ts'

function fixture() {
  let resets = 0, fail = false
  const inputs: number[][] = []
  const model = {
    config: {eos_token_id: 2},
    generate: (ids: number[]) => [...ids, 3, 4, 2],
    create_session() {
      let step = 0
      return {
        reset() { resets++ },
        forward(ids: number[]) {
          inputs.push([...ids])
          if (fail) throw Error('test decode failure')
          const best = [3, 4, 2][step++]
          return {logits: tensor([ids.map(() => Array.from({length:5}, (_, i) => i === best ? 1 : 0))])}
        },
      }
    },
  }
  const processor = {
    encode: () => [1, 0], encode_chat: () => [1, 0],
    decode: (ids: number[]) => ids.includes(4) ? '你好' : ids.includes(3) ? '你�' : '',
  }
  const models = {default_text_model:'smollm2', images:{}, texts:{smollm2:{id:'fixture',chat:true,model,processor}}} as unknown as InferenceModels
  return {models, inputs, resets:()=>resets, fail:()=>{fail=true}}
}

test('incremental decoding preserves byte boundaries, cache inputs, EOS and buffered output', () => {
  const f = fixture(), stream = create_text_generation(f.models, 'Hi', 8)
  expect(stream.next().text).toBe('你')
  expect(stream.next().text).toBe('你好')
  const last = stream.next()
  expect(last.done).toBe(true)
  expect(last.truncated).toBe(false)
  expect(last.text).toBe(generate_text(f.models, 'Hi', 8).text)
  expect(f.inputs).toEqual([[1,0],[3],[4]])
  expect(f.resets()).toBe(1)
  expect(() => stream.next()).toThrow('finished')
})

test('token budget, cancellation and failed decoding release the cache', () => {
  const f = fixture(), bounded = create_text_generation(f.models, 'Hi', 1)
  expect(bounded.next().truncated).toBe(true)
  const cancelled = create_text_generation(f.models, 'Hi', 8)
  cancelled.close()
  expect(() => cancelled.next()).toThrow('finished')
  const failed = create_text_generation(f.models, 'Hi', 8)
  f.fail()
  expect(() => failed.next()).toThrow('test decode failure')
  expect(f.resets()).toBe(3)
})

test('incremental real Llama decoding matches buffered greedy generation', () => {
  const model = load_llama('packages/@affon/huggingface/test/fixtures/llama')
  const processor = {encode:()=>[1,7,12], decode:(ids:number[])=>ids.join(',')}
  const models = {texts:{tiny:{id:'tiny',chat:false,model,processor}},default_text_model:'tiny'} as unknown as InferenceModels
  const stream = create_text_generation(models,'test',4)
  let last = stream.next()
  while (!last.done) last = stream.next()
  expect(last.text).toBe(generate_text(models,'test',4).text)
})

test('HTTP generation reserves the model until finish/cancel and recovers after errors', async () => {
  const f = fixture()
  const handler = create_handler({port:8766,origin:'http://127.0.0.1:8766',device:'cpu'} as PlaygroundConfig, f.models)
  const request = (path:string, value:unknown, origin='http://127.0.0.1:8766') => handler({method:'POST',url:path,headers:{host:'127.0.0.1:8766',origin,'content-type':'application/json'},json:<T>()=>value as T,text:()=>JSON.stringify(value),bytes:()=>new Uint8Array()} satisfies HttpServerRequest)
  const start = () => request('/api/generate/start',{prompt:'Hi',max_new_tokens:8})
  const first = await start(), id = (first.json as {id:string}).id
  expect((await start()).status).toBe(429)
  expect((await request('/api/generate/next',{id:'wrong'})).status).toBe(404)
  expect((await request('/api/generate/cancel',{id},'http://other')).status).toBe(403)
  expect((await request('/api/generate/next',{id})).json).toBeDefined()
  expect((await request('/api/generate/cancel',{id})).status).toBe(200)
  expect((await request('/api/generate/next',{id})).status).toBe(404)
  const second = (await start()).json as {id:string}
  for (let i=0;i<3;i++) await request('/api/generate/next',second)
  const third = (await start()).json as {id:string}
  f.fail()
  expect((await request('/api/generate/next',third)).status).toBe(500)
  const fourth = (await start()).json as {id:string}
  expect(typeof fourth.id).toBe('string')
  await request('/api/generate/cancel',fourth)
})
