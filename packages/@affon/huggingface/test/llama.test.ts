import { expect, test } from 'std:test'
import fs from 'std:fs'
import { getEnv } from 'std:process'
import { load_llama, load_model, format_smollm2_chat } from '../src/index.ts'
import type { Device, Tensor } from 'affon:compute'
const directory = 'packages/@affon/huggingface/test/fixtures/llama'
const reference = JSON.parse(fs.readFileSync(`${directory}/reference.json`))
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
function close(tensor: Tensor, expected: number[][][]) {
  const actual = (tensor.to_array() as number[]).flat(Infinity) as number[], ref = expected.flat(Infinity) as number[]
  expect(actual.length).toBe(ref.length)
  for (let i = 0; i < actual.length; i++) expect(Number.isFinite(actual[i]) && Math.abs(actual[i] - ref[i]) < 1e-5).toBe(true)
}
test('Llama matches independent PyTorch GQA/RoPE hidden states, logits and greedy decoding', () => {
  const model = load_model(directory, {task:'text-generation', device})
  const out = model.forward(reference.ids)
  close(out.logits, reference.logits)
  out.hidden_states.forEach((state, i) => close(state, reference.hidden_states[i]))
  expect(model.generate(reference.ids, 4)).toEqual(reference.generated)
  expect(model.generate(reference.ids, 4, {use_cache:false})).toEqual(reference.generated)
})
test('Llama sessions preserve RoPE positions, offset masks, isolation and state on invalid inputs', () => {
  const model = load_llama(directory, device), a = model.create_session(), b = model.create_session()
  a.forward(reference.ids.slice(0,2))
  close(a.forward(reference.ids.slice(2)).logits, [reference.logits[0].slice(2)])
  expect(a.length).toBe(5); expect(b.length).toBe(0)
  expect(() => a.forward([-1])).toThrow(); expect(a.length).toBe(5)
  expect(() => a.forward(Array(28).fill(1))).toThrow(); expect(a.length).toBe(5)
  a.reset(); expect(a.length).toBe(0)
  close(a.forward(reference.ids).logits, reference.logits)
  expect(model.generate(reference.ids, 0)).toEqual(reference.ids)
  for (const ids of [[], [33], [NaN], [1.5]]) expect(() => model.forward(ids)).toThrow()
  for (const budget of [-1, 0.5, 28]) expect(() => model.generate(reference.ids, budget)).toThrow()
})
test('SmolLM2 formats default and explicit system messages without duplicating the system prompt', () => {
  expect(format_smollm2_chat([{role:'user',content:'Hi'}])).toBe('<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n')
  expect(format_smollm2_chat([{role:'system',content:'Be brief.'},{role:'user',content:'Hi'}], false)).toBe('<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n')
  expect(() => format_smollm2_chat([])).toThrow()
})

test('Llama rejects unsupported variants and malformed dimensions before weight loading', async () => {
  const { run } = await import('std:process')
  const temp = (await run({cmd:'mktemp',args:['-d','/tmp/affon-llama-validation.XXXXXX']})).stdout.trim()
  const config = JSON.parse(fs.readFileSync(`${directory}/config.json`))
  try {
    for (const change of [{tie_word_embeddings:false}, {attention_bias:true}, {mlp_bias:true}, {rope_scaling:{type:'linear',factor:2}}, {rope_interleaved:true}, {num_key_value_heads:3}, {hidden_size:15}, {eos_token_id:33}, {rope_theta:0}, {rms_norm_eps:0}]) {
      fs.writeFileSync(`${temp}/config.json`, JSON.stringify({...config,...change}))
      expect(() => load_llama(temp,device)).toThrow('Llama')
    }
  } finally { await run({cmd:'rm',args:['-rf',temp]}) }
})

test('Llama retained outputs survive subsequent cached and uncached calls', () => {
  const model = load_llama(directory, device)
  const first = model.forward(reference.ids), session = model.create_session()
  const cached = session.forward(reference.ids)
  for (let i = 0; i < 5; i++) session.forward([7])
  model.forward([1,2,3])
  session.reset()
  close(first.logits, reference.logits)
  close(cached.logits, reference.logits)
  first.hidden_states.forEach((state, i) => close(state, reference.hidden_states[i]))
})
