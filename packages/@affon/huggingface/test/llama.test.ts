import { expect, test } from 'std:test'
import fs from 'std:fs'
import { run } from 'std:process'
import { load_llama, load_model, format_smollm2_chat } from '../src/index.ts'
import { execute_causal, generate_causal } from './execute-causal.ts'

const directory = 'packages/@affon/huggingface/test/fixtures/llama'
const reference = JSON.parse(fs.readFileSync(`${directory}/reference.json`))

function close(actual: unknown, expected: unknown) {
  const values = (actual as number[]).flat(Infinity) as number[]
  const refs = (expected as number[]).flat(Infinity) as number[]
  expect(values.length).toBe(refs.length)
  for (let index = 0; index < values.length; index++) expect(Number.isFinite(values[index]) && Math.abs(values[index] - refs[index]) < 1e-5).toBe(true)
}

test('Llama Programs match independent PyTorch GQA/RoPE hidden states, logits and greedy decoding', () => {
  const model = load_model(directory, { task: 'text-generation' })
  const result = execute_causal(model, reference.ids)
  close(result.logits, reference.logits)
  result.hidden_states.forEach((state, index) => close(state, reference.hidden_states[index]))
  expect(generate_causal(model, reference.ids, 4)).toEqual(reference.generated)
})

test('Llama output windows expose decode-shaped results without a fake cache', () => {
  const model = load_llama(directory)
  const suffix = execute_causal(model, reference.ids, 2)
  close(suffix.logits, [reference.logits[0].slice(2)])
  suffix.hidden_states.forEach((state, index) => close(state, [reference.hidden_states[index][0].slice(2)]))
  expect(generate_causal(model, reference.ids, 0)).toEqual(reference.ids)
  expect(() => model.forward(0)).toThrow()
  expect(() => model.forward(model.config.max_position_embeddings + 1)).toThrow()
  for (const budget of [-1, 0.5, 28]) expect(() => generate_causal(model, reference.ids, budget)).toThrow()
})

test('SmolLM2 formats default and explicit system messages without duplicating the system prompt', () => {
  expect(format_smollm2_chat([{role:'user',content:'Hi'}])).toBe('<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n')
  expect(format_smollm2_chat([{role:'system',content:'Be brief.'},{role:'user',content:'Hi'}], false)).toBe('<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n')
  expect(() => format_smollm2_chat([])).toThrow()
})

test('Llama rejects unsupported variants and malformed dimensions before weight loading', async () => {
  const temp = (await run({cmd:'mktemp',args:['-d','/tmp/affon-llama-validation.XXXXXX']})).stdout.trim()
  const config = JSON.parse(fs.readFileSync(`${directory}/config.json`))
  try {
    for (const change of [{tie_word_embeddings:false}, {attention_bias:true}, {mlp_bias:true}, {rope_scaling:{type:'linear',factor:2}}, {rope_interleaved:true}, {num_key_value_heads:3}, {hidden_size:15}, {eos_token_id:33}, {rope_theta:0}, {rms_norm_eps:0}]) {
      fs.writeFileSync(`${temp}/config.json`, JSON.stringify({...config,...change}))
      expect(() => load_llama(temp)).toThrow('Llama')
    }
  } finally { await run({cmd:'rm',args:['-rf',temp]}) }
})
