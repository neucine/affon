import { beforeAll, afterAll, test, expect } from 'std:test'
import { run } from 'std:process'
import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { Session, type Tensor } from 'affon:compute'
import { load_gpt2 } from '../src/adapters/gpt2.ts'
import { execute_causal, generate_causal } from './execute-causal.ts'

let directory = ''
let model: ReturnType<typeof load_gpt2>
let source: Session
beforeAll(async () => {
  directory = (await run({ cmd: 'mktemp', args: ['-d', '/tmp/affon-program-window-test.XXXXXX'] })).stdout.trim()
  const config = { model_type: 'gpt2', activation_function: 'gelu_new', n_embd: 8, n_head: 2, n_layer: 2, n_positions: 8, vocab_size: 17, layer_norm_epsilon: 1e-5 }
  fs.writeFileSync(`${directory}/config.json`, JSON.stringify(config))
  source = new Session({ device: 'cpu' })
  const weights: Record<string, Tensor> = {}
  let seed = 0
  function weight(name: string, shape: number[], fill?: number) {
    function values(dims: number[]): any { return dims.length ? Array.from({ length: dims[0] }, () => values(dims.slice(1))) : fill ?? Math.sin(++seed) * 0.1 }
    weights[name] = source.tensor(values(shape))
  }
  weight('transformer.wte.weight', [17, 8]); weight('transformer.wpe.weight', [8, 8])
  for (const prefix of ['transformer.ln_f', ...[0, 1].flatMap(i => [`transformer.h.${i}.ln_1`, `transformer.h.${i}.ln_2`])]) {
    weight(`${prefix}.weight`, [8], 1); weight(`${prefix}.bias`, [8], 0)
  }
  for (let i = 0; i < 2; i++) for (const [name, input, output] of [['attn.c_attn', 8, 24], ['attn.c_proj', 8, 8], ['mlp.c_fc', 8, 32], ['mlp.c_proj', 32, 8]] as const) {
    weight(`transformer.h.${i}.${name}.weight`, [input, output]); weight(`transformer.h.${i}.${name}.bias`, [output], 0)
  }
  checkpoint.save(weights, `${directory}/model.safetensors`)
  model = load_gpt2(directory)
})
afterAll(async () => { source?.dispose(); if (directory) await run({ cmd: 'rm', args: ['-rf', directory] }) })

function close(actual: unknown, expected: unknown) {
  const a = (actual as number[]).flat(Infinity) as number[]
  const b = (expected as number[]).flat(Infinity) as number[]
  expect(a.length).toBe(b.length)
  expect(a.every((value, index) => Number.isFinite(value) && Math.abs(value - b[index]) <= 1e-4 + 1e-4 * Math.abs(b[index]))).toBe(true)
}

test('explicit GPT-2 output windows match slices of full-prefix execution', () => {
  const ids = [1, 2, 3, 4, 5, 6, 7, 8]
  let offset = 0
  for (const length of [3, 1, 2, 2]) {
    const prefix = ids.slice(0, offset + length)
    const actual = execute_causal(model, prefix, offset)
    const expected = execute_causal(model, prefix)
    close(actual.logits, (expected.logits as number[][][]).map(batch => batch.slice(offset)))
    actual.hidden_states.forEach((value, index) => close(value, (expected.hidden_states[index] as number[][][]).map(batch => batch.slice(offset))))
    offset += length
  }
})

test('caller-authored greedy generation preserves tokens, limits and EOS', () => {
  const ids = [1, 2]
  expect(generate_causal(model, ids, 0)).toEqual(ids)
  expect(ids).toEqual([1, 2])
  expect(() => generate_causal(model, ids, 7)).toThrow()
  expect(() => generate_causal(model, ids, -1)).toThrow()
  const eos = generate_causal(model, ids, 1)[2]
  model.config.eos_token_id = eos
  try { expect(generate_causal(model, ids, 6)).toEqual([...ids, eos]) }
  finally { delete model.config.eos_token_id }
})
