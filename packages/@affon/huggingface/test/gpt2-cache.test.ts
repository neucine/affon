import { beforeAll, afterAll, test, expect } from 'std:test'
import { run, getEnv } from 'std:process'
import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { Session } from 'affon:compute'
import type { Tensor, Device } from 'affon:compute'
import { load_gpt2 } from '../src/adapters/gpt2.ts'

let directory = ''
let model: ReturnType<typeof load_gpt2>
let source: Session
beforeAll(async () => {
  directory = (await run({ cmd: 'mktemp', args: ['-d', '/tmp/affon-kv-test.XXXXXX'] })).stdout.trim()
  const config = { model_type: 'gpt2', activation_function: 'gelu_new', n_embd: 8, n_head: 2, n_layer: 2, n_positions: 8, vocab_size: 17, layer_norm_epsilon: 1e-5 }
  fs.writeFileSync(`${directory}/config.json`, JSON.stringify(config))
  source = new Session({ device: 'cpu' })
  const weights: Record<string, Tensor> = {}
  let seed = 0
  function weight(name: string, shape: number[], fill?: number) {
    function values(dims: number[]): any {
      return dims.length ? Array.from({ length: dims[0] }, () => values(dims.slice(1))) : fill ?? Math.sin(++seed) * 0.1
    }
    weights[name] = source.tensor(values(shape)) as Tensor
  }
  weight('transformer.wte.weight', [17, 8]); weight('transformer.wpe.weight', [8, 8])
  for (const prefix of ['transformer.ln_f', ...[0, 1].flatMap(i => [`transformer.h.${i}.ln_1`, `transformer.h.${i}.ln_2`])]) {
    weight(`${prefix}.weight`, [8], 1); weight(`${prefix}.bias`, [8], 0)
  }
  for (let i = 0; i < 2; i++) for (const [name, input, output] of [['attn.c_attn', 8, 24], ['attn.c_proj', 8, 8], ['mlp.c_fc', 8, 32], ['mlp.c_proj', 32, 8]] as const) {
    weight(`transformer.h.${i}.${name}.weight`, [input, output]); weight(`transformer.h.${i}.${name}.bias`, [output], 0)
  }
  checkpoint.save(weights, `${directory}/model.safetensors`)
  model = load_gpt2(directory, (getEnv('AFFON_DEVICE') ?? 'cpu') as Device)
})
afterAll(async () => { model?.dispose(); source?.dispose(); if (directory) await run({ cmd: 'rm', args: ['-rf', directory] }) })
function close(actual: Tensor, expected: Tensor | unknown) {
  const expectedArray = expected && typeof expected === 'object' && 'to_array' in expected ? (expected as Tensor).to_array() : expected
  const a = (actual.to_array() as number[]).flat(Infinity) as number[]
  const b = (expectedArray as number[]).flat(Infinity) as number[]
  expect(a.length).toBe(b.length)
  expect(a.every((x, i) => Number.isFinite(x) && Math.abs(x - b[i]) <= 1e-4 + 1e-4 * Math.abs(b[i]))).toBe(true)
}
test('chunked prefill and single-token decode match full forward, including hidden states', () => {
  const ids = [1, 2, 3, 4, 5, 6, 7, 8], session = model.create_session()
  let offset = 0
  for (const length of [3, 1, 2, 2]) {
    const actual = session.forward(ids.slice(offset, offset + length))
    const expected = model.forward(ids.slice(0, offset + length))
    close(actual.logits, (expected.logits.to_array() as number[][][]).map(batch => batch.slice(offset, offset + length)))
    actual.hidden_states.forEach((value, index) => close(value, (expected.hidden_states[index].to_array() as number[][][]).map(batch => batch.slice(offset, offset + length))))
    offset += length
    expect(session.length).toBe(offset)
  }
  expect(() => session.forward([1])).toThrow()
  expect(session.length).toBe(8)
  session.reset(); expect(session.length).toBe(0)
  close(session.forward([3]).logits, model.forward([3]).logits)
})
test('sessions are independent and invalid steps preserve existing context', () => {
  const a = model.create_session(), b = model.create_session()
  a.forward([1, 2]); b.forward([7])
  for (const ids of [[], [-1], [17], [1.5]]) expect(() => a.forward(ids)).toThrow()
  expect(a.length).toBe(2)
  close(a.forward([3]).logits, (model.forward([1, 2, 3]).logits.to_array() as number[][][]).map(batch => batch.slice(2, 3)))
  close(b.forward([8]).logits, (model.forward([7, 8]).logits.to_array() as number[][][]).map(batch => batch.slice(1, 2)))
})
test('cached greedy generation preserves tokens, zero budget, context limits and EOS', () => {
  const ids = [1, 2]
  expect(model.generate(ids, 6)).toEqual(model.generate(ids, 6, { use_cache: false }))
  expect(model.generate(ids, 0)).toEqual(ids)
  expect(ids).toEqual([1, 2])
  expect(() => model.generate(ids, 7)).toThrow()
  expect(() => model.generate(ids, -1)).toThrow()
  const eos = model.generate(ids, 1)[2]
  model.config.eos_token_id = eos
  try {
    expect(model.generate(ids, 6)).toEqual([...ids, eos])
    expect(model.generate(ids, 6, { use_cache: false })).toEqual([...ids, eos])
  } finally { delete model.config.eos_token_id }
})
