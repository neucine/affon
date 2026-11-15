import { afterAll, beforeAll, expect, test } from 'std:test'
import { run } from 'std:process'
import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { Session, type Tensor } from 'affon:core'
import { load_model, load_olmoe } from '../src/index.ts'
import { execute_causal } from './execute-causal.ts'

let directory = ''
let session: Session
beforeAll(async () => {
  directory = (await run({ cmd: 'mktemp', args: ['-d', '/tmp/affon-olmoe-test.XXXXXX'] })).stdout.trim()
  fs.writeFileSync(`${directory}/config.json`, JSON.stringify({ model_type: 'olmoe', hidden_act: 'silu',
    tie_word_embeddings: false, attention_bias: false, attention_dropout: 0, norm_topk_prob: false,
    hidden_size: 4, intermediate_size: 4, num_attention_heads: 1, num_key_value_heads: 1,
    num_hidden_layers: 1, num_experts: 2, num_experts_per_tok: 1,
    vocab_size: 4, max_position_embeddings: 8, rms_norm_eps: 1e-5, rope_theta: 10000 }))
  session = new Session({ device: 'cpu' })
  const weights: Record<string, Tensor> = {}
  const matrix = (diagonal: number) => Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => i === j ? diagonal : 0))
  const weight = (name: string, values: number[] | number[][]) => { weights[name] = session.tensor(values) }
  const p = 'model.layers.0'
  weight('model.embed_tokens.weight', matrix(1)); weight('model.norm.weight', [1, 1, 1, 1]); weight('lm_head.weight', matrix(1))
  weight(`${p}.input_layernorm.weight`, [1, 1, 1, 1]); weight(`${p}.post_attention_layernorm.weight`, [1, 1, 1, 1])
  weight(`${p}.self_attn.q_norm.weight`, [1, 1, 1, 1]); weight(`${p}.self_attn.k_norm.weight`, [1, 1, 1, 1])
  for (const name of ['q_proj', 'k_proj', 'v_proj', 'o_proj']) weight(`${p}.self_attn.${name}.weight`, matrix(0))
  weight(`${p}.mlp.gate.weight`, [[1, 0, 0, 0], [-1, 0, 0, 0]])
  for (let i = 0; i < 2; i++) for (const name of ['gate_proj', 'up_proj', 'down_proj'])
    weight(`${p}.mlp.experts.${i}.${name}.weight`, i === 0 && name !== 'down_proj'
      ? [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] : matrix(i === 0 ? 1 : 0))
  checkpoint.save(weights, `${directory}/model.safetensors`)
})
afterAll(async () => { session?.dispose(); if (directory) await run({ cmd: 'rm', args: ['-rf', directory] }) })

test('OLMoE maps separate expert weights and executes top-1 routing', () => {
  const model = load_model(directory, { task: 'text-generation' })
  expect(model.parameters.block_0_expert_0_gate.shape).toEqual([4, 4])
  const result = execute_causal(model, [0])
  const actual = (result.logits as number[][][])[0][0]
  const x = 1 / Math.sqrt(0.25 + 1e-5)
  const routed = Math.exp(x) / (Math.exp(x) + Math.exp(-x))
  const second = routed * (x / (1 + Math.exp(-x))) * x
  const denominator = Math.sqrt((1 + second * second) / 4 + 1e-5)
  expect(Math.abs(actual[0] - 1 / denominator) < 1e-4).toBe(true)
  expect(Math.abs(actual[1] - second / denominator) < 1e-4).toBe(true)
  expect(actual.slice(2).every(value => Math.abs(value) < 1e-6)).toBe(true)
  expect(() => load_olmoe(directory).forward(9)).toThrow()
})
