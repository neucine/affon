import fs from 'std:fs'
import { create_llama, type LlamaWeights } from '../../../models/src/llama/index.ts'
import { prepare_encoder_checkpoint } from './encoder-checkpoint.ts'

/** Load the tied-head, bias-free, unscaled-RoPE Llama variant used by SmolLM2.
 * BF16 checkpoint storage is widened to f32 by checkpoint.load; execution is f32.
 */
export function load_llama(directory: string) {
  const config = JSON.parse(fs.readFileSync(`${directory}/config.json`))
  if (config.model_type !== 'llama' || config.hidden_act !== 'silu' || config.tie_word_embeddings !== true
    || config.attention_bias || config.mlp_bias || config.rope_scaling != null || config.rope_interleaved
    || (config.pretraining_tp ?? 1) !== 1 || config.cross_attention || config.add_cross_attention
    || (config.partial_rotary_factor ?? 1) !== 1 || config.sliding_window != null
    || (config.head_dim ?? config.hidden_size / config.num_attention_heads) !== config.hidden_size / config.num_attention_heads)
    throw Error('HF adapter supports tied-head bias-free Llama with full, non-interleaved, unscaled RoPE')
  const { hidden_size: d, intermediate_size: inner, num_attention_heads: heads, num_key_value_heads: kvHeads, num_hidden_layers: layers } = config
  if ([d, inner, heads, kvHeads, layers, config.vocab_size, config.max_position_embeddings].some(v => !Number.isInteger(v) || v <= 0)
    || d % heads || heads % kvHeads || (d / heads) % 2
    || !Number.isFinite(config.rms_norm_eps) || config.rms_norm_eps <= 0
    || !Number.isFinite(config.rope_theta) || config.rope_theta <= 0
    || !Number.isInteger(config.eos_token_id) || config.eos_token_id < 0 || config.eos_token_id >= config.vocab_size)
    throw Error('Invalid Llama dimensions, RoPE, normalization or EOS token')
  const ops = prepare_encoder_checkpoint(directory)
  const require = ops.require_weight, w = ops.weights, kvWidth = d / heads * kvHeads
  require('model.embed_tokens.weight', [config.vocab_size, d]); require('model.norm.weight', [d])
  const blocks: LlamaWeights['blocks'] = []
  for (let i = 0; i < layers; i++) {
    const p = `model.layers.${i}`
    require(`${p}.input_layernorm.weight`, [d]); require(`${p}.post_attention_layernorm.weight`, [d])
    for (const [name, input, output] of [
      ['self_attn.q_proj', d, d], ['self_attn.k_proj', d, kvWidth], ['self_attn.v_proj', d, kvWidth], ['self_attn.o_proj', d, d],
      ['mlp.gate_proj', d, inner], ['mlp.up_proj', d, inner], ['mlp.down_proj', inner, d],
    ] as const) require(`${p}.${name}.weight`, [output, input], true)
    const weight = (name: string) => w[`${p}.${name}.weight`]
    blocks.push({ attentionNorm: weight('input_layernorm'), feedForwardNorm: weight('post_attention_layernorm'), query: weight('self_attn.q_proj'), key: weight('self_attn.k_proj'), value: weight('self_attn.v_proj'), attentionOutput: weight('self_attn.o_proj'), gate: weight('mlp.gate_proj'), up: weight('mlp.up_proj'), down: weight('mlp.down_proj') })
  }
  ops.finish()
  const model = create_llama({ width: d, innerWidth: inner, heads, kvHeads, layers, contextLength: config.max_position_embeddings, vocabSize: config.vocab_size, epsilon: config.rms_norm_eps, ropeTheta: config.rope_theta, eosTokenId: config.eos_token_id }, { tokenEmbedding: w['model.embed_tokens.weight'], finalNorm: w['model.norm.weight'], blocks })
  return { ...model, config }
}
