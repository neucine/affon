import fs from 'std:fs'
import { create_olmoe, type OlmoeWeights } from '../../../models/src/olmoe/model.ts'
import { prepare_encoder_checkpoint } from './encoder-checkpoint.ts'

/** Load the original per-expert OLMoE SafeTensors checkpoint layout. */
export function load_olmoe(directory: string) {
  const config = JSON.parse(fs.readFileSync(`${directory}/config.json`))
  if (config.model_type !== 'olmoe' || config.hidden_act !== 'silu' || config.tie_word_embeddings !== false
    || config.attention_bias || (config.attention_dropout ?? 0) !== 0 || config.clip_qkv != null
    || config.rope_scaling != null || config.sliding_window != null || config.norm_topk_prob !== false)
    throw Error('Unsupported OLMoE architecture')
  const { hidden_size: d, intermediate_size: inner, num_attention_heads: heads,
    num_key_value_heads: kvHeads, num_hidden_layers: layers, num_experts: experts,
    num_experts_per_tok: expertsPerToken } = config
  if ([d, inner, heads, kvHeads, layers, experts, expertsPerToken, config.vocab_size, config.max_position_embeddings]
    .some(v => !Number.isInteger(v) || v <= 0) || d % heads || heads % kvHeads || (d / heads) % 2
    || expertsPerToken > experts || !Number.isFinite(config.rms_norm_eps) || config.rms_norm_eps <= 0
    || !Number.isFinite(config.rope_theta) || config.rope_theta <= 0)
    throw Error('Invalid OLMoE dimensions, RoPE or normalization')
  const ops = prepare_encoder_checkpoint(directory), w = ops.weights, require = ops.require_weight
  const kvWidth = d / heads * kvHeads
  require('model.embed_tokens.weight', [config.vocab_size, d])
  require('model.norm.weight', [d])
  require('lm_head.weight', [config.vocab_size, d], true)
  const blocks: OlmoeWeights['blocks'] = []
  for (let i = 0; i < layers; i++) {
    const prefix = `model.layers.${i}`
    require(`${prefix}.input_layernorm.weight`, [d])
    require(`${prefix}.post_attention_layernorm.weight`, [d])
    require(`${prefix}.self_attn.q_norm.weight`, [d])
    require(`${prefix}.self_attn.k_norm.weight`, [kvWidth])
    for (const [name, output, input] of [
      ['self_attn.q_proj', d, d], ['self_attn.k_proj', kvWidth, d], ['self_attn.v_proj', kvWidth, d],
      ['self_attn.o_proj', d, d], ['mlp.gate', experts, d],
    ] as const) require(`${prefix}.${name}.weight`, [output, input], true)
    const expertWeights: OlmoeWeights['blocks'][number]['experts'] = []
    for (let j = 0; j < experts; j++) {
      const expert = `${prefix}.mlp.experts.${j}`
      require(`${expert}.gate_proj.weight`, [inner, d], true)
      require(`${expert}.up_proj.weight`, [inner, d], true)
      require(`${expert}.down_proj.weight`, [d, inner], true)
      expertWeights.push({ gate: w[`${expert}.gate_proj.weight`], up: w[`${expert}.up_proj.weight`], down: w[`${expert}.down_proj.weight`] })
    }
    const weight = (name: string) => w[`${prefix}.${name}.weight`]
    blocks.push({ attentionNorm: weight('input_layernorm'), feedForwardNorm: weight('post_attention_layernorm'),
      queryNorm: weight('self_attn.q_norm'), keyNorm: weight('self_attn.k_norm'),
      query: weight('self_attn.q_proj'), key: weight('self_attn.k_proj'), value: weight('self_attn.v_proj'),
      attentionOutput: weight('self_attn.o_proj'), router: weight('mlp.gate'), experts: expertWeights })
  }
  ops.finish()
  const model = create_olmoe({ width: d, innerWidth: inner, heads, kvHeads, layers, experts, expertsPerToken,
    contextLength: config.max_position_embeddings, vocabSize: config.vocab_size,
    epsilon: config.rms_norm_eps, ropeTheta: config.rope_theta, normalizeTopK: false },
    { tokenEmbedding: w['model.embed_tokens.weight'], finalNorm: w['model.norm.weight'], lmHead: w['lm_head.weight'], blocks })
  return { ...model, config }
}
