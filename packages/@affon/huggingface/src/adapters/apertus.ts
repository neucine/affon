import fs from 'std:fs'
import { create_apertus, type ApertusWeights } from '../../../models/src/apertus/model.ts'
import { prepare_encoder_checkpoint } from './encoder-checkpoint.ts'

/** Apertus-8B-2509 checkpoint mapping; preserve published BF16 parameter storage. */
export function load_apertus(directory: string) {
  const config = JSON.parse(fs.readFileSync(`${directory}/config.json`))
  const rope = config.rope_scaling
  if (config.model_type !== 'apertus' || config.hidden_act !== 'xielu' || config.tie_word_embeddings !== false
    || config.attention_bias || config.mlp_bias || config.qk_norm !== true || config.post_norm
    || config.rope_interleaved || config.sliding_window != null || (config.attention_dropout ?? 0) !== 0
    || (config.hidden_dropout ?? 0) !== 0 || !rope || rope.rope_type !== 'llama3') throw Error('Unsupported Apertus architecture')
  const {hidden_size:d,intermediate_size:inner,num_attention_heads:heads,num_key_value_heads:kvHeads,num_hidden_layers:layers} = config
  const headWidth=d/heads, kvWidth=headWidth*kvHeads
  if ([d,inner,heads,kvHeads,layers,config.vocab_size,config.max_position_embeddings,rope.original_max_position_embeddings]
      .some(v=>!Number.isInteger(v)||v<=0) || d%heads || heads%kvHeads || headWidth%2
    || !Number.isFinite(config.rms_norm_eps) || config.rms_norm_eps<=0
    || !Number.isFinite(config.rope_theta) || config.rope_theta<=0
    || !Number.isFinite(rope.factor) || rope.factor<=0
    || !Number.isFinite(rope.low_freq_factor) || rope.low_freq_factor<=0
    || !Number.isFinite(rope.high_freq_factor) || rope.high_freq_factor<=rope.low_freq_factor
    || !Number.isInteger(config.eos_token_id) || config.eos_token_id<0 || config.eos_token_id>=config.vocab_size)
    throw Error('Invalid Apertus dimensions or RoPE configuration')
  const ops=prepare_encoder_checkpoint(directory), w=ops.weights
  const require=ops.require_weight
  require('model.embed_tokens.weight',[config.vocab_size,d])
  require('model.norm.weight',[d])
  require('lm_head.weight',[config.vocab_size,d])
  const blocks:ApertusWeights['blocks']=[]
  for(let i=0;i<layers;i++){
    const p=`model.layers.${i}`
    for(const name of ['attention_layernorm','feedforward_layernorm']) require(`${p}.${name}.weight`,[d])
    for(const name of ['q_norm','k_norm']) require(`${p}.self_attn.${name}.weight`,[headWidth])
    for(const [name,output,input] of [
      ['self_attn.q_proj',d,d],['self_attn.k_proj',kvWidth,d],['self_attn.v_proj',kvWidth,d],['self_attn.o_proj',d,d],
      ['mlp.up_proj',inner,d],['mlp.down_proj',d,inner],
    ] as const) require(`${p}.${name}.weight`,[output,input])
    for(const name of ['alpha_p','alpha_n']) require(`${p}.mlp.act_fn.${name}`,[1])
    for(const name of ['beta','eps']) require(`${p}.mlp.act_fn.${name}`,[])
    const weight=(name:string)=>w[`${p}.${name}.weight`]
    const activation=(name:string)=>w[`${p}.mlp.act_fn.${name}`]
    blocks.push({attentionNorm:weight('attention_layernorm'),feedForwardNorm:weight('feedforward_layernorm'),
      queryNorm:weight('self_attn.q_norm'),keyNorm:weight('self_attn.k_norm'),
      query:weight('self_attn.q_proj'),key:weight('self_attn.k_proj'),value:weight('self_attn.v_proj'),
      attentionOutput:weight('self_attn.o_proj'),up:weight('mlp.up_proj'),down:weight('mlp.down_proj'),
      alphaP:activation('alpha_p'),alphaN:activation('alpha_n'),beta:activation('beta'),activationEpsilon:activation('eps')})
  }
  ops.finish()
  const model=create_apertus({width:d,innerWidth:inner,heads,kvHeads,layers,contextLength:config.max_position_embeddings,
    vocabSize:config.vocab_size,epsilon:config.rms_norm_eps,ropeTheta:config.rope_theta,eosTokenId:config.eos_token_id,
    ropeScaling:{factor:rope.factor,lowFreqFactor:rope.low_freq_factor,highFreqFactor:rope.high_freq_factor,
      originalContext:rope.original_max_position_embeddings}},
    {tokenEmbedding:w['model.embed_tokens.weight'],finalNorm:w['model.norm.weight'],lmHead:w['lm_head.weight'],blocks})
  return {...model,config}
}
