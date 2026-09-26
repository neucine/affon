import fs from 'std:fs'
import { add, contiguous, index_select, mul, no_grad, reshape, sqrt, sum, tanh, tensor, div } from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'
import { encoder_ops, erf_gelu } from './encoder-ops.ts'

/** Load an f32 absolute-position BERT base encoder with its pooler.
 * @param directory Prepared local HF artifact directory.
 * @param device Execution device.
 * @returns Forward over rectangular ID, mask, and type-ID batches; includes hidden states and pooling.
 */
export function load_bert(directory: string, device: Device) {
  const config = JSON.parse(fs.readFileSync(`${directory}/config.json`))
  if (config.model_type !== 'bert' || config.hidden_act !== 'gelu' || config.is_decoder
    || config.add_cross_attention || (config.position_embedding_type ?? 'absolute') !== 'absolute') {
    throw new Error('HF adapter supports absolute-position BERT encoders with GELU')
  }
  const { hidden_size: d, intermediate_size: inner, num_attention_heads: heads, num_hidden_layers: layers } = config
  for (const value of [d, inner, heads, layers, config.vocab_size, config.max_position_embeddings, config.type_vocab_size]) {
    if (!Number.isInteger(value) || value <= 0) throw new Error('Invalid BERT dimension')
  }
  if (d % heads || !Number.isFinite(config.layer_norm_eps) || config.layer_norm_eps <= 0) throw new Error('Invalid BERT head/norm configuration')
  const ops = encoder_ops(directory, device)
  const { weights, require_weight: requireWeight, dense, norm, attention } = ops
  const requireNorm = (prefix: string) => {
    requireWeight(`${prefix}.weight`, [d]); requireWeight(`${prefix}.bias`, [d])
  }
  const requireDense = (prefix: string, input: number, output: number) => {
    requireWeight(`${prefix}.weight`, [output, input], true); requireWeight(`${prefix}.bias`, [output])
  }
  requireWeight('embeddings.word_embeddings.weight', [config.vocab_size, d])
  requireWeight('embeddings.position_embeddings.weight', [config.max_position_embeddings, d])
  requireWeight('embeddings.token_type_embeddings.weight', [config.type_vocab_size, d])
  requireNorm('embeddings.LayerNorm')
  for (let i = 0; i < layers; i++) {
    const p = `encoder.layer.${i}`
    for (const name of ['query', 'key', 'value']) requireDense(`${p}.attention.self.${name}`, d, d)
    requireDense(`${p}.attention.output.dense`, d, d)
    requireNorm(`${p}.attention.output.LayerNorm`)
    requireDense(`${p}.intermediate.dense`, d, inner)
    requireDense(`${p}.output.dense`, inner, d)
    requireNorm(`${p}.output.LayerNorm`)
  }
  requireDense('pooler.dense', d, d)
  ops.finish()
  function forward(ids: number[][], masks: number[][], types: number[][]) {
    const batch = ids.length, length = ids[0]?.length ?? 0
    if (!batch || !length || length > config.max_position_embeddings || masks.length !== batch || types.length !== batch) throw new Error('Invalid BERT batch')
    for (let b = 0; b < batch; b++) {
      for (const [row, limit] of [[ids[b], config.vocab_size], [types[b], config.type_vocab_size], [masks[b], 2]] as [number[], number][]) {
        if (row.length !== length || row.some(value => !Number.isInteger(value) || value < 0 || value >= limit)) throw new Error('Invalid BERT input row')
      }
      if (!masks[b].some(value => value === 1)) throw new Error('All-masked BERT inputs are not supported')
    }
    return no_grad(() => {
      const embedding = (name: string, values: number[]) => reshape(index_select(weights[name], 0, tensor(values, { dtype: 'f32', device })), [batch, length, d])
      const positions = Array.from({ length: batch * length }, (_, i) => i % length)
      let x = norm(add(add(embedding('embeddings.word_embeddings.weight', ids.flat()),
        embedding('embeddings.position_embeddings.weight', positions)),
        embedding('embeddings.token_type_embeddings.weight', types.flat())), 'embeddings.LayerNorm', config.layer_norm_eps)
      const hidden_states: Tensor[] = [x]
      const mask = reshape(tensor(masks.map(row => row.map(value => 1 - value)), { dtype: 'f32', device }), [batch, 1, 1, length])
      for (let i = 0; i < layers; i++) {
        const p = `encoder.layer.${i}`
        const context = attention(dense(x, `${p}.attention.self.query`), dense(x, `${p}.attention.self.key`), dense(x, `${p}.attention.self.value`), heads, mask)
        x = norm(add(x, dense(context, `${p}.attention.output.dense`)), `${p}.attention.output.LayerNorm`, config.layer_norm_eps)
        x = norm(add(x, dense(erf_gelu(dense(x, `${p}.intermediate.dense`)), `${p}.output.dense`)), `${p}.output.LayerNorm`, config.layer_norm_eps)
        hidden_states.push(x)
      }
      const valid = reshape(tensor(masks, { dtype: 'f32', device }), [batch, length, 1])
      const mean = div(sum(mul(x, valid), 1), sum(valid, 1))
      const pooled = div(mean, sqrt(sum(mul(mean, mean), 1, true)))
      const pooler = tanh(dense(reshape(contiguous(x.slice([':', '0:1', ':'])), [batch, d]), 'pooler.dense'))
      return { output: x, pooled, pooler, hidden_states }
    })
  }
  return { config, forward }
}
