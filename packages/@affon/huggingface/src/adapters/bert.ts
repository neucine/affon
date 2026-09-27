import { create_bert, type BertWeights } from '../../../models/src/bert/index.ts'
import fs from 'std:fs'
import type { Device, Tensor } from 'affon:compute'
import { prepare_encoder_checkpoint } from './encoder-checkpoint.ts'

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
  const ops = prepare_encoder_checkpoint(directory, device)
  const { weights, require_weight: requireWeight } = ops
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
  const { affine } = ops
  const parameters: BertWeights = {
    tokenEmbedding: weights['embeddings.word_embeddings.weight'], positionEmbedding: weights['embeddings.position_embeddings.weight'], typeEmbedding: weights['embeddings.token_type_embeddings.weight'], embeddingNorm: affine('embeddings.LayerNorm'), pooler: affine('pooler.dense'),
    blocks: Array.from({ length: layers }, (_, i) => {
      const p = `encoder.layer.${i}`
      return { query: affine(`${p}.attention.self.query`), key: affine(`${p}.attention.self.key`), value: affine(`${p}.attention.self.value`), attentionOutput: affine(`${p}.attention.output.dense`), attentionNorm: affine(`${p}.attention.output.LayerNorm`), feedForwardNorm: affine(`${p}.output.LayerNorm`), expand: affine(`${p}.intermediate.dense`), contract: affine(`${p}.output.dense`) }
    }),
  }
  const model = create_bert({ width: config.hidden_size, innerWidth: config.intermediate_size, heads: config.num_attention_heads, layers: config.num_hidden_layers, vocabSize: config.vocab_size, contextLength: config.max_position_embeddings, typeVocabSize: config.type_vocab_size, epsilon: config.layer_norm_eps }, parameters, device)
  return { ...model, config }
}
