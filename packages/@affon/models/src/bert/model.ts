import { positive_dimensions, parameter_checks } from '../shared/parameters.ts'
import { add, contiguous, index_select, mul, no_grad, reshape, sqrt, sum, tanh, tensor, div } from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'
import { encoder_ops, erf_gelu } from '../shared/encoder.ts'

import type { AffineWeights } from '../shared/encoder.ts'

export interface BertConfig { width: number; innerWidth: number; heads: number; layers: number; vocabSize: number; contextLength: number; typeVocabSize: number; epsilon: number }
export interface BertWeights {
  blocks: { query: AffineWeights; key: AffineWeights; value: AffineWeights; attentionOutput: AffineWeights; attentionNorm: AffineWeights; feedForwardNorm: AffineWeights; expand: AffineWeights; contract: AffineWeights }[]
  tokenEmbedding: Tensor; positionEmbedding: Tensor; typeEmbedding: Tensor; embeddingNorm: AffineWeights; pooler: AffineWeights
}

/** Construct f32 Bert from normalized configuration and prepared tensors.
 * Affine weights use [input, output] layout; normalization uses vectors.
 * Tensors must be on the requested device and remain alive with the model.
 */
export function create_bert(config: BertConfig, weights: BertWeights, device: Device = 'cpu') {
  if (!Number.isInteger(config.width) || config.width <= 0 || !Number.isInteger(config.heads) || config.heads <= 0 || config.width % config.heads || !Number.isInteger(config.layers) || config.layers <= 0 || weights.blocks.length !== config.layers || !Number.isFinite(config.epsilon) || config.epsilon <= 0) throw new Error('Invalid model dimensions or normalization')
  const check = parameter_checks(device)
  const width = config.width
  positive_dimensions(config.innerWidth)
  for (const block of weights.blocks) {
    for (const params of [block.query, block.key, block.value, block.attentionOutput]) check.affine(params, [width, width])
    check.affine(block.attentionNorm, [width]); check.affine(block.feedForwardNorm, [width])
    check.affine(block.expand, [width, config.innerWidth]); check.affine(block.contract, [config.innerWidth, width])
  }
  positive_dimensions(config.vocabSize, config.contextLength, config.typeVocabSize)
  check.tensor(weights.tokenEmbedding, [config.vocabSize, width])
  check.tensor(weights.positionEmbedding, [config.contextLength, width])
  check.tensor(weights.typeEmbedding, [config.typeVocabSize, width])
  check.affine(weights.embeddingNorm, [width]); check.affine(weights.pooler, [width, width])
  const { width: d, heads, layers } = config
  const { dense, norm, attention } = encoder_ops(device)
  function forward(ids: number[][], masks: number[][], types: number[][]) {
    const batch = ids.length, length = ids[0]?.length ?? 0
    if (!batch || !length || length > config.contextLength || masks.length !== batch || types.length !== batch) throw new Error('Invalid BERT batch')
    for (let b = 0; b < batch; b++) {
      for (const [row, limit] of [[ids[b], config.vocabSize], [types[b], config.typeVocabSize], [masks[b], 2]] as [number[], number][]) {
        if (row.length !== length || row.some(value => !Number.isInteger(value) || value < 0 || value >= limit)) throw new Error('Invalid BERT input row')
      }
      if (!masks[b].some(value => value === 1)) throw new Error('All-masked BERT inputs are not supported')
    }
    return no_grad(() => {
      const embedding = (weight: Tensor, values: number[]) => reshape(index_select(weight, 0, tensor(values, { dtype: 'f32', device })), [batch, length, d])
      const positions = Array.from({ length: batch * length }, (_, i) => i % length)
      let x = norm(add(add(embedding(weights.tokenEmbedding, ids.flat()),
        embedding(weights.positionEmbedding, positions)),
        embedding(weights.typeEmbedding, types.flat())), weights.embeddingNorm, config.epsilon)
      const hidden_states: Tensor[] = [x]
      const mask = reshape(tensor(masks.map(row => row.map(value => 1 - value)), { dtype: 'f32', device }), [batch, 1, 1, length])
      for (let i = 0; i < layers; i++) {
        const block = weights.blocks[i]
        const context = attention(dense(x, block.query), dense(x, block.key), dense(x, block.value), heads, mask)
        x = norm(add(x, dense(context, block.attentionOutput)), block.attentionNorm, config.epsilon)
        x = norm(add(x, dense(erf_gelu(dense(x, block.expand)), block.contract)), block.feedForwardNorm, config.epsilon)
        hidden_states.push(x)
      }
      const valid = reshape(tensor(masks, { dtype: 'f32', device }), [batch, length, 1])
      const mean = div(sum(mul(x, valid), 1), sum(valid, 1))
      const pooled = div(mean, sqrt(sum(mul(mean, mean), 1, true)))
      const pooler = tanh(dense(reshape(contiguous(x.slice([':', '0:1', ':'])), [batch, d]), weights.pooler))
      return { output: x, pooled, pooler, hidden_states }
    })
  }
  return { config, forward }
}
