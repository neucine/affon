import { add, contiguous, div, embedding, gelu, layer_norm, masked_fill, matmul, mul, reshape, slice, softmax, transpose } from 'affon:ops'
import { Tensor, program, type FormalTensor } from 'affon:compute'
import { positive_dimensions, parameter_checks } from '../shared/parameters.ts'
import type { ModelAffineWeights, ModelTensor } from '../shared/parameters.ts'

export interface GPT2Config { width: number; heads: number; layers: number; contextLength: number; vocabSize: number; epsilon: number; eosTokenId?: number; scaleAttention?: boolean }
export interface GPT2Weights {
  tokenEmbedding: ModelTensor; positionEmbedding: ModelTensor; finalNorm: ModelAffineWeights
  blocks: { attentionNorm: ModelAffineWeights; feedForwardNorm: ModelAffineWeights; qkv: ModelAffineWeights; attentionOutput: ModelAffineWeights; expand: ModelAffineWeights; contract: ModelAffineWeights }[]
}

/** Author shape-specialized f32 GPT-2 Programs and their parameter initializer. */
export function create_gpt2(config: GPT2Config, weights: GPT2Weights) {
  if (!Number.isInteger(config.width) || config.width <= 0 || !Number.isInteger(config.heads) || config.heads <= 0 || config.width % config.heads || !Number.isInteger(config.layers) || config.layers <= 0 || weights.blocks.length !== config.layers || !Number.isFinite(config.epsilon) || config.epsilon <= 0) throw new Error('Invalid model dimensions or normalization')
  const check = parameter_checks()
  const width = config.width
  positive_dimensions(config.contextLength, config.vocabSize)
  if (config.eosTokenId !== undefined && (!Number.isInteger(config.eosTokenId) || config.eosTokenId < 0 || config.eosTokenId >= config.vocabSize)) throw new Error('Invalid EOS token')
  check.tensor(weights.tokenEmbedding, [config.vocabSize, width])
  check.tensor(weights.positionEmbedding, [config.contextLength, width])
  check.affine(weights.finalNorm, [width])
  for (const block of weights.blocks) {
    const inner = block.expand?.bias?.shape[0]
    positive_dimensions(inner)
    check.affine(block.attentionNorm, [width]); check.affine(block.feedForwardNorm, [width])
    check.affine(block.qkv, [width, 3 * width]); check.affine(block.attentionOutput, [width, width])
    check.affine(block.expand, [width, inner]); check.affine(block.contract, [inner, width])
  }

  const parameterValues: Record<string, ModelTensor> = {
    token_embedding: weights.tokenEmbedding,
    position_embedding: weights.positionEmbedding,
    final_norm_weight: weights.finalNorm.weight,
    final_norm_bias: weights.finalNorm.bias,
  }
  weights.blocks.forEach((block, index) => {
    for (const [part, affine] of Object.entries(block) as [string, ModelAffineWeights][]) {
      parameterValues[`block_${index}_${part}_weight`] = affine.weight
      parameterValues[`block_${index}_${part}_bias`] = affine.bias
    }
  })

  const parameter = (p: any, name: string, shape: readonly number[]) => p.parameter(name, Tensor.f32(shape)) as FormalTensor
  const affine = (p: any, name: string, shape: readonly number[]) => ({
    weight: parameter(p, `${name}_weight`, shape),
    bias: parameter(p, `${name}_bias`, shape.length === 1 ? shape : [shape[shape.length - 1]]),
  })

  function forward(length: number, outputStart = 0) {
    positive_dimensions(length)
    if (length > config.contextLength || !Number.isInteger(outputStart) || outputStart < 0 || outputStart >= length) throw new Error('Invalid GPT-2 sequence or output window')
    const source = program('gpt2', p => {
      const ids = p.argument('ids', Tensor.i64([length]))
      const positions = p.argument('positions', Tensor.i64([length]))
      const mask = p.argument('mask', Tensor.i64([1, 1, length, length]))
      const tokenEmbedding = parameter(p, 'token_embedding', [config.vocabSize, width])
      const positionEmbedding = parameter(p, 'position_embedding', [config.contextLength, width])
      const finalNorm = affine(p, 'final_norm', [width])
      const scalar = (name: string, value: number) => p.constant(name, value, Tensor.f32([1]))
      const dense = (x: FormalTensor, params: { weight: FormalTensor; bias: FormalTensor }) => add(matmul(x, params.weight), params.bias)
      const norm = (x: FormalTensor, params: { weight: FormalTensor; bias: FormalTensor }) => add(mul(layer_norm(x, 2, config.epsilon), params.weight), params.bias)
      let x = reshape(add(embedding(tokenEmbedding, ids), embedding(positionEmbedding, positions)), [1, length, width])
      const hiddenStates: FormalTensor[] = []
      const headWidth = width / config.heads
      const full = (start: number, stop: number) => [{ start: 0, stop: 1 }, { start: 0, stop: length }, { start, stop }]
      const splitHeads = (value: FormalTensor) => transpose(reshape(value, [1, length, config.heads, headWidth]), [0, 2, 1, 3])
      for (let index = 0; index < config.layers; index++) {
        hiddenStates.push(x)
        const prefix = `block_${index}`
        const attentionNorm = affine(p, `${prefix}_attentionNorm`, [width])
        const feedForwardNorm = affine(p, `${prefix}_feedForwardNorm`, [width])
        const qkvWeights = affine(p, `${prefix}_qkv`, [width, 3 * width])
        const attentionOutput = affine(p, `${prefix}_attentionOutput`, [width, width])
        const expand = affine(p, `${prefix}_expand`, [width, weights.blocks[index].expand.bias.shape[0]])
        const contract = affine(p, `${prefix}_contract`, [weights.blocks[index].expand.bias.shape[0], width])
        const qkv = dense(norm(x, attentionNorm), qkvWeights)
        const q = splitHeads(contiguous(slice(qkv, full(0, width))))
        const k = splitHeads(contiguous(slice(qkv, full(width, 2 * width))))
        const v = splitHeads(contiguous(slice(qkv, full(2 * width, 3 * width))))
        let scores = matmul(q, transpose(k, [0, 1, 3, 2]))
        if (config.scaleAttention !== false) scores = div(scores, scalar(`attention_scale_${index}`, Math.sqrt(headWidth)))
        const attention = reshape(contiguous(transpose(matmul(softmax(masked_fill(scores, mask, -3.4028234663852886e38), 3), v), [0, 2, 1, 3])), [1, length, width])
        x = add(x, contiguous(dense(attention, attentionOutput)))
        x = add(x, contiguous(dense(gelu(dense(norm(x, feedForwardNorm), expand)), contract)))
      }
      x = norm(x, finalNorm)
      hiddenStates.push(x)
      const suffix = (value: FormalTensor) => contiguous(slice(value, [{ start: 0, stop: 1 }, { start: outputStart, stop: length }, { start: 0, stop: width }]))
      const logits = matmul(suffix(x), transpose(tokenEmbedding, [1, 0]))
      return [logits, ...hiddenStates.map(suffix)]
    })
    return source
  }
  return { config, forward, parameters: parameterValues }
}
