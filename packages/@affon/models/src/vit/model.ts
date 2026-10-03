import { add, cat, contiguous, div, erf, matmul, mean, mul, reshape, slice, softmax, sqrt, sub, transpose } from 'affon:ops'
import { Session, Tensor, program, type Device, type FormalTensor } from 'affon:compute'
import { positive_dimensions, parameter_checks } from '../shared/parameters.ts'
import type { ModelTensor } from '../shared/parameters.ts'

export interface ViTConfig { width: number; innerWidth: number; heads: number; layers: number; imageSize: number; patchSize: number; epsilon: number }
export interface ViTAffineWeights { weight: ModelTensor; bias: ModelTensor }
export interface ViTWeights {
  blocks: { query: ViTAffineWeights; key: ViTAffineWeights; value: ViTAffineWeights; attentionOutput: ViTAffineWeights; attentionNorm: ViTAffineWeights; feedForwardNorm: ViTAffineWeights; expand: ViTAffineWeights; contract: ViTAffineWeights }[]
  classToken: ModelTensor; positionEmbedding: ModelTensor; patchProjection: ViTAffineWeights; finalNorm: ViTAffineWeights; classifier: ViTAffineWeights
}

/** Fixed-resolution, batch-one ViT executed as one reusable Program. */
export function create_vit(config: ViTConfig, weights: ViTWeights, device: Device = 'cpu') {
  const { width, innerWidth, heads, layers, imageSize, patchSize } = config
  positive_dimensions(width, innerWidth, heads, layers, imageSize, patchSize)
  if (width % heads || weights.blocks.length !== layers || imageSize % patchSize || !Number.isFinite(config.epsilon) || config.epsilon <= 0) throw new Error('Invalid ViT dimensions or normalization')
  const labels = weights.classifier?.bias?.shape[0]
  positive_dimensions(labels)
  const check = parameter_checks(device)
  for (const block of weights.blocks) {
    for (const params of [block.query, block.key, block.value, block.attentionOutput]) check.affine(params as any, [width, width])
    check.affine(block.attentionNorm as any, [width]); check.affine(block.feedForwardNorm as any, [width])
    check.affine(block.expand as any, [width, innerWidth]); check.affine(block.contract as any, [innerWidth, width])
  }
  const patchCount = (imageSize / patchSize) ** 2
  check.tensor(weights.classToken as any, [1, 1, width])
  check.tensor(weights.positionEmbedding as any, [1, patchCount + 1, width])
  check.tensor(weights.patchProjection.weight as any, [width, 3, patchSize, patchSize])
  check.tensor(weights.patchProjection.bias as any, [width])
  check.affine(weights.finalNorm as any, [width]); check.affine(weights.classifier as any, [width, labels])

  const session = new Session({ device })
  const values: Record<string, ModelTensor> = {
    class_token: weights.classToken,
    position_embedding: weights.positionEmbedding,
    patch_weight: weights.patchProjection.weight,
    patch_bias: weights.patchProjection.bias,
    final_norm_weight: weights.finalNorm.weight,
    final_norm_bias: weights.finalNorm.bias,
    classifier_weight: weights.classifier.weight,
    classifier_bias: weights.classifier.bias,
  }
  weights.blocks.forEach((block, index) => {
    for (const [part, affine] of Object.entries(block) as [string, ViTAffineWeights][]) {
      values[`block_${index}_${part}_weight`] = affine.weight
      values[`block_${index}_${part}_bias`] = affine.bias
    }
  })
  const source = program('vit', p => {
    const pixels = p.argument('pixels', Tensor.f32([1, 3, imageSize, imageSize]))
    const parameter = (name: string, shape: readonly number[]) => p.parameter(name, Tensor.f32(shape))
    const affine = (name: string, shape: readonly number[]) => ({
      weight: parameter(`${name}_weight`, shape),
      bias: parameter(`${name}_bias`, shape.length === 1 ? shape : [shape[shape.length - 1]]),
    })
    const dense = (x: FormalTensor, params: { weight: FormalTensor; bias: FormalTensor }) => add(matmul(x, params.weight), params.bias)
    const epsilon = p.constant('epsilon', config.epsilon, Tensor.f32([1]))
    const sqrtTwo = p.constant('sqrt_two', Math.sqrt(2), Tensor.f32([1]))
    const half = p.constant('half', 0.5, Tensor.f32([1]))
    const one = p.constant('one', 1, Tensor.f32([1]))
    const norm = (x: FormalTensor, params: { weight: FormalTensor; bias: FormalTensor }) => {
      const centered = sub(x, mean(x, -1, true))
      return add(mul(div(centered, sqrt(add(mean(mul(centered, centered), -1, true), epsilon))), params.weight), params.bias)
    }
    const gelu = (x: FormalTensor) => mul(mul(x, half), add(erf(div(x, sqrtTwo)), one))
    const split = (x: FormalTensor) => transpose(reshape(x, [1, patchCount + 1, heads, width / heads]), [0, 2, 1, 3])
    const attention = (q: FormalTensor, k: FormalTensor, v: FormalTensor) => {
      const scale = p.constant(`attention_scale_${q.id}`, Math.sqrt(width / heads), Tensor.f32([1]))
      return reshape(contiguous(transpose(matmul(softmax(div(matmul(split(q), transpose(split(k), [0, 1, 3, 2])), scale), 3), split(v)), [0, 2, 1, 3])), [1, patchCount + 1, width])
    }
    const patchWeight = contiguous(transpose(reshape(parameter('patch_weight', [width, 3, patchSize, patchSize]), [width, 3 * patchSize * patchSize]), [1, 0]))
    const patches = reshape(contiguous(transpose(reshape(pixels, [1, 3, imageSize / patchSize, patchSize, imageSize / patchSize, patchSize]), [0, 2, 4, 1, 3, 5])), [1, patchCount, 3 * patchSize * patchSize])
    const projected = add(matmul(patches, patchWeight), parameter('patch_bias', [width]))
    let x = add(cat([parameter('class_token', [1, 1, width]), projected], 1), parameter('position_embedding', [1, patchCount + 1, width]))
    const hiddenStates: FormalTensor[] = [x]
    for (let index = 0; index < layers; index++) {
      const prefix = `block_${index}`
      const query = affine(`${prefix}_query`, [width, width])
      const key = affine(`${prefix}_key`, [width, width])
      const value = affine(`${prefix}_value`, [width, width])
      const attentionOutput = affine(`${prefix}_attentionOutput`, [width, width])
      const attentionNorm = affine(`${prefix}_attentionNorm`, [width])
      const feedForwardNorm = affine(`${prefix}_feedForwardNorm`, [width])
      const expand = affine(`${prefix}_expand`, [width, innerWidth])
      const contract = affine(`${prefix}_contract`, [innerWidth, width])
      const input = norm(x, attentionNorm)
      x = add(x, dense(attention(dense(input, query), dense(input, key), dense(input, value)), attentionOutput))
      x = add(x, dense(gelu(dense(norm(x, feedForwardNorm), expand)), contract))
      hiddenStates.push(x)
    }
    const final = norm(x, affine('final_norm', [width]))
    hiddenStates[hiddenStates.length - 1] = final
    const cls = reshape(contiguous(slice(final, [{ start: 0, stop: 1 }, { start: 0, stop: 1 }, { start: 0, stop: width }])), [1, width])
    return [dense(cls, affine('classifier', [width, labels])), ...hiddenStates]
  })
  const state = session.initialize(source, { parameters: values })
  const executable = session.compile(source)

  function forward(pixels: ModelTensor) {
    if (pixels.dtype !== 'f32' || JSON.stringify(pixels.shape) !== JSON.stringify([1, 3, imageSize, imageSize])) throw new Error('Expected fixed-size batch-one RGB ViT input')
    const input = session.tensor(pixels.to_array())
    try {
      const outputs = executable.run({ pixels: input }, state) as Tensor[]
      return { output: outputs[0], hidden_states: outputs.slice(1) }
    } finally {
      input.dispose()
    }
  }
  function dispose() { state.dispose(); session.dispose() }
  return { config, forward, dispose }
}
