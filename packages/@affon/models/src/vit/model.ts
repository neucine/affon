import { positive_dimensions, parameter_checks } from '../shared/parameters.ts'
import { add, cat, contiguous, matmul, no_grad, permute, reshape, tensor, transpose } from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'
import { encoder_ops, erf_gelu } from '../shared/encoder.ts'

import type { AffineWeights } from '../shared/encoder.ts'

export interface ViTConfig { width: number; innerWidth: number; heads: number; layers: number; imageSize: number; patchSize: number; epsilon: number }
export interface ViTWeights {
  blocks: { query: AffineWeights; key: AffineWeights; value: AffineWeights; attentionOutput: AffineWeights; attentionNorm: AffineWeights; feedForwardNorm: AffineWeights; expand: AffineWeights; contract: AffineWeights }[]
  classToken: Tensor; positionEmbedding: Tensor; patchProjection: AffineWeights; finalNorm: AffineWeights; classifier: AffineWeights
}

/** Construct f32 ViT from normalized configuration and prepared tensors.
 * Affine weights use [input, output] layout; normalization uses vectors.
 * Tensors must be on the requested device and remain alive with the model.
 */
export function create_vit(config: ViTConfig, weights: ViTWeights, device: Device = 'cpu') {
  if (!Number.isInteger(config.width) || config.width <= 0 || !Number.isInteger(config.heads) || config.heads <= 0 || config.width % config.heads || !Number.isInteger(config.layers) || config.layers <= 0 || weights.blocks.length !== config.layers || !Number.isFinite(config.epsilon) || config.epsilon <= 0) throw new Error('Invalid model dimensions or normalization')
  const check = parameter_checks(device)
  const width = config.width
  positive_dimensions(config.innerWidth)
  for (const block of weights.blocks) {
    for (const params of [block.query, block.key, block.value, block.attentionOutput]) check.affine(params, [width, width])
    check.affine(block.attentionNorm, [width]); check.affine(block.feedForwardNorm, [width])
    check.affine(block.expand, [width, config.innerWidth]); check.affine(block.contract, [config.innerWidth, width])
  }
  positive_dimensions(config.imageSize, config.patchSize)
  if (config.imageSize % config.patchSize) throw new Error('Image size must be divisible by patch size')
  const labels = weights.classifier?.bias?.shape[0]
  positive_dimensions(labels)
  check.tensor(weights.classToken, [1, 1, width])
  check.tensor(weights.positionEmbedding, [1, (config.imageSize / config.patchSize) ** 2 + 1, width])
  check.tensor(weights.patchProjection.weight, [width, 3, config.patchSize, config.patchSize])
  check.tensor(weights.patchProjection.bias, [width])
  check.affine(weights.finalNorm, [width]); check.affine(weights.classifier, [width, labels])
  const { width: d, heads, layers } = config
  const { dense, norm, attention } = encoder_ops(device)
  const { imageSize: size, patchSize: patch } = config
  const count = (size / patch) ** 2
  const patchWeight = contiguous(transpose(reshape(weights.patchProjection.weight, [d, 3 * patch * patch]), 0, 1))
  function forward(pixels: Tensor) {
    if (pixels.dtype !== 'f32' || JSON.stringify(pixels.shape) !== JSON.stringify([1, 3, size, size])) throw new Error('Expected fixed-size batch-one RGB ViT input')
    return no_grad(() => {
      const image = pixels.device === device ? pixels : pixels.to(device)
      // Nonoverlapping Conv2d(kernel=stride=patch) expressed as patches + matmul.
      const blocks = reshape(image, [1, 3, size / patch, patch, size / patch, patch])
      const patches = reshape(contiguous(permute(blocks, [0, 2, 4, 1, 3, 5])), [1, count, 3 * patch * patch])
      let x = add(cat([weights.classToken, add(matmul(patches, patchWeight), weights.patchProjection.bias)], 1), weights.positionEmbedding)
      const hidden_states: Tensor[] = [x]
      for (let i = 0; i < layers; i++) {
        const block = weights.blocks[i]
        const input = norm(x, block.attentionNorm, config.epsilon)
        const context = attention(dense(input, block.query), dense(input, block.key), dense(input, block.value), heads)
        x = add(x, dense(context, block.attentionOutput))
        x = add(x, dense(erf_gelu(dense(norm(x, block.feedForwardNorm, config.epsilon), block.expand)), block.contract))
        hidden_states.push(x)
      }
      const final = norm(x, weights.finalNorm, config.epsilon)
      // Transformers 4.56's output-capture wrapper replaces the last recorded
      // block output with the model's normalized last_hidden_state.
      hidden_states[hidden_states.length - 1] = final
      return { output: dense(reshape(contiguous(final.slice([0, '0:1', ':'])), [1, d]), weights.classifier), hidden_states }
    })
  }
  return { config, forward }
}
