import { create_vit, type ViTWeights } from '../../../models/src/vit/index.ts'
import fs from 'std:fs'
import type { Device } from 'affon:compute'
import { prepare_encoder_checkpoint } from './encoder-checkpoint.ts'

/** Load an f32 fixed-size RGB ViT classifier.
 * @param directory Prepared local HF artifact directory.
 * @param device Execution device.
 * @returns Forward over batch-one NCHW pixels, returning logits and hidden states.
 * @remarks Strict hidden-state parity remains unresolved for the audited checkpoint.
 */
export function load_vit(directory: string, device: Device) {
  const config = JSON.parse(fs.readFileSync(`${directory}/config.json`))
  if (config.model_type !== 'vit' || config.hidden_act !== 'gelu' || config.qkv_bias === false || config.num_channels !== 3) throw new Error('HF adapter supports RGB ViT with GELU and biased QKV')
  const { hidden_size: d, intermediate_size: inner, num_attention_heads: heads, num_hidden_layers: layers, image_size: size, patch_size: patch } = config
  const labels = Object.keys(config.id2label ?? {}).length || config.num_labels
  for (const value of [d, inner, heads, layers, size, patch, labels]) {
    if (!Number.isInteger(value) || value <= 0) throw new Error('Invalid ViT dimension')
  }
  if (d % heads || size % patch || !Number.isFinite(config.layer_norm_eps) || config.layer_norm_eps <= 0) throw new Error('Invalid ViT head/patch/norm configuration')
  const count = (size / patch) ** 2
  const ops = prepare_encoder_checkpoint(directory, device)
  const { weights, require_weight: requireWeight } = ops
  const requireNorm = (prefix: string) => { requireWeight(`${prefix}.weight`, [d]); requireWeight(`${prefix}.bias`, [d]) }
  const requireDense = (prefix: string, input: number, output: number) => {
    requireWeight(`${prefix}.weight`, [output, input], true); requireWeight(`${prefix}.bias`, [output])
  }
  requireWeight('vit.embeddings.cls_token', [1, 1, d])
  requireWeight('vit.embeddings.position_embeddings', [1, count + 1, d])
  requireWeight('vit.embeddings.patch_embeddings.projection.weight', [d, 3, patch, patch])
  requireWeight('vit.embeddings.patch_embeddings.projection.bias', [d])
  for (let i = 0; i < layers; i++) {
    const p = `vit.encoder.layer.${i}`
    requireNorm(`${p}.layernorm_before`); requireNorm(`${p}.layernorm_after`)
    for (const name of ['query', 'key', 'value']) requireDense(`${p}.attention.attention.${name}`, d, d)
    requireDense(`${p}.attention.output.dense`, d, d)
    requireDense(`${p}.intermediate.dense`, d, inner)
    requireDense(`${p}.output.dense`, inner, d)
  }
  requireNorm('vit.layernorm')
  requireDense('classifier', d, labels)
  ops.finish()
  const { affine } = ops
  const parameters: ViTWeights = {
    classToken: weights['vit.embeddings.cls_token'], positionEmbedding: weights['vit.embeddings.position_embeddings'], patchProjection: affine('vit.embeddings.patch_embeddings.projection'), finalNorm: affine('vit.layernorm'), classifier: affine('classifier'),
    blocks: Array.from({ length: layers }, (_, i) => {
      const p = `vit.encoder.layer.${i}`
      return { query: affine(`${p}.attention.attention.query`), key: affine(`${p}.attention.attention.key`), value: affine(`${p}.attention.attention.value`), attentionOutput: affine(`${p}.attention.output.dense`), attentionNorm: affine(`${p}.layernorm_before`), feedForwardNorm: affine(`${p}.layernorm_after`), expand: affine(`${p}.intermediate.dense`), contract: affine(`${p}.output.dense`) }
    }),
  }
  const model = create_vit({ width: config.hidden_size, innerWidth: config.intermediate_size, heads: config.num_attention_heads, layers: config.num_hidden_layers, imageSize: config.image_size, patchSize: config.patch_size, epsilon: config.layer_norm_eps }, parameters, device)
  return { ...model, config }
}
