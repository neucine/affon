import fs from 'std:fs'
import { add, cat, contiguous, matmul, no_grad, permute, reshape, tensor, transpose } from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'
import { encoder_ops, erf_gelu } from './encoder-ops.ts'
import { resize_rgb } from './resize-rgb.ts'

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
  const ops = encoder_ops(directory, device)
  const { weights, require_weight: requireWeight, dense, norm, attention } = ops
  const requireNorm = (prefix: string) => { requireWeight(`${prefix}.weight`, [d]); requireWeight(`${prefix}.bias`, [d]) }
  const requireDense = (prefix: string, input: number, output: number) => {
    requireWeight(`${prefix}.weight`, [output, input], true); requireWeight(`${prefix}.bias`, [output])
  }
  requireWeight('vit.embeddings.cls_token', [1, 1, d])
  requireWeight('vit.embeddings.position_embeddings', [1, count + 1, d])
  requireWeight('vit.embeddings.patch_embeddings.projection.weight', [d, 3, patch, patch])
  requireWeight('vit.embeddings.patch_embeddings.projection.bias', [d])
  const patchWeight = contiguous(transpose(reshape(weights['vit.embeddings.patch_embeddings.projection.weight'], [d, 3 * patch * patch]), 0, 1))
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
  function forward(pixels: Tensor) {
    if (pixels.dtype !== 'f32' || JSON.stringify(pixels.shape) !== JSON.stringify([1, 3, size, size])) throw new Error('Expected fixed-size batch-one RGB ViT input')
    return no_grad(() => {
      const image = pixels.device === device ? pixels : pixels.to(device)
      // Nonoverlapping Conv2d(kernel=stride=patch) expressed as patches + matmul.
      const blocks = reshape(image, [1, 3, size / patch, patch, size / patch, patch])
      const patches = reshape(contiguous(permute(blocks, [0, 2, 4, 1, 3, 5])), [1, count, 3 * patch * patch])
      let x = add(cat([weights['vit.embeddings.cls_token'], add(matmul(patches, patchWeight), weights['vit.embeddings.patch_embeddings.projection.bias'])], 1), weights['vit.embeddings.position_embeddings'])
      const hidden_states: Tensor[] = [x]
      for (let i = 0; i < layers; i++) {
        const p = `vit.encoder.layer.${i}`
        const input = norm(x, `${p}.layernorm_before`, config.layer_norm_eps)
        const context = attention(dense(input, `${p}.attention.attention.query`), dense(input, `${p}.attention.attention.key`), dense(input, `${p}.attention.attention.value`), heads)
        x = add(x, dense(context, `${p}.attention.output.dense`))
        x = add(x, dense(erf_gelu(dense(norm(x, `${p}.layernorm_after`, config.layer_norm_eps), `${p}.intermediate.dense`)), `${p}.output.dense`))
        hidden_states.push(x)
      }
      const final = norm(x, 'vit.layernorm', config.layer_norm_eps)
      // Transformers 4.56's output-capture wrapper replaces the last recorded
      // block output with the model's normalized last_hidden_state.
      hidden_states[hidden_states.length - 1] = final
      return { output: dense(reshape(contiguous(final.slice([0, '0:1', ':'])), [1, d]), 'classifier'), hidden_states }
    })
  }
  return { config, forward }
}

/** Prepare RGB8 arrays using the local ViT bilinear resize/rescale/normalize config.
 * @param directory Directory containing preprocessor_config.json.
 * @param rgb Rectangular height × width × three-channel uint8 values.
 * @param device Output tensor device.
 * @returns Batch-one f32 NCHW pixels. Does not decode image files.
 */
export function process_rgb_image(directory: string, rgb: number[][][], device: Device): Tensor {
  // Legacy ViTFeatureExtractor artifacts omit defaults and use a scalar size.
  const raw = JSON.parse(fs.readFileSync(`${directory}/preprocessor_config.json`))
  const config = { do_resize: true, resample: 2, do_rescale: true, rescale_factor: 1 / 255,
    do_normalize: true, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5], ...raw,
    size: typeof raw.size === 'number' ? { height: raw.size, width: raw.size } : raw.size ?? { height: 224, width: 224 } }
  let height = rgb.length, width = rgb[0]?.length ?? 0
  if (!height || !width || rgb.some(row => row.length !== width || row.some(pixel => pixel.length !== 3 || pixel.some(value => !Number.isInteger(value) || value < 0 || value > 255)))) throw new Error('Expected a rectangular RGB uint8 image')
  if (config.do_resize) {
    if (config.resample !== 2) throw new Error('Only bilinear RGB resizing is supported')
    rgb = resize_rgb(rgb, config.size.height, config.size.width)
    height = rgb.length; width = rgb[0].length
  }
  if (!config.do_rescale || !config.do_normalize || !Number.isFinite(config.rescale_factor)
    || !Array.isArray(config.image_mean) || config.image_mean.length !== 3 || config.image_mean.some((x: number) => !Number.isFinite(x))
    || !Array.isArray(config.image_std) || config.image_std.length !== 3 || config.image_std.some((x: number) => !Number.isFinite(x) || x <= 0)) throw new Error('Unsupported ViT processor configuration')
  const pixels: number[][][][] = [Array.from({ length: 3 }, (_, c) => Array.from({ length: height }, (_, y) => Array.from({ length: width }, (_, x) => {
    const rescaled = Math.fround(rgb[y][x][c] * config.rescale_factor)
    return Math.fround(Math.fround(rescaled - config.image_mean[c]) / config.image_std[c])
  })))]
  return tensor(pixels, { dtype: 'f32', device })
}
