import fs from 'std:fs'
import { Session } from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'
import { resize_rgb } from './shared/resize-rgb.ts'

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
  const session = new Session({ device })
  const result = session.tensor(pixels) as Tensor
  session.dispose()
  return result
}
