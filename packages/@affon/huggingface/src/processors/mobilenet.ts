import fs from "std:fs";
import { Session, type Device } from "affon:compute";
import { resize_rgb } from "./shared/resize-rgb.ts";

/** MobileNetV2's RGB8 bilinear shortest-edge resize, center crop and f32 normalization. */
export function process_mobilenet_image(
  directory: string,
  rgb: number[][][] | number[],
  device: Device,
  flat_shape?: [number, number],
) {
  const raw = JSON.parse(
    fs.readFileSync(`${directory}/preprocessor_config.json`),
  );
  const c = {
    do_resize: true,
    do_center_crop: true,
    do_rescale: true,
    do_normalize: true,
    size: { shortest_edge: 256 },
    crop_size: { height: 224, width: 224 },
    resample: 2,
    rescale_factor: 1 / 255,
    image_mean: [0.5, 0.5, 0.5],
    image_std: [0.5, 0.5, 0.5],
    ...raw,
  };
  const height = flat_shape?.[0] ?? rgb.length,
    width = flat_shape?.[1] ?? (rgb as number[][][])[0]?.length ?? 0;
  if (!height || !width || (flat_shape !== undefined && rgb.length !== height * width * 3))
    throw Error("Expected a rectangular RGB uint8 image");
  let source: number[][][]
  if (flat_shape) {
    const flat = rgb as number[]
    source = Array.from({ length: height }, (_, y) => Array.from({ length: width }, (_, x) => flat.slice((y * width + x) * 3, (y * width + x + 1) * 3)))
  } else source = rgb as number[][][]
  if (source.some(row => row.length !== width || row.some(pixel => pixel.length !== 3 || pixel.some(value => !Number.isInteger(value) || value < 0 || value > 255)))) throw Error("Expected a rectangular RGB uint8 image");
  const short = c.size.shortest_edge,
    h = c.crop_size.height,
    w = c.crop_size.width;
  if (
    ![short, h, w].every((n) => Number.isInteger(n) && n > 0) ||
    c.resample !== 2 ||
    !c.do_resize ||
    !c.do_center_crop ||
    !c.do_rescale ||
    !c.do_normalize ||
    !Number.isFinite(c.rescale_factor) ||
    !Array.isArray(c.image_mean) ||
    c.image_mean.length !== 3 ||
    c.image_mean.some((v: number) => !Number.isFinite(v)) ||
    !Array.isArray(c.image_std) ||
    c.image_std.length !== 3 ||
    c.image_std.some((v: number) => !Number.isFinite(v) || v <= 0)
  )
    throw Error("Unsupported MobileNetV2 processor configuration");
  const rh = height <= width ? short : Math.floor((short * height) / width);
  const rw = width <= height ? short : Math.floor((short * width) / height);
  // This subset deliberately rejects crops requiring padding.
  if (h > rh || w > rw)
    throw Error("MobileNetV2 crop must fit the resized image");
  source = resize_rgb(source, rh, rw);
  const top = Math.floor((rh - h) / 2),
    left = Math.floor((rw - w) / 2);
  const planes: number[][][] = new Array(3);
  for (let channel = 0; channel < 3; channel++) {
    const plane = (planes[channel] = new Array(h));
    const mean = c.image_mean[channel],
      std = c.image_std[channel];
    for (let y = 0; y < h; y++) {
      const row = (plane[y] = new Array(w));
      for (let x = 0; x < w; x++) {
        const scaled = Math.fround(
          source[y + top][x + left][channel] * c.rescale_factor,
        );
        row[x] = Math.fround(Math.fround(scaled - mean) / std);
      }
    }
  }
  const session = new Session({ device });
  const result = session.tensor([planes]);
  session.dispose();
  return result;
}
