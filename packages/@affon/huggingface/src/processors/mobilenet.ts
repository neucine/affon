import fs from "std:fs";
import { tensor, type Device } from "affon:compute";
import { resize_rgb } from "./shared/resize-rgb.ts";

/** MobileNetV2's RGB8 bilinear shortest-edge resize, center crop and f32 normalization. */
export function process_mobilenet_image(
  directory: string,
  rgb: number[][][],
  device: Device,
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
  const height = rgb.length,
    width = rgb[0]?.length ?? 0;
  if (
    !height ||
    !width ||
    rgb.some(
      (row) =>
        row.length !== width ||
        row.some(
          (p) =>
            p.length !== 3 ||
            p.some((v) => !Number.isInteger(v) || v < 0 || v > 255),
        ),
    )
  )
    throw Error("Expected a rectangular RGB uint8 image");
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
  rgb = resize_rgb(rgb, rh, rw);
  const top = Math.floor((rh - h) / 2),
    left = Math.floor((rw - w) / 2);
  return tensor(
    [
      Array.from({ length: 3 }, (_, channel) =>
        Array.from({ length: h }, (_, y) =>
          Array.from({ length: w }, (_, x) => {
            const scaled = Math.fround(
              rgb[y + top][x + left][channel] * c.rescale_factor,
            );
            return Math.fround(
              Math.fround(scaled - c.image_mean[channel]) /
                c.image_std[channel],
            );
          }),
        ),
      ),
    ],
    { dtype: "f32", device },
  );
}
